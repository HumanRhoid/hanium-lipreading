"""Dashboard persistence behavior against real PostgreSQL."""

import asyncio
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from src.backend.auth.adapters.repository import User
from src.backend.dashboard.adapters.repository import (
    RepositoryTransitionConflictError,
    SQLAlchemyDashboardRepository,
)
from src.backend.dashboard.models import (
    CommunicationRequest,
    PatientProfile,
    RequestEvent,
    RequestIdempotency,
    StaffWardAccess,
    Ward,
)
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
    Utterance,
)
from src.backend.recognition.domain import (
    PhraseCategory,
    Prediction,
)

pytestmark = pytest.mark.integration


async def _create_user(
    postgres_session_factory,
    *,
    display_name: str,
    role: str = "PATIENT",
) -> int:
    user = User(
        username=f"dashboard-{uuid4().hex[:16]}",
        password_hash="test-password-hash",
        display_name=display_name,
        role=role,
    )

    async with postgres_session_factory.begin() as session:
        session.add(user)
        await session.flush()
        return user.user_id


async def _create_patient(
    postgres_session_factory,
    *,
    user_id: int,
    ward_code: str = "WARD-3",
) -> int:
    async with postgres_session_factory.begin() as session:
        ward = Ward(
            ward_code=ward_code,
            ward_name="3병동",
        )

        session.add(ward)
        await session.flush()

        patient = PatientProfile(
            user_id=user_id,
            patient_code=f"P-{uuid4().hex[:8]}",
            display_name="테스트환자",
            ward_code=ward_code,
            room_number="302",
            admitted_on=date(2026, 9, 1),
            communication_status="VOICE_DIFFICULT",
            assistive_method="LIP_READING",
            notes=None,
        )

        session.add(patient)
        await session.flush()

        return patient.patient_id


async def _create_upload_utterance(
    postgres_session_factory,
    *,
    user_id: int,
) -> int:
    async with postgres_session_factory.begin() as session:
        utterance = Utterance(
            user_id=user_id,
        )

        session.add(utterance)
        await session.flush()

        return utterance.utt_id


async def test_inference_result_creates_one_request_and_event(
    postgres_session_factory,
):
    user_id = await _create_user(
        postgres_session_factory,
        display_name="환자",
    )

    patient_id = await _create_patient(
        postgres_session_factory,
        user_id=user_id,
    )

    utterance_id = await _create_upload_utterance(
        postgres_session_factory,
        user_id=user_id,
    )

    repository = SQLAlchemyRecognitionRepository(
        postgres_session_factory
    )

    await repository.sync_phrases(
        [
            (
                "SYMPTOM_BREATHING_DIFFICULTY",
                "숨 쉬기 힘들어요",
                PhraseCategory.ETC,
            )
        ]
    )

    prediction = Prediction(
        text="숨 쉬기 힘들어요",
        confidence=0.93,
        phrase_code=(
            "SYMPTOM_BREATHING_DIFFICULTY"
        ),
    )

    first = await repository.save_inference_result(
        utterance_id=utterance_id,
        prediction=prediction,
        model_version="integration-test",
    )

    second = await repository.save_inference_result(
        utterance_id=utterance_id,
        prediction=prediction,
        model_version="integration-test",
    )

    assert first.utterance_id == utterance_id
    assert second.utterance_id == utterance_id

    async with postgres_session_factory() as session:
        requests = list(
            await session.scalars(
                select(CommunicationRequest).where(
                    CommunicationRequest.utterance_id
                    == utterance_id
                )
            )
        )

        events = list(
            await session.scalars(
                select(RequestEvent)
            )
        )

    assert len(requests) == 1

    request = requests[0]

    assert request.patient_id == patient_id
    assert request.ward_code == "WARD-3"
    assert request.room_number == "302"
    assert (
        request.phrase_code
        == "SYMPTOM_BREATHING_DIFFICULTY"
    )
    assert request.text == "숨 쉬기 힘들어요"
    assert request.category == "ETC"
    assert request.priority == "CRITICAL"
    assert request.status == "NEW"
    assert float(request.confidence) == pytest.approx(
        0.93
    )

    requested_events = [
        event
        for event in events
        if (
            event.request_id == request.request_id
            and event.event_type == "REQUESTED"
        )
    ]

    assert len(requested_events) == 1
    assert requested_events[0].actor_user_id is None

    assert (
        requested_events[0].occurred_at
        == request.requested_at
    )


async def test_non_patient_inference_keeps_legacy_flow(
    postgres_session_factory,
):
    user_id = await _create_user(
        postgres_session_factory,
        display_name="일반 사용자",
    )

    utterance_id = await _create_upload_utterance(
        postgres_session_factory,
        user_id=user_id,
    )

    repository = SQLAlchemyRecognitionRepository(
        postgres_session_factory
    )

    result = await repository.save_inference_result(
        utterance_id=utterance_id,
        prediction=Prediction(
            text="테스트",
            confidence=0.9,
            phrase_code=None,
        ),
        model_version="integration-test",
    )

    assert result.text == "테스트"

    async with postgres_session_factory() as session:
        request_count = await session.scalar(
            select(
                func.count(
                    CommunicationRequest.request_id
                )
            )
        )

    assert request_count == 0


async def test_dashboard_state_changes_are_idempotent_in_postgres(
    postgres_session_factory,
):
    patient_user_id = await _create_user(
        postgres_session_factory,
        display_name="환자",
    )

    patient_id = await _create_patient(
        postgres_session_factory,
        user_id=patient_user_id,
    )

    staff_user_id = await _create_user(
        postgres_session_factory,
        display_name="간호사",
        role="STAFF",
    )

    utterance_id = await _create_upload_utterance(
        postgres_session_factory,
        user_id=patient_user_id,
    )

    requested_at = datetime(
        2026,
        9,
        3,
        1,
        0,
        tzinfo=UTC,
    )

    async with postgres_session_factory.begin() as session:
        session.add(
            StaffWardAccess(
                staff_user_id=staff_user_id,
                ward_code="WARD-3",
            )
        )

        request = CommunicationRequest(
            utterance_id=utterance_id,
            patient_id=patient_id,
            ward_code="WARD-3",
            room_number="302",
            phrase_code=None,
            text="테스트 요청",
            category="ETC",
            confidence=None,
            priority="NORMAL",
            status="NEW",
            requested_at=requested_at,
        )

        session.add(request)
        await session.flush()

        session.add(
            RequestEvent(
                request_id=request.request_id,
                event_type="REQUESTED",
                actor_user_id=None,
                occurred_at=requested_at,
                note=None,
            )
        )

        await session.flush()
        request_id = request.request_id

    repository = SQLAlchemyDashboardRepository(
        postgres_session_factory
    )

    acknowledged_at = datetime(
        2026,
        9,
        3,
        1,
        1,
        tzinfo=UTC,
    )

    first_ack = await repository.acknowledge_request(
        staff_user_id=staff_user_id,
        request_id=request_id,
        idempotency_key="ack-key",
        request_fingerprint="a" * 64,
        note="확인 중",
        occurred_at=acknowledged_at,
    )

    replay_ack = await repository.acknowledge_request(
        staff_user_id=staff_user_id,
        request_id=request_id,
        idempotency_key="ack-key",
        request_fingerprint="a" * 64,
        note="확인 중",
        occurred_at=acknowledged_at,
    )

    assert first_ack.summary.status == "ACKNOWLEDGED"
    assert replay_ack.summary.status == "ACKNOWLEDGED"

    completed_at = datetime(
        2026,
        9,
        3,
        1,
        2,
        tzinfo=UTC,
    )

    first_complete = await repository.complete_request(
        staff_user_id=staff_user_id,
        request_id=request_id,
        idempotency_key="complete-key",
        request_fingerprint="b" * 64,
        resolution_note="처리 완료",
        occurred_at=completed_at,
    )

    replay_complete = await repository.complete_request(
        staff_user_id=staff_user_id,
        request_id=request_id,
        idempotency_key="complete-key",
        request_fingerprint="b" * 64,
        resolution_note="처리 완료",
        occurred_at=completed_at,
    )

    assert first_complete.summary.status == "COMPLETED"
    assert replay_complete.summary.status == "COMPLETED"

    async with postgres_session_factory() as session:
        request = await session.get(
            CommunicationRequest,
            request_id,
        )

        event_types = list(
            await session.scalars(
                select(RequestEvent.event_type)
                .where(
                    RequestEvent.request_id
                    == request_id
                )
                .order_by(RequestEvent.event_id)
            )
        )

        idempotency_count = await session.scalar(
            select(
                func.count(
                    RequestIdempotency.idempotency_id
                )
            ).where(
                RequestIdempotency.request_id
                == request_id
            )
        )

    assert request is not None
    assert request.status == "COMPLETED"
    assert request.acknowledged_by == staff_user_id
    assert request.completed_by == staff_user_id
    assert request.resolution_note == "처리 완료"

    assert event_types == [
        "REQUESTED",
        "ACKNOWLEDGED",
        "COMPLETED",
    ]

    assert idempotency_count == 2



async def test_concurrent_acknowledge_only_first_staff_wins(
    postgres_session_factory,
):
    patient_user_id = await _create_user(
        postgres_session_factory,
        display_name="??",
    )

    patient_id = await _create_patient(
        postgres_session_factory,
        user_id=patient_user_id,
    )

    staff_one_id = await _create_user(
        postgres_session_factory,
        display_name="???1",
        role="STAFF",
    )

    staff_two_id = await _create_user(
        postgres_session_factory,
        display_name="???2",
        role="STAFF",
    )

    utterance_id = await _create_upload_utterance(
        postgres_session_factory,
        user_id=patient_user_id,
    )

    requested_at = datetime(
        2026,
        9,
        3,
        2,
        0,
        tzinfo=UTC,
    )

    async with postgres_session_factory.begin() as session:
        session.add_all(
            [
                StaffWardAccess(
                    staff_user_id=staff_one_id,
                    ward_code="WARD-3",
                ),
                StaffWardAccess(
                    staff_user_id=staff_two_id,
                    ward_code="WARD-3",
                ),
            ]
        )

        request = CommunicationRequest(
            utterance_id=utterance_id,
            patient_id=patient_id,
            ward_code="WARD-3",
            room_number="302",
            phrase_code="REQUEST_HELP",
            text="?????",
            category="REQUEST",
            confidence=None,
            priority="HIGH",
            status="NEW",
            requested_at=requested_at,
        )

        session.add(request)
        await session.flush()

        request_id = request.request_id

        session.add(
            RequestEvent(
                request_id=request_id,
                event_type="REQUESTED",
                actor_user_id=None,
                occurred_at=requested_at,
                note=None,
            )
        )

    repository = SQLAlchemyDashboardRepository(
        postgres_session_factory
    )

    acknowledged_at = datetime(
        2026,
        9,
        3,
        2,
        1,
        tzinfo=UTC,
    )

    async def attempt_acknowledge(
        *,
        staff_user_id: int,
        idempotency_key: str,
    ):
        try:
            detail = await repository.acknowledge_request(
                staff_user_id=staff_user_id,
                request_id=request_id,
                idempotency_key=idempotency_key,
                request_fingerprint=(
                    str(staff_user_id).zfill(64)[-64:]
                ),
                note=None,
                occurred_at=acknowledged_at,
            )

            return (
                "success",
                staff_user_id,
                detail,
            )

        except RepositoryTransitionConflictError as exc:
            return (
                "conflict",
                staff_user_id,
                exc,
            )

    results = await asyncio.gather(
        attempt_acknowledge(
            staff_user_id=staff_one_id,
            idempotency_key="concurrent-ack-1",
        ),
        attempt_acknowledge(
            staff_user_id=staff_two_id,
            idempotency_key="concurrent-ack-2",
        ),
    )

    statuses = sorted(
        result[0]
        for result in results
    )

    assert statuses == [
        "conflict",
        "success",
    ]

    winner_id = next(
        staff_user_id
        for status, staff_user_id, _value in results
        if status == "success"
    )

    loser_id = next(
        staff_user_id
        for status, staff_user_id, _value in results
        if status == "conflict"
    )

    assert winner_id != loser_id

    async with postgres_session_factory() as session:
        stored_request = await session.get(
            CommunicationRequest,
            request_id,
        )

        acknowledge_events = list(
            await session.scalars(
                select(RequestEvent)
                .where(
                    RequestEvent.request_id
                    == request_id,
                    RequestEvent.event_type
                    == "ACKNOWLEDGED",
                )
            )
        )

        idempotency_rows = list(
            await session.scalars(
                select(RequestIdempotency)
                .where(
                    RequestIdempotency.request_id
                    == request_id,
                    RequestIdempotency.operation
                    == "ACKNOWLEDGE",
                )
            )
        )

    assert stored_request is not None
    assert stored_request.status == "ACKNOWLEDGED"

    # The staff member who acquired the row lock first
    # remains the sole acknowledged_by value.
    assert stored_request.acknowledged_by == winner_id
    assert stored_request.acknowledged_at == acknowledged_at

    # Losing concurrent request must not overwrite the winner
    # or produce duplicate audit/idempotency records.
    assert len(acknowledge_events) == 1
    assert (
        acknowledge_events[0].actor_user_id
        == winner_id
    )

    assert len(idempotency_rows) == 1
    assert (
        idempotency_rows[0].actor_user_id
        == winner_id
    )
