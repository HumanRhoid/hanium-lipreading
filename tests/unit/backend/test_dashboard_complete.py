"""communication request 완료 처리 계약을 검증한다."""

from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.dashboard.adapters.repository import (
    RepositoryIdempotencyConflictError,
    RepositoryRequestNotFoundError,
    RepositoryTransitionConflictError,
    RequestDetailRecord,
    RequestEventRecord,
    RequestSummaryRecord,
)
from src.backend.dashboard.api import (
    DashboardAPIError,
    dashboard_api_error_handler,
    router,
)
from src.backend.dashboard.service import (
    DashboardIdempotencyConflictError,
    DashboardService,
    InvalidRequestTransitionError,
    ResourceNotFoundError,
)


IDEMPOTENCY_KEY = "complete-key-1042"


def completed_detail_record() -> RequestDetailRecord:
    requested_at = datetime(
        2026,
        9,
        2,
        8,
        40,
        tzinfo=UTC,
    )
    acknowledged_at = datetime(
        2026,
        9,
        2,
        8,
        41,
        tzinfo=UTC,
    )
    completed_at = datetime(
        2026,
        9,
        2,
        8,
        44,
        tzinfo=UTC,
    )

    return RequestDetailRecord(
        summary=RequestSummaryRecord(
            request_id=1042,
            patient_id=302,
            patient_code="P-2026-0302",
            patient_display_name="김민수",
            ward_code="WARD-3",
            ward_name="3병동",
            room_number="302",
            utterance_id=8801,
            text="자세를 바꿔 주세요",
            phrase_code="REQUEST_REPOSITION",
            category="REQUEST",
            confidence=0.88,
            priority="NORMAL",
            status="COMPLETED",
            requested_at=requested_at,
            acknowledged_at=acknowledged_at,
            acknowledged_by_user_id=51,
            acknowledged_by_display_name="이간호사",
            completed_at=completed_at,
            completed_by_user_id=52,
            completed_by_display_name="박간호사",
        ),
        resolution_note="체위 변경 완료",
        timeline=[
            RequestEventRecord(
                event_type="REQUESTED",
                occurred_at=requested_at,
                actor_user_id=None,
                actor_display_name=None,
                note=None,
            ),
            RequestEventRecord(
                event_type="ACKNOWLEDGED",
                occurred_at=acknowledged_at,
                actor_user_id=51,
                actor_display_name="이간호사",
                note=None,
            ),
            RequestEventRecord(
                event_type="COMPLETED",
                occurred_at=completed_at,
                actor_user_id=52,
                actor_display_name="박간호사",
                note="체위 변경 완료",
            ),
        ],
    )


class FakeRepository:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    async def complete_request(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return completed_detail_record()


async def test_complete_service_returns_completed_detail():
    repository = FakeRepository()
    service = DashboardService(repository)

    result = await service.complete_request(
        staff_user_id=52,
        request_id=1042,
        idempotency_key=IDEMPOTENCY_KEY,
        resolution_note="체위 변경 완료",
        now=datetime(
            2026,
            9,
            2,
            8,
            44,
            tzinfo=UTC,
        ),
    )

    assert result.status == "COMPLETED"
    assert result.resolution_note == "체위 변경 완료"

    assert result.completed_by is not None
    assert result.completed_by.user_id == 52

    assert result.acknowledged_by is not None
    assert result.acknowledged_by.user_id == 51

    assert [
        event.event_type
        for event in result.timeline
    ] == [
        "REQUESTED",
        "ACKNOWLEDGED",
        "COMPLETED",
    ]

    assert result.timeline[-1].note == "체위 변경 완료"

    call = repository.calls[0]

    assert call["staff_user_id"] == 52
    assert call["request_id"] == 1042
    assert call["resolution_note"] == "체위 변경 완료"
    assert len(call["request_fingerprint"]) == 64


async def test_complete_service_maps_not_found():
    service = DashboardService(
        FakeRepository(
            error=RepositoryRequestNotFoundError(),
        )
    )

    try:
        await service.complete_request(
            staff_user_id=52,
            request_id=1042,
            idempotency_key=IDEMPOTENCY_KEY,
            resolution_note=None,
        )
    except ResourceNotFoundError:
        pass
    else:
        raise AssertionError("ResourceNotFoundError expected")


async def test_complete_service_maps_transition_conflict():
    service = DashboardService(
        FakeRepository(
            error=RepositoryTransitionConflictError(),
        )
    )

    try:
        await service.complete_request(
            staff_user_id=52,
            request_id=1042,
            idempotency_key=IDEMPOTENCY_KEY,
            resolution_note=None,
        )
    except InvalidRequestTransitionError:
        pass
    else:
        raise AssertionError(
            "InvalidRequestTransitionError expected"
        )


async def test_complete_service_maps_idempotency_conflict():
    service = DashboardService(
        FakeRepository(
            error=RepositoryIdempotencyConflictError(),
        )
    )

    try:
        await service.complete_request(
            staff_user_id=52,
            request_id=1042,
            idempotency_key=IDEMPOTENCY_KEY,
            resolution_note=None,
        )
    except DashboardIdempotencyConflictError:
        pass
    else:
        raise AssertionError(
            "DashboardIdempotencyConflictError expected"
        )


class FakeAuthService:
    async def get_current_user(self, _token: str):
        return SimpleNamespace(
            user_id=52,
            role="STAFF",
        )


class FakeDashboardService:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    async def complete_request(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return DashboardService._to_request_detail(
            completed_detail_record(),
            generated_at=datetime(
                2026,
                9,
                2,
                8,
                44,
                tzinfo=UTC,
            ),
        )


def make_app(*, dashboard_service=None):
    app = FastAPI()

    app.add_exception_handler(
        DashboardAPIError,
        dashboard_api_error_handler,
    )

    app.include_router(router)

    app.state.auth_service = FakeAuthService()
    app.state.dashboard_service = (
        dashboard_service
        if dashboard_service is not None
        else FakeDashboardService()
    )

    return app


async def post_complete(
    app,
    *,
    key: str | None = IDEMPOTENCY_KEY,
    body=None,
):
    headers = {
        "X-Session-Token": "staff-session",
    }

    if key is not None:
        headers["Idempotency-Key"] = key

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/api/v1/requests/1042/complete",
            headers=headers,
            json=body if body is not None else {},
        )


async def test_complete_api_returns_200_request_detail():
    app = make_app()

    response = await post_complete(
        app,
        body={
            "resolution_note": "체위 변경 완료",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "COMPLETED"
    assert payload["resolution_note"] == "체위 변경 완료"
    assert payload["completed_by"]["user_id"] == 52
    assert payload["timeline"][-1]["event_type"] == "COMPLETED"

    call = app.state.dashboard_service.calls[0]

    assert call["idempotency_key"] == IDEMPOTENCY_KEY
    assert call["resolution_note"] == "체위 변경 완료"


async def test_complete_missing_idempotency_key_returns_400():
    response = await post_complete(
        make_app(),
        key=None,
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_complete_blank_idempotency_key_returns_400():
    response = await post_complete(
        make_app(),
        key="   ",
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_complete_too_long_resolution_note_returns_400():
    response = await post_complete(
        make_app(),
        body={
            "resolution_note": "가" * 501,
        },
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_complete_transition_conflict_returns_exact_409():
    response = await post_complete(
        make_app(
            dashboard_service=FakeDashboardService(
                error=InvalidRequestTransitionError(),
            )
        )
    )

    assert response.status_code == 409

    assert response.json() == {
        "detail": "요청 상태를 변경할 수 없습니다.",
        "code": "INVALID_REQUEST_TRANSITION",
    }


async def test_complete_idempotency_conflict_returns_409():
    response = await post_complete(
        make_app(
            dashboard_service=FakeDashboardService(
                error=DashboardIdempotencyConflictError(),
            )
        )
    )

    assert response.status_code == 409
    assert response.json()["code"] == "IDEMPOTENCY_CONFLICT"


async def test_complete_inaccessible_request_returns_404():
    response = await post_complete(
        make_app(
            dashboard_service=FakeDashboardService(
                error=ResourceNotFoundError(),
            )
        )
    )

    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"
