"""PostgreSQL repository의 transaction과 FK 동작을 검증한다."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from src.backend.auth.adapters.repository import User
from src.backend.recognition.adapters.repository import (
    Phrase,
    RecognitionSession,
    SQLAlchemyRecognitionRepository,
    Utterance,
    VideoAsset,
)
from src.backend.recognition.domain import (
    PhraseCategory,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import SessionAlreadyEndedError

pytestmark = pytest.mark.integration


DEMO_PHRASES = [
    (
        "REQUEST_WATER",
        "물 주세요",
        PhraseCategory.REQUEST,
    ),
    (
        "PAIN_GENERAL",
        "아파요",
        PhraseCategory.PAIN,
    ),
]


async def _create_user(
    postgres_session_factory,
    *,
    display_name: str = "테스트 사용자",
) -> int:
    """영상 FK 테스트에 사용할 실제 사용자 row를 생성한다."""

    user = User(
        username=f"test-{uuid4().hex[:16]}",
        password_hash="test-password-hash",
        display_name=display_name,
    )

    async with postgres_session_factory.begin() as db_session:
        db_session.add(user)
        await db_session.flush()

        return user.user_id


async def test_create_and_complete_session_in_short_transactions(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    await repository.sync_phrases(DEMO_PHRASES)

    session_id = await repository.create_session(RecognitionMode.CLOSED)

    utterance_id = await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="물 주세요",
            corrected_text="물 주세요",
            confidence=0.912,
            phrase_code="REQUEST_WATER",
        ),
    )

    async with postgres_session_factory() as db_session:
        session = await db_session.get(
            RecognitionSession,
            session_id,
        )

        utterance = await db_session.get(
            Utterance,
            utterance_id,
        )

        phrase = await db_session.scalar(
            select(Phrase).where(Phrase.phrase_code == "REQUEST_WATER")
        )

    assert session is not None
    assert utterance is not None
    assert phrase is not None

    assert session.ended_at is not None
    assert utterance.session_id == session_id
    assert utterance.phrase_id == phrase.phrase_id
    assert utterance.raw_text == "물 주세요"
    assert utterance.corrected_text == "물 주세요"

    assert float(utterance.confidence) == pytest.approx(0.912)


async def test_end_session_is_idempotent(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    session_id = await repository.create_session(RecognitionMode.OPEN)

    await repository.end_session(session_id)

    await repository.end_session(session_id)

    async with postgres_session_factory() as db_session:
        session = await db_session.get(
            RecognitionSession,
            session_id,
        )

    assert session is not None
    assert session.ended_at is not None


async def test_reconcile_abandoned_sessions_only_ends_old_open_sessions(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    old_session_id = await repository.create_session(RecognitionMode.CLOSED)

    recent_session_id = await repository.create_session(RecognitionMode.OPEN)

    completed_session_id = await repository.create_session(RecognitionMode.CLOSED)

    now = datetime.now(UTC)

    cutoff = now - timedelta(minutes=10)

    async with postgres_session_factory.begin() as db_session:
        old_session = await db_session.get(
            RecognitionSession,
            old_session_id,
        )

        completed_session = await db_session.get(
            RecognitionSession,
            completed_session_id,
        )

        assert old_session is not None
        assert completed_session is not None

        old_session.started_at = now - timedelta(minutes=20)

        completed_session.started_at = now - timedelta(minutes=20)

    await repository.end_session(completed_session_id)

    reconciled = await repository.reconcile_abandoned_sessions(before=cutoff)

    async with postgres_session_factory() as db_session:
        old_session = await db_session.get(
            RecognitionSession,
            old_session_id,
        )

        recent_session = await db_session.get(
            RecognitionSession,
            recent_session_id,
        )

        completed_session = await db_session.get(
            RecognitionSession,
            completed_session_id,
        )

    assert old_session is not None
    assert recent_session is not None
    assert completed_session is not None

    assert reconciled == 1
    assert old_session.ended_at is not None
    assert recent_session.ended_at is None
    assert completed_session.ended_at is not None


async def test_completed_result_cannot_be_added_after_session_end(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    session_id = await repository.create_session(RecognitionMode.OPEN)

    await repository.end_session(session_id)

    with pytest.raises(SessionAlreadyEndedError):
        await repository.complete_session(
            session_id,
            RecognitionOutput(raw_text="테스트 결과"),
        )

    async with postgres_session_factory() as db_session:
        utterances = list(await db_session.scalars(select(Utterance)))

    assert utterances == []


async def test_deleting_session_cascades_utterances(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    session_id = await repository.create_session(RecognitionMode.OPEN)

    await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="안녕하세요",
            confidence=None,
        ),
    )

    async with postgres_session_factory.begin() as db_session:
        await db_session.execute(
            delete(RecognitionSession).where(
                RecognitionSession.session_id == session_id
            )
        )

    async with postgres_session_factory() as db_session:
        utterances = list(await db_session.scalars(select(Utterance)))

    assert utterances == []


async def test_deleting_phrase_keeps_utterance(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    await repository.sync_phrases(DEMO_PHRASES)

    session_id = await repository.create_session(RecognitionMode.CLOSED)

    utterance_id = await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="아파요",
            confidence=0.8,
            phrase_code="PAIN_GENERAL",
        ),
    )

    async with postgres_session_factory.begin() as db_session:
        await db_session.execute(
            delete(Phrase).where(Phrase.phrase_code == "PAIN_GENERAL")
        )

    async with postgres_session_factory() as db_session:
        utterance = await db_session.get(
            Utterance,
            utterance_id,
        )

    assert utterance is not None
    assert utterance.phrase_id is None


async def test_sync_phrases_removes_retired_phrase_and_keeps_utterance(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    await repository.sync_phrases(
        [
            (
                "REQUEST_LIGHTS_OFF",
                "불 꺼 주세요",
                PhraseCategory.REQUEST,
            )
        ]
    )

    session_id = await repository.create_session(RecognitionMode.CLOSED)

    utterance_id = await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="불 꺼 주세요",
            confidence=0.8,
            phrase_code="REQUEST_LIGHTS_OFF",
        ),
    )

    await repository.sync_phrases(DEMO_PHRASES)

    async with postgres_session_factory() as db_session:
        retired_phrase = await db_session.scalar(
            select(Phrase).where(Phrase.phrase_code == "REQUEST_LIGHTS_OFF")
        )

        remaining_codes = set(await db_session.scalars(select(Phrase.phrase_code)))

        utterance = await db_session.get(
            Utterance,
            utterance_id,
        )

    assert retired_phrase is None

    assert remaining_codes == {
        "REQUEST_WATER",
        "PAIN_GENERAL",
    }

    assert utterance is not None
    assert utterance.raw_text == "불 꺼 주세요"
    assert utterance.phrase_id is None


async def test_unknown_model_phrase_code_does_not_block_final_text(
    postgres_session_factory,
):
    """모델 label map이 바뀌어도 text는 저장하고 알 수 없는 FK만 비운다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    session_id = await repository.create_session(RecognitionMode.CLOSED)

    utterance_id = await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="새 문구",
            confidence=0.7,
            phrase_code="MODEL_LABEL_NOT_SEEDED",
        ),
    )

    async with postgres_session_factory() as db_session:
        utterance = await db_session.get(
            Utterance,
            utterance_id,
        )

    assert utterance is not None
    assert utterance.raw_text == "새 문구"
    assert utterance.phrase_id is None


async def test_create_video_asset_creates_pending_utterance(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    user_id = await _create_user(postgres_session_factory)

    retention_until = datetime.now(UTC) + timedelta(hours=24)

    result = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=("11111111-1111-4111-8111-111111111111"),
        object_key=("storage-user/2026/08/video-source.webm"),
        original_mime_type="video/webm",
        size_bytes=12345,
        checksum="a" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=retention_until,
    )

    assert result.created is True

    assert result.asset.user_id == user_id
    assert result.asset.video_id > 0
    assert result.asset.utterance_id > 0

    assert result.asset.object_key == ("storage-user/2026/08/video-source.webm")

    assert result.asset.original_mime_type == "video/webm"

    assert result.asset.size_bytes == 12345
    assert result.asset.checksum == "a" * 64

    assert result.asset.storage_status == "UPLOADED"

    assert result.asset.storage_purpose == "TEMPORARY_INFERENCE"

    async with postgres_session_factory() as db_session:
        utterance = await db_session.get(
            Utterance,
            result.asset.utterance_id,
        )

        video_asset = await db_session.get(
            VideoAsset,
            result.asset.video_id,
        )

    assert utterance is not None

    assert utterance.user_id == user_id
    assert utterance.session_id is None
    assert utterance.raw_text is None

    assert video_asset is not None
    assert video_asset.user_id == user_id

    assert video_asset.utterance_id == utterance.utt_id


async def test_find_video_asset_by_idempotency_key(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    user_id = await _create_user(postgres_session_factory)

    idempotency_key = "22222222-2222-4222-8222-222222222222"

    created = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=idempotency_key,
        object_key=("storage-user/2026/08/source.mp4"),
        original_mime_type="video/mp4",
        size_bytes=54321,
        checksum="b" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    found = await repository.find_video_asset_by_idempotency_key(
        user_id=user_id,
        idempotency_key=idempotency_key,
    )

    assert found is not None

    assert found.video_id == created.asset.video_id

    assert found.utterance_id == created.asset.utterance_id

    assert found.checksum == "b" * 64


async def test_find_video_asset_by_id(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    user_id = await _create_user(postgres_session_factory)

    created = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key="55555555-5555-4555-8555-555555555555",
        object_key="storage-user/2026/08/find-by-id.webm",
        original_mime_type="video/webm",
        size_bytes=67890,
        checksum="1" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    found = await repository.find_video_asset_by_id(
        video_id=created.asset.video_id,
    )

    assert found is not None
    assert found.video_id == created.asset.video_id
    assert found.utterance_id == created.asset.utterance_id
    assert found.user_id == user_id
    assert found.object_key == "storage-user/2026/08/find-by-id.webm"
    assert found.checksum == "1" * 64


async def test_find_video_asset_by_id_returns_none_for_unknown_video(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    found = await repository.find_video_asset_by_id(
        video_id=2_147_483_647,
    )

    assert found is None


async def test_duplicate_idempotency_key_returns_existing_video_asset(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    user_id = await _create_user(postgres_session_factory)

    idempotency_key = "33333333-3333-4333-8333-333333333333"

    first = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=idempotency_key,
        object_key=("storage-user/2026/08/first.webm"),
        original_mime_type="video/webm",
        size_bytes=1000,
        checksum="c" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    second = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=idempotency_key,
        object_key=("storage-user/2026/08/second.webm"),
        original_mime_type="video/webm",
        size_bytes=2000,
        checksum="d" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    assert first.created is True
    assert second.created is False

    assert second.asset.video_id == first.asset.video_id

    assert second.asset.utterance_id == first.asset.utterance_id

    # 중복 요청에서는 최초 저장값을 유지한다.
    assert second.asset.object_key == first.asset.object_key

    assert second.asset.checksum == first.asset.checksum

    async with postgres_session_factory() as db_session:
        video_assets = list(
            await db_session.scalars(
                select(VideoAsset).where(VideoAsset.user_id == user_id)
            )
        )

        utterances = list(
            await db_session.scalars(
                select(Utterance).where(Utterance.user_id == user_id)
            )
        )

    assert len(video_assets) == 1

    assert len(utterances) == 1


async def test_same_idempotency_key_is_independent_per_user(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    first_user_id = await _create_user(
        postgres_session_factory,
        display_name="첫 번째 사용자",
    )

    second_user_id = await _create_user(
        postgres_session_factory,
        display_name="두 번째 사용자",
    )

    idempotency_key = "44444444-4444-4444-8444-444444444444"

    first = await repository.create_or_get_video_asset(
        user_id=first_user_id,
        idempotency_key=idempotency_key,
        object_key=("first-user/2026/08/source.mp4"),
        original_mime_type="video/mp4",
        size_bytes=1000,
        checksum="e" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    second = await repository.create_or_get_video_asset(
        user_id=second_user_id,
        idempotency_key=idempotency_key,
        object_key=("second-user/2026/08/source.mp4"),
        original_mime_type="video/mp4",
        size_bytes=1000,
        checksum="f" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=None,
    )

    assert first.created is True
    assert second.created is True

    assert first.asset.video_id != second.asset.video_id

    assert first.asset.utterance_id != second.asset.utterance_id

    assert first.asset.user_id == first_user_id

    assert second.asset.user_id == second_user_id


async def test_purge_requires_explicit_target_and_deletes_matching_sessions(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    old_session_id = await repository.create_session(RecognitionMode.CLOSED)

    new_session_id = await repository.create_session(RecognitionMode.OPEN)

    cutoff = datetime.now(UTC) - timedelta(days=1)

    async with postgres_session_factory.begin() as db_session:
        old_session = await db_session.get(
            RecognitionSession,
            old_session_id,
        )

        assert old_session is not None

        old_session.started_at = cutoff - timedelta(days=1)

    with pytest.raises(
        ValueError,
        match="session_id 또는 before",
    ):
        await repository.purge()

    deleted = await repository.purge(before=cutoff)

    assert deleted == 1

    async with postgres_session_factory() as db_session:
        assert (
            await db_session.get(
                RecognitionSession,
                old_session_id,
            )
            is None
        )

        assert (
            await db_session.get(
                RecognitionSession,
                new_session_id,
            )
            is not None
        )



async def test_video_retention_cleanup_candidates_follow_policy(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(
        postgres_session_factory
    )

    user_id = await _create_user(
        postgres_session_factory,
        display_name="retention-user",
    )

    now = datetime.now(UTC)

    ready = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=(
            "91000000-0000-4000-8000-000000000001"
        ),
        object_key="retention/ready.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="1" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=now + timedelta(hours=23),
    )

    expired = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=(
            "91000000-0000-4000-8000-000000000002"
        ),
        object_key="retention/expired.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="2" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=now - timedelta(seconds=1),
    )

    active = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=(
            "91000000-0000-4000-8000-000000000003"
        ),
        object_key="retention/active.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="3" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=now + timedelta(hours=23),
    )

    training = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=(
            "91000000-0000-4000-8000-000000000004"
        ),
        object_key="retention/training.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="4" * 64,
        storage_purpose="MODEL_TRAINING",
        consent_version="2026-09-v1",
        retention_until=None,
    )

    async with postgres_session_factory.begin() as session:
        ready_row = await session.get(
            VideoAsset,
            ready.asset.video_id,
        )

        training_row = await session.get(
            VideoAsset,
            training.asset.video_id,
        )

        assert ready_row is not None
        assert training_row is not None

        ready_row.storage_status = "READY"
        training_row.storage_status = "READY"

    candidates = (
        await repository.list_video_assets_due_for_cleanup(
            now=now,
            limit=100,
        )
    )

    candidate_ids = {
        candidate.video_id
        for candidate in candidates
    }

    assert ready.asset.video_id in candidate_ids
    assert expired.asset.video_id in candidate_ids

    assert active.asset.video_id not in candidate_ids

    # Training-purpose videos are never selected by the
    # automatic temporary-video retention cleanup.
    assert training.asset.video_id not in candidate_ids



async def test_inference_result_records_phrase_usage_exactly_once(
    postgres_session_factory,
):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from src.backend.recognition.domain import (
        PhraseCategory,
        Prediction,
    )

    repository = SQLAlchemyRecognitionRepository(
        postgres_session_factory
    )

    user_id = await _create_user(
        postgres_session_factory,
        display_name="personalization-user",
    )

    phrase_code = "PERSONALIZATION_REQUEST_HELP"

    await repository.sync_phrases(
        [
            (
                phrase_code,
                "?????",
                PhraseCategory.REQUEST,
            )
        ]
    )

    now = datetime.now(UTC)

    saved = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=str(uuid4()),
        object_key="personalization/input.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="9" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=now + timedelta(hours=24),
    )

    prediction = Prediction(
        text="?????",
        confidence=0.9,
        phrase_code=phrase_code,
    )

    await repository.save_inference_result(
        utterance_id=saved.asset.utterance_id,
        prediction=prediction,
        model_version="personalization-test",
    )

    # Worker retry / duplicate persistence must not double-count.
    await repository.save_inference_result(
        utterance_id=saved.asset.utterance_id,
        prediction=prediction,
        model_version="personalization-test",
    )

    stats = await repository.list_phrase_usage_stats(
        user_id=user_id
    )

    matching = [
        stat
        for stat in stats
        if stat.phrase_code == phrase_code
    ]

    assert len(matching) == 1

    usage = matching[0]

    assert usage.usage_count == 1
    assert usage.accepted_count == 0
    assert usage.corrected_count == 0
    assert usage.last_used_at is not None



async def test_training_candidate_is_durable_idempotent_and_consent_gated(
    postgres_session_factory,
):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from sqlalchemy import select

    from src.backend.recognition.adapters.repository import (
        TrainingCandidate,
    )
    from src.backend.recognition.domain import (
        PhraseCategory,
        Prediction,
    )

    repository = SQLAlchemyRecognitionRepository(
        postgres_session_factory
    )

    phrase_code = "TRAINING_REQUEST_HELP"

    await repository.sync_phrases(
        [
            (
                phrase_code,
                "?????",
                PhraseCategory.REQUEST,
            )
        ]
    )

    user_id = await _create_user(
        postgres_session_factory,
        display_name="training-consent-user",
    )

    await repository.upsert_user_consent(
        user_id=user_id,
        model_training_consent=True,
        consent_version="2026-09-v1",
    )

    now = datetime.now(UTC)

    saved = await repository.create_or_get_video_asset(
        user_id=user_id,
        idempotency_key=str(uuid4()),
        object_key="training/consented.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="7" * 64,
        storage_purpose="MODEL_TRAINING",
        consent_version="2026-09-v1",
        retention_until=None,
    )

    prediction = Prediction(
        text="?????",
        confidence=0.91,
        phrase_code=phrase_code,
    )

    await repository.save_inference_result(
        utterance_id=saved.asset.utterance_id,
        prediction=prediction,
        model_version="bundle-v1",
    )

    # Worker/result-persistence retry must not duplicate.
    await repository.save_inference_result(
        utterance_id=saved.asset.utterance_id,
        prediction=prediction,
        model_version="bundle-v1",
    )

    async with postgres_session_factory() as session:
        candidates = list(
            await session.scalars(
                select(TrainingCandidate).where(
                    TrainingCandidate.user_id
                    == user_id
                )
            )
        )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.utterance_id == (
        saved.asset.utterance_id
    )

    assert candidate.video_id == saved.asset.video_id
    assert candidate.status == "UNLABELED"
    assert candidate.model_version == "bundle-v1"

    assert (
        candidate.predicted_phrase_code
        == phrase_code
    )

    assert float(candidate.confidence) == pytest.approx(
        0.91
    )

    assert candidate.published_at is None

    object_key = (
        await repository.get_training_candidate_object_key(
            sample_id=candidate.sample_id
        )
    )

    assert object_key == "training/consented.webm"

    pending = (
        await repository.list_unpublished_training_candidates(
            limit=100
        )
    )

    assert [
        item.sample_id
        for item in pending
    ] == [
        candidate.sample_id
    ]

    # Withdrawal durably excludes the candidate.
    await repository.upsert_user_consent(
        user_id=user_id,
        model_training_consent=False,
        consent_version="2026-09-v1",
    )

    async with postgres_session_factory() as session:
        withdrawn = await session.get(
            TrainingCandidate,
            candidate.sample_id,
        )

    assert withdrawn is not None
    assert withdrawn.status == "REJECTED"

    assert (
        await repository.get_training_candidate_object_key(
            sample_id=candidate.sample_id
        )
        is None
    )

    pending_after_withdrawal = (
        await repository.list_unpublished_training_candidates(
            limit=100
        )
    )

    assert all(
        item.sample_id != candidate.sample_id
        for item in pending_after_withdrawal
    )

    # A temporary/non-consented inference must never become
    # a training candidate.
    other_user_id = await _create_user(
        postgres_session_factory,
        display_name="training-nonconsent-user",
    )

    temporary = await repository.create_or_get_video_asset(
        user_id=other_user_id,
        idempotency_key=str(uuid4()),
        object_key="training/temporary.webm",
        original_mime_type="video/webm",
        size_bytes=100,
        checksum="8" * 64,
        storage_purpose="TEMPORARY_INFERENCE",
        consent_version=None,
        retention_until=now + timedelta(hours=24),
    )

    await repository.save_inference_result(
        utterance_id=temporary.asset.utterance_id,
        prediction=prediction,
        model_version="bundle-v1",
    )

    async with postgres_session_factory() as session:
        unexpected = await session.scalar(
            select(TrainingCandidate).where(
                TrainingCandidate.utterance_id
                == temporary.asset.utterance_id
            )
        )

    assert unexpected is None
