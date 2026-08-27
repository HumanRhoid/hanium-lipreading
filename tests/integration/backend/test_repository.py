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
