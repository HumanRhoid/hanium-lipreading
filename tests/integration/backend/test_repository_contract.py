"""인식 repository가 지켜야 하는 PostgreSQL 데이터 계약 테스트."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.backend.recognition.adapters.repository import (
    Phrase,
    RecognitionSession,
    SQLAlchemyRecognitionRepository,
    Utterance,
)
from src.backend.recognition.domain import PhraseCategory, RecognitionMode

pytestmark = pytest.mark.integration


async def test_seed_phrases_is_idempotent_and_updates_mutable_fields(
    postgres_session_factory,
):
    """동일한 불변 코드는 중복하지 않고 표시 문구와 분류만 갱신한다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    await repository.seed_phrases(
        [("REQUEST_WATER", "물 주세요", PhraseCategory.REQUEST)]
    )

    async with postgres_session_factory() as db_session:
        original = await db_session.scalar(
            select(Phrase).where(Phrase.phrase_code == "REQUEST_WATER")
        )
        original_id = original.phrase_id

    await repository.seed_phrases(
        [("REQUEST_WATER", "물을 주세요", PhraseCategory.REPLY)]
    )
    await repository.seed_phrases(
        [("REQUEST_WATER", "물을 주세요", PhraseCategory.REPLY)]
    )

    async with postgres_session_factory() as db_session:
        phrases = list(await db_session.scalars(select(Phrase)))

    assert len(phrases) == 1
    assert phrases[0].phrase_id == original_id
    assert phrases[0].phrase_text == "물을 주세요"
    assert phrases[0].category == PhraseCategory.REPLY.value


async def test_purge_rejects_missing_or_ambiguous_target(postgres_session_factory):
    """무대상 전체 삭제와 두 기준을 동시에 지정한 모호한 삭제를 거부한다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    session_id = await repository.create_session(RecognitionMode.OPEN)

    with pytest.raises(ValueError, match="session_id 또는 before 중 하나만"):
        await repository.purge()

    with pytest.raises(ValueError, match="session_id 또는 before 중 하나만"):
        await repository.purge(session_id=session_id, before=datetime.now(UTC))

    async with postgres_session_factory() as db_session:
        assert await db_session.get(RecognitionSession, session_id) is not None


async def test_purge_by_session_id_deletes_only_explicit_target(
    postgres_session_factory,
):
    """ID purge는 다른 세션을 건드리지 않고 존재하지 않는 ID에는 0을 반환한다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    target_id = await repository.create_session(RecognitionMode.CLOSED)
    retained_id = await repository.create_session(RecognitionMode.OPEN)

    assert await repository.purge(session_id=target_id) == 1
    assert await repository.purge(session_id=target_id) == 0

    async with postgres_session_factory() as db_session:
        assert await db_session.get(RecognitionSession, target_id) is None
        assert await db_session.get(RecognitionSession, retained_id) is not None


async def test_purge_before_uses_strict_cutoff(postgres_session_factory):
    """시각 purge는 기준 시각보다 과거인 세션만 제거한다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    old_id = await repository.create_session(RecognitionMode.CLOSED)
    boundary_id = await repository.create_session(RecognitionMode.CLOSED)
    recent_id = await repository.create_session(RecognitionMode.OPEN)
    cutoff = datetime.now(UTC) - timedelta(days=1)

    async with postgres_session_factory.begin() as db_session:
        (await db_session.get(RecognitionSession, old_id)).started_at = (
            cutoff - timedelta(seconds=1)
        )
        (await db_session.get(RecognitionSession, boundary_id)).started_at = cutoff
        (await db_session.get(RecognitionSession, recent_id)).started_at = (
            cutoff + timedelta(seconds=1)
        )

    assert await repository.purge(before=cutoff) == 1

    async with postgres_session_factory() as db_session:
        remaining_ids = set(
            await db_session.scalars(select(RecognitionSession.session_id))
        )

    assert remaining_ids == {boundary_id, recent_id}


@pytest.mark.parametrize("mode", ["", "closed", "INVALID"])
async def test_session_rejects_mode_outside_contract(
    postgres_session_factory,
    mode,
):
    """session.mode CHECK가 애플리케이션 enum 밖의 값을 차단한다."""

    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(RecognitionSession(mode=mode))


async def test_session_rejects_end_before_start(postgres_session_factory):
    """종료 시각은 시작 시각보다 이를 수 없다."""

    started_at = datetime.now(UTC)
    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(
                RecognitionSession(
                    mode=RecognitionMode.OPEN.value,
                    started_at=started_at,
                    ended_at=started_at - timedelta(microseconds=1),
                )
            )


@pytest.mark.parametrize(
    ("phrase_text", "category"),
    [
        ("   ", PhraseCategory.REQUEST.value),
        ("\t\n", PhraseCategory.REQUEST.value),
        ("물 주세요", "INVALID"),
    ],
)
async def test_phrase_rejects_values_outside_contract(
    postgres_session_factory,
    phrase_text,
    category,
):
    """문구의 공백 표시값과 허용되지 않은 분류를 차단한다."""

    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(
                Phrase(
                    phrase_code="REQUEST_WATER",
                    phrase_text=phrase_text,
                    category=category,
                )
            )


async def test_phrase_code_is_unique(postgres_session_factory):
    """환경별 PK와 무관한 phrase_code의 유일성을 보장한다."""

    async with postgres_session_factory.begin() as db_session:
        db_session.add(
            Phrase(
                phrase_code="REQUEST_WATER",
                phrase_text="물 주세요",
                category=PhraseCategory.REQUEST.value,
            )
        )

    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(
                Phrase(
                    phrase_code="REQUEST_WATER",
                    phrase_text="물을 주세요",
                    category=PhraseCategory.REQUEST.value,
                )
            )


@pytest.mark.parametrize(
    ("raw_text", "corrected_text", "confidence"),
    [
        ("   ", None, None),
        ("\t\n", None, None),
        ("안녕", "   ", None),
        ("안녕", "\t\n", None),
        ("안녕", None, -0.001),
        ("안녕", None, 1.001),
    ],
)
async def test_utterance_rejects_values_outside_contract(
    postgres_session_factory,
    raw_text,
    corrected_text,
    confidence,
):
    """발화 텍스트와 confidence 범위를 PostgreSQL에서도 강제한다."""

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    session_id = await repository.create_session(RecognitionMode.OPEN)

    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(
                Utterance(
                    session_id=session_id,
                    raw_text=raw_text,
                    corrected_text=corrected_text,
                    confidence=confidence,
                )
            )

    async with postgres_session_factory() as db_session:
        assert await db_session.scalar(select(func.count(Utterance.utt_id))) == 0


async def test_utterance_requires_existing_session(postgres_session_factory):
    """존재하지 않는 세션에는 발화를 기록할 수 없다."""

    with pytest.raises(IntegrityError):
        async with postgres_session_factory.begin() as db_session:
            db_session.add(Utterance(session_id=999_999, raw_text="안녕하세요"))
