"""PostgreSQL repository의 transaction과 FK 동작을 검증한다."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from src.backend.recognition.adapters.repository import (
    Phrase,
    RecognitionSession,
    SQLAlchemyRecognitionRepository,
    Utterance,
)
from src.backend.recognition.domain import (
    PhraseCategory,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import SessionAlreadyEndedError

pytestmark = pytest.mark.integration


DEMO_PHRASES = [
    ("REQUEST_WATER", "물 주세요", PhraseCategory.REQUEST),
    ("PAIN_GENERAL", "아파요", PhraseCategory.PAIN),
]


async def test_create_and_complete_session_in_short_transactions(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    await repository.seed_phrases(DEMO_PHRASES)
    session_id = await repository.create_session(RecognitionMode.CLOSED)

    utterance_id = await repository.complete_session(
        session_id,
        RecognitionOutput(
            raw_text="물 주세오",
            corrected_text="물 주세요",
            confidence=0.912,
            phrase_code="REQUEST_WATER",
        ),
    )

    async with postgres_session_factory() as db_session:
        session = await db_session.get(RecognitionSession, session_id)
        utterance = await db_session.get(Utterance, utterance_id)
        phrase = await db_session.scalar(
            select(Phrase).where(Phrase.phrase_code == "REQUEST_WATER")
        )

    assert session.ended_at is not None
    assert utterance.session_id == session_id
    assert utterance.phrase_id == phrase.phrase_id
    assert utterance.raw_text == "물 주세오"
    assert utterance.corrected_text == "물 주세요"
    assert float(utterance.confidence) == pytest.approx(0.912)


async def test_end_session_is_idempotent(postgres_session_factory):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    session_id = await repository.create_session(RecognitionMode.OPEN)

    await repository.end_session(session_id)
    await repository.end_session(session_id)

    async with postgres_session_factory() as db_session:
        session = await db_session.get(RecognitionSession, session_id)

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
        old_session = await db_session.get(RecognitionSession, old_session_id)
        old_session.started_at = now - timedelta(minutes=20)
        completed_session = await db_session.get(
            RecognitionSession, completed_session_id
        )
        completed_session.started_at = now - timedelta(minutes=20)
    await repository.end_session(completed_session_id)

    reconciled = await repository.reconcile_abandoned_sessions(before=cutoff)

    async with postgres_session_factory() as db_session:
        old_session = await db_session.get(RecognitionSession, old_session_id)
        recent_session = await db_session.get(RecognitionSession, recent_session_id)
        completed_session = await db_session.get(
            RecognitionSession, completed_session_id
        )

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
            RecognitionOutput(raw_text="늦은 결과"),
        )

    async with postgres_session_factory() as db_session:
        utterances = list(await db_session.scalars(select(Utterance)))

    assert utterances == []


async def test_deleting_session_cascades_utterances(postgres_session_factory):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    session_id = await repository.create_session(RecognitionMode.OPEN)
    await repository.complete_session(
        session_id,
        RecognitionOutput(raw_text="안녕하세요", confidence=None),
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


async def test_deleting_phrase_keeps_utterance(postgres_session_factory):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    await repository.seed_phrases(DEMO_PHRASES)
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
        utterance = await db_session.get(Utterance, utterance_id)

    assert utterance is not None
    assert utterance.phrase_id is None


async def test_unknown_model_phrase_code_does_not_block_final_text(
    postgres_session_factory,
):
    """모델 label map이 바뀌어도 API 결과는 저장하고 FK만 비워 둔다."""

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
        utterance = await db_session.get(Utterance, utterance_id)

    assert utterance.raw_text == "새 문구"
    assert utterance.phrase_id is None


async def test_purge_requires_explicit_target_and_deletes_matching_sessions(
    postgres_session_factory,
):
    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)
    old_session_id = await repository.create_session(RecognitionMode.CLOSED)
    new_session_id = await repository.create_session(RecognitionMode.OPEN)

    cutoff = datetime.now(UTC) - timedelta(days=1)
    async with postgres_session_factory.begin() as db_session:
        old_session = await db_session.get(RecognitionSession, old_session_id)
        old_session.started_at = cutoff - timedelta(days=1)

    with pytest.raises(ValueError, match="session_id 또는 before"):
        await repository.purge()

    deleted = await repository.purge(before=cutoff)

    assert deleted == 1
    async with postgres_session_factory() as db_session:
        assert await db_session.get(RecognitionSession, old_session_id) is None
        assert await db_session.get(RecognitionSession, new_session_id) is not None
