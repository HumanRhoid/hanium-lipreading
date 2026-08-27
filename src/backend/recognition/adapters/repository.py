"""인식 도메인의 SQLAlchemy 모델과 PostgreSQL repository."""

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.backend.core.database import Base
from src.backend.recognition.domain import (
    PhraseCategory,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import SessionAlreadyEndedError


class RecognitionSession(Base):
    __tablename__ = "session"
    __table_args__ = (
        CheckConstraint("mode IN ('CLOSED', 'OPEN')", name="mode"),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="ended_after_started",
        ),
        Index("ix_session_started_at", "started_at"),
    )

    session_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    utterances: Mapped[list["Utterance"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Phrase(Base):
    __tablename__ = "phrase"
    __table_args__ = (
        UniqueConstraint("phrase_code"),
        CheckConstraint("phrase_text ~ '[^[:space:]]'", name="text_not_blank"),
        CheckConstraint(
            "category IN ('PAIN', 'REQUEST', 'REPLY', 'ETC')", name="category"
        ),
    )

    phrase_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    phrase_code: Mapped[str] = mapped_column(String(64), nullable=False)
    phrase_text: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    utterances: Mapped[list["Utterance"]] = relationship(
        back_populates="phrase", passive_deletes=True
    )


class Utterance(Base):
    __tablename__ = "utterance"
    __table_args__ = (
        CheckConstraint(
            "raw_text IS NULL OR raw_text ~ '[^[:space:]]'",
            name="raw_text_not_blank",
        ),
        CheckConstraint(
            "corrected_text IS NULL OR corrected_text ~ '[^[:space:]]'",
            name="corrected_text_not_blank",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        Index("ix_utterance_session_created_at", "session_id", "created_at"),
        Index("ix_utterance_user_created_at", "user_id", "created_at"),
        Index("ix_utterance_phrase_id", "phrase_id"),
    )

    utt_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True
    )

    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("session.session_id", ondelete="CASCADE"), nullable=True
    )

    phrase_id: Mapped[int | None] = mapped_column(
        ForeignKey("phrase.phrase_id", ondelete="SET NULL"), nullable=True
    )

    raw_text: Mapped[str | None] = mapped_column(String(200), nullable=True)
    corrected_text: Mapped[str | None] = mapped_column(String(200), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped[RecognitionSession | None] = relationship(
        back_populates="utterances"
    )
    phrase: Mapped[Phrase | None] = relationship(back_populates="utterances")


class VideoAsset(Base):
    """Object Storage에 저장된 인식 영상의 메타데이터."""

    __tablename__ = "video_asset"
    __table_args__ = (
        CheckConstraint(
            (
                "storage_status IN "
                "('UPLOADED', 'NORMALIZING', 'READY', "
                "'DELETE_PENDING', 'DELETED', 'FAILED')"
            ),
            name="storage_status",
        ),
        CheckConstraint(
            "storage_purpose IN ('TEMPORARY_INFERENCE', 'MODEL_TRAINING')",
            name="storage_purpose",
        ),
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_video_asset_user_id_idempotency_key",
        ),
        Index("ix_video_asset_user_created_at", "user_id", "created_at"),
        Index("ix_video_asset_retention_until", "retention_until"),
    )

    video_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )

    utterance_id: Mapped[int] = mapped_column(
        ForeignKey("utterance.utt_id", ondelete="CASCADE"), nullable=False
    )

    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    normalized_mime_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    codec: Mapped[str | None] = mapped_column(String(50), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[Decimal | None] = mapped_column(Numeric(7, 3), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_status: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    consent_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    retention_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class UserConsent(Base):
    """사용자의 모델 재학습용 영상 활용 동의 상태."""

    __tablename__ = "user_consent"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_training_consent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    consent_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PhraseUsageStat(Base):
    """사용자별 문구 개인화 통계의 PostgreSQL 원본."""

    __tablename__ = "phrase_usage_stat"
    __table_args__ = (
        CheckConstraint("usage_count >= 0", name="usage_count_nonnegative"),
        CheckConstraint("accepted_count >= 0", name="accepted_count_nonnegative"),
        CheckConstraint("corrected_count >= 0", name="corrected_count_nonnegative"),
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    phrase_code: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("phrase.phrase_code", ondelete="CASCADE"),
        primary_key=True,
    )
    usage_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    accepted_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    corrected_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SQLAlchemyRecognitionRepository:
    """WebSocket 수명과 분리된 짧은 AsyncSession을 사용한다."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def create_session(self, mode: RecognitionMode) -> int:
        async with self._session_factory.begin() as db_session:
            session = RecognitionSession(mode=mode.value)
            db_session.add(session)
            await db_session.flush()
            return session.session_id

    async def complete_session(self, session_id: int, output: RecognitionOutput) -> int:
        async with self._session_factory.begin() as db_session:
            session = await db_session.get(
                RecognitionSession, session_id, with_for_update=True
            )

            if session is None:
                raise LookupError(f"인식 세션을 찾을 수 없습니다: {session_id}")

            if session.ended_at is not None:
                raise SessionAlreadyEndedError("이미 종료된 인식 세션입니다.")

            phrase_id = None
            if output.phrase_code is not None:
                phrase_id = await db_session.scalar(
                    select(Phrase.phrase_id).where(
                        Phrase.phrase_code == output.phrase_code
                    )
                )

            utterance = Utterance(
                session_id=session_id,
                phrase_id=phrase_id,
                raw_text=output.raw_text,
                corrected_text=output.corrected_text,
                confidence=(
                    Decimal(str(output.confidence))
                    if output.confidence is not None
                    else None
                ),
            )

            db_session.add(utterance)
            await db_session.execute(
                update(RecognitionSession)
                .where(RecognitionSession.session_id == session_id)
                .values(ended_at=func.now())
            )
            await db_session.flush()
            return utterance.utt_id

    async def end_session(self, session_id: int) -> None:
        async with self._session_factory.begin() as db_session:
            await db_session.execute(
                update(RecognitionSession)
                .where(
                    RecognitionSession.session_id == session_id,
                    RecognitionSession.ended_at.is_(None),
                )
                .values(ended_at=func.now())
            )

    async def reconcile_abandoned_sessions(self, *, before: datetime) -> int:
        """기준 시각보다 오래 열린 세션을 강제 종료 상태로 전환한다."""

        if before.tzinfo is None or before.utcoffset() is None:
            raise ValueError("before에는 timezone 정보가 포함되어야 합니다.")

        async with self._session_factory.begin() as db_session:
            result = await db_session.execute(
                update(RecognitionSession)
                .where(
                    RecognitionSession.ended_at.is_(None),
                    RecognitionSession.started_at < before,
                )
                .values(ended_at=func.now())
            )

            return result.rowcount or 0

    async def sync_phrases(
        self, phrases: Iterable[tuple[str, str, PhraseCategory]]
    ) -> None:
        """권위 있는 문구 목록으로 phrase 테이블을 동기화한다."""

        values = [
            {
                "phrase_code": code,
                "phrase_text": text,
                "category": category.value,
            }
            for code, text, category in phrases
        ]

        if not values:
            raise ValueError("동기화할 문구가 비어 있습니다.")

        phrase_codes = [value["phrase_code"] for value in values]
        if len(phrase_codes) != len(set(phrase_codes)):
            raise ValueError("phrase_code는 중복될 수 없습니다.")

        statement = insert(Phrase).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[Phrase.phrase_code],
            set_={
                "phrase_text": statement.excluded.phrase_text,
                "category": statement.excluded.category,
            },
        )

        async with self._session_factory.begin() as db_session:
            await db_session.execute(statement)
            await db_session.execute(
                delete(Phrase).where(Phrase.phrase_code.not_in(phrase_codes))
            )

    async def purge(
        self,
        *,
        session_id: int | None = None,
        before: datetime | None = None,
    ) -> int:
        if (session_id is None) == (before is None):
            raise ValueError("session_id 또는 before 중 하나만 지정해야 합니다.")

        condition = (
            RecognitionSession.session_id == session_id
            if session_id is not None
            else RecognitionSession.started_at < before
        )

        async with self._session_factory.begin() as db_session:
            result = await db_session.execute(
                delete(RecognitionSession).where(condition)
            )

            return result.rowcount or 0