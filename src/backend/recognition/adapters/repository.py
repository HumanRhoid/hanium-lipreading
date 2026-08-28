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
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import SessionAlreadyEndedError
from src.backend.recognition.ports import (
    InferenceResultRecord,
    VideoAssetRecord,
    VideoAssetSaveResult,
)


class RecognitionSession(Base):
    __tablename__ = "session"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('CLOSED', 'OPEN')",
            name="mode",
        ),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="ended_after_started",
        ),
        Index(
            "ix_session_started_at",
            "started_at",
        ),
    )

    session_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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
        CheckConstraint(
            "phrase_text ~ '[^[:space:]]'",
            name="text_not_blank",
        ),
        CheckConstraint(
            "category IN ('PAIN', 'REQUEST', 'REPLY', 'ETC')",
            name="category",
        ),
    )

    phrase_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    phrase_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    phrase_text: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    utterances: Mapped[list["Utterance"]] = relationship(
        back_populates="phrase",
        passive_deletes=True,
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
        Index(
            "ix_utterance_session_created_at",
            "session_id",
            "created_at",
        ),
        Index(
            "ix_utterance_user_created_at",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_utterance_phrase_id",
            "phrase_id",
        ),
    )

    utt_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "users.user_id",
            ondelete="CASCADE",
        ),
        nullable=True,
    )

    session_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "session.session_id",
            ondelete="CASCADE",
        ),
        nullable=True,
    )

    phrase_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "phrase.phrase_id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    raw_text: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    corrected_text: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 3),
        nullable=True,
    )

    model_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    session: Mapped[RecognitionSession | None] = relationship(
        back_populates="utterances",
    )

    phrase: Mapped[Phrase | None] = relationship(
        back_populates="utterances",
    )


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
            ("storage_purpose IN ('TEMPORARY_INFERENCE', 'MODEL_TRAINING')"),
            name="storage_purpose",
        ),
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_video_asset_user_id_idempotency_key",
        ),
        Index(
            "ix_video_asset_user_created_at",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_video_asset_retention_until",
            "retention_until",
        ),
    )

    video_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.user_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    utterance_id: Mapped[int] = mapped_column(
        ForeignKey(
            "utterance.utt_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    object_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    original_mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    normalized_mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    codec: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    fps: Mapped[Decimal | None] = mapped_column(
        Numeric(7, 3),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    checksum: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    storage_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    storage_purpose: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    consent_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    retention_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class UserConsent(Base):
    """사용자의 모델 재학습 영상 사용 동의 상태."""

    __tablename__ = "user_consent"

    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.user_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    model_training_consent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    consent_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PhraseUsageStat(Base):
    """사용자별 문구 개인화 통계의 PostgreSQL 원본."""

    __tablename__ = "phrase_usage_stat"
    __table_args__ = (
        CheckConstraint(
            "usage_count >= 0",
            name="usage_count_nonnegative",
        ),
        CheckConstraint(
            "accepted_count >= 0",
            name="accepted_count_nonnegative",
        ),
        CheckConstraint(
            "corrected_count >= 0",
            name="corrected_count_nonnegative",
        ),
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "users.user_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    phrase_code: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "phrase.phrase_code",
            ondelete="CASCADE",
        ),
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
        DateTime(timezone=True),
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


def _to_video_asset_record(
    asset: VideoAsset,
) -> VideoAssetRecord:
    """SQLAlchemy 모델을 서비스 계층용 불변 DTO로 변환한다."""

    return VideoAssetRecord(
        video_id=asset.video_id,
        utterance_id=asset.utterance_id,
        user_id=asset.user_id,
        idempotency_key=asset.idempotency_key,
        object_key=asset.object_key,
        original_mime_type=asset.original_mime_type,
        size_bytes=asset.size_bytes,
        checksum=asset.checksum,
        storage_status=asset.storage_status,
        storage_purpose=asset.storage_purpose,
        created_at=asset.created_at,
        retention_until=asset.retention_until,
    )


def _to_inference_result_record(
    utterance: Utterance,
    phrase_code: str | None,
) -> InferenceResultRecord | None:
    """완료된 utterance를 외부 노출용 결과로 변환한다."""

    text = utterance.corrected_text or utterance.raw_text

    if text is None:
        return None

    return InferenceResultRecord(
        utterance_id=utterance.utt_id,
        text=text,
        phrase_code=phrase_code,
        confidence=(
            float(utterance.confidence)
            if utterance.confidence is not None
            else None
        ),
        model_version=utterance.model_version,
        created_at=utterance.created_at,
    )


class SQLAlchemyRecognitionRepository:
    """인식 및 영상 메타데이터를 AsyncSession으로 저장한다."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def create_session(
        self,
        mode: RecognitionMode,
    ) -> int:
        async with self._session_factory.begin() as db_session:
            session = RecognitionSession(
                mode=mode.value,
            )

            db_session.add(session)
            await db_session.flush()

            return session.session_id

    async def complete_session(
        self,
        session_id: int,
        output: RecognitionOutput,
    ) -> int:
        async with self._session_factory.begin() as db_session:
            session = await db_session.get(
                RecognitionSession,
                session_id,
                with_for_update=True,
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

    async def end_session(
        self,
        session_id: int,
    ) -> None:
        async with self._session_factory.begin() as db_session:
            await db_session.execute(
                update(RecognitionSession)
                .where(
                    RecognitionSession.session_id == session_id,
                    RecognitionSession.ended_at.is_(None),
                )
                .values(ended_at=func.now())
            )

    async def find_video_asset_by_id(
        self,
        *,
        video_id: int,
    ) -> VideoAssetRecord | None:
        """video_id로 영상 메타데이터를 조회한다."""

        async with self._session_factory() as db_session:
            asset = await db_session.get(
                VideoAsset,
                video_id,
            )

            if asset is None:
                return None

            return _to_video_asset_record(asset)

    async def find_video_asset_by_idempotency_key(
        self,
        *,
        user_id: int,
        idempotency_key: str,
    ) -> VideoAssetRecord | None:
        """사용자와 idempotency key로 기존 영상을 조회한다."""

        async with self._session_factory() as db_session:
            asset = await db_session.scalar(
                select(VideoAsset).where(
                    VideoAsset.user_id == user_id,
                    VideoAsset.idempotency_key == idempotency_key,
                )
            )

            if asset is None:
                return None

            return _to_video_asset_record(asset)

    async def create_or_get_video_asset(
        self,
        *,
        user_id: int,
        idempotency_key: str,
        object_key: str,
        original_mime_type: str,
        size_bytes: int,
        checksum: str,
        storage_purpose: str,
        consent_version: str | None,
        retention_until: datetime | None,
    ) -> VideoAssetSaveResult:
        """새 영상 메타데이터를 저장하거나 중복 요청의 기존 영상을 반환한다."""

        async with self._session_factory.begin() as db_session:
            existing_asset = await db_session.scalar(
                select(VideoAsset).where(
                    VideoAsset.user_id == user_id,
                    VideoAsset.idempotency_key == idempotency_key,
                )
            )

            if existing_asset is not None:
                return VideoAssetSaveResult(
                    asset=_to_video_asset_record(existing_asset),
                    created=False,
                )

            utterance = Utterance(
                user_id=user_id,
            )

            db_session.add(utterance)
            await db_session.flush()

            statement = (
                insert(VideoAsset)
                .values(
                    user_id=user_id,
                    utterance_id=utterance.utt_id,
                    idempotency_key=idempotency_key,
                    object_key=object_key,
                    original_mime_type=original_mime_type,
                    normalized_mime_type=None,
                    codec=None,
                    width=None,
                    height=None,
                    fps=None,
                    duration_ms=None,
                    size_bytes=size_bytes,
                    checksum=checksum,
                    storage_status="UPLOADED",
                    storage_purpose=storage_purpose,
                    consent_version=consent_version,
                    retention_until=retention_until,
                    deleted_at=None,
                )
                .on_conflict_do_nothing(
                    constraint=("uq_video_asset_user_id_idempotency_key")
                )
                .returning(VideoAsset.video_id)
            )

            inserted_video_id = await db_session.scalar(statement)

            if inserted_video_id is None:
                # 동시에 같은 Idempotency-Key가 들어온 경우
                # 방금 만든 빈 utterance는 필요 없으므로 제거한다.
                await db_session.execute(
                    delete(Utterance).where(Utterance.utt_id == utterance.utt_id)
                )

                existing_asset = await db_session.scalar(
                    select(VideoAsset).where(
                        VideoAsset.user_id == user_id,
                        VideoAsset.idempotency_key == idempotency_key,
                    )
                )

                if existing_asset is None:
                    raise RuntimeError("중복 영상 메타데이터를 조회할 수 없습니다.")

                return VideoAssetSaveResult(
                    asset=_to_video_asset_record(existing_asset),
                    created=False,
                )

            asset = await db_session.get(
                VideoAsset,
                inserted_video_id,
            )

            if asset is None:
                raise RuntimeError("저장된 영상 메타데이터를 조회할 수 없습니다.")

            return VideoAssetSaveResult(
                asset=_to_video_asset_record(asset),
                created=True,
            )

    async def save_inference_result(
        self,
        *,
        utterance_id: int,
        prediction: Prediction,
        model_version: str | None,
    ) -> InferenceResultRecord:
        """Worker 예측을 기존 업로드 utterance에 멱등 저장한다."""

        async with self._session_factory.begin() as db_session:
            utterance = await db_session.get(
                Utterance,
                utterance_id,
                with_for_update=True,
            )

            if utterance is None:
                raise LookupError(
                    f"업로드 발화 정보를 찾을 수 없습니다: {utterance_id}"
                )

            if utterance.raw_text is None:
                phrase_id = None

                if prediction.phrase_code is not None:
                    phrase_id = await db_session.scalar(
                        select(Phrase.phrase_id).where(
                            Phrase.phrase_code == prediction.phrase_code
                        )
                    )

                utterance.phrase_id = phrase_id
                utterance.raw_text = prediction.text
                utterance.corrected_text = None
                utterance.confidence = (
                    Decimal(str(prediction.confidence))
                    if prediction.confidence is not None
                    else None
                )
                utterance.model_version = model_version

                await db_session.execute(
                    update(VideoAsset)
                    .where(VideoAsset.utterance_id == utterance_id)
                    .values(storage_status="READY")
                )
                await db_session.flush()

            phrase_code = None

            if utterance.phrase_id is not None:
                phrase_code = await db_session.scalar(
                    select(Phrase.phrase_code).where(
                        Phrase.phrase_id == utterance.phrase_id
                    )
                )

            result = _to_inference_result_record(
                utterance,
                phrase_code,
            )

            if result is None:
                raise RuntimeError("추론 결과를 저장할 수 없습니다.")

            return result

    async def get_inference_result(
        self,
        *,
        utterance_id: int,
    ) -> InferenceResultRecord | None:
        """완료된 업로드 utterance 결과를 조회한다."""

        async with self._session_factory() as db_session:
            row = (
                await db_session.execute(
                    select(
                        Utterance,
                        Phrase.phrase_code,
                    )
                    .outerjoin(
                        Phrase,
                        Phrase.phrase_id == Utterance.phrase_id,
                    )
                    .where(Utterance.utt_id == utterance_id)
                )
            ).one_or_none()

            if row is None:
                return None

            utterance, phrase_code = row

            return _to_inference_result_record(
                utterance,
                phrase_code,
            )

    async def reconcile_abandoned_sessions(
        self,
        *,
        before: datetime,
    ) -> int:
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
        self,
        phrases: Iterable[
            tuple[
                str,
                str,
                PhraseCategory,
            ]
        ],
    ) -> None:
        """권위 있는 문구 목록으로 phrase 테이블을 동기화한다."""

        values = [
            {
                "phrase_code": code,
                "phrase_text": phrase_text,
                "category": category.value,
            }
            for (
                code,
                phrase_text,
                category,
            ) in phrases
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
                "phrase_text": (statement.excluded.phrase_text),
                "category": (statement.excluded.category),
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
