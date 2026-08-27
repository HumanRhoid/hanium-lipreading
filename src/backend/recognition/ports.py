"""인식 애플리케이션과 외부 구현 사이의 포트."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from src.backend.recognition.domain import (
    ModelManifest,
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)

InferenceJobStatus = Literal[
    "QUEUED",
    "PROCESSING",
    "SUCCEEDED",
    "FAILED",
]


@dataclass(frozen=True, slots=True)
class VideoAssetRecord:
    """업로드된 영상 메타데이터의 애플리케이션용 표현."""

    video_id: int
    utterance_id: int
    user_id: int
    idempotency_key: str
    object_key: str
    original_mime_type: str
    size_bytes: int
    checksum: str
    storage_status: str
    storage_purpose: str
    created_at: datetime
    retention_until: datetime | None


@dataclass(frozen=True, slots=True)
class VideoAssetSaveResult:
    """영상 메타데이터 저장 결과와 신규 생성 여부."""

    asset: VideoAssetRecord
    created: bool


@dataclass(frozen=True, slots=True)
class InferenceJobRecord:
    """Redis에 저장되는 비동기 추론 Job 상태."""

    job_id: str
    utterance_id: int
    video_id: int
    object_key: str
    mode: RecognitionMode
    status: InferenceJobStatus
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceJobEnqueueResult:
    """추론 Job enqueue 결과와 신규 생성 여부."""

    job: InferenceJobRecord
    created: bool


class RecognitionRepository(Protocol):
    """인식 세션과 최종 발화를 저장하는 포트."""

    async def create_session(self, mode: RecognitionMode) -> int: ...

    async def complete_session(
        self,
        session_id: int,
        output: RecognitionOutput,
    ) -> int: ...

    async def end_session(self, session_id: int) -> None: ...


class VideoUploadRepository(Protocol):
    """비동기 영상 업로드 메타데이터를 저장하는 포트."""

    async def find_video_asset_by_idempotency_key(
        self,
        *,
        user_id: int,
        idempotency_key: str,
    ) -> VideoAssetRecord | None: ...

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
    ) -> VideoAssetSaveResult: ...


class InferenceJobQueue(Protocol):
    """비동기 영상 추론 Job을 queue에 등록하고 상태를 조회하는 포트."""

    async def enqueue_or_get(
        self,
        *,
        job_id: str,
        utterance_id: int,
        video_id: int,
        object_key: str,
        mode: RecognitionMode,
        created_at: datetime,
    ) -> InferenceJobEnqueueResult:
        """video_id 기준 중복 enqueue 없이 Job을 생성하거나 기존 Job을 반환한다."""
        ...

    async def get_job(
        self,
        *,
        job_id: str,
    ) -> InferenceJobRecord | None:
        """job_id에 대응하는 현재 추론 상태를 조회한다."""
        ...


class RecognitionGateway(Protocol):
    """구체적인 ML 프레임워크와 분리한 비동기 추론 포트."""

    @property
    def manifest(self) -> ModelManifest | None: ...

    @property
    def ready(self) -> bool: ...

    async def start(self) -> None: ...

    async def predict(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> Prediction: ...

    async def close(self) -> None: ...


class TextCorrector(Protocol):
    """인식 문자열을 선택적으로 교정하는 포트."""

    async def correct(self, text: str) -> str | None: ...


class FrameValidator(Protocol):
    """전송된 프레임이 입력 계약을 만족하는지 검증하는 포트."""

    async def validate(self, frame: bytes) -> None: ...

    async def close(self) -> None: ...


class ObjectStorage(Protocol):
    """인식 영상을 private Object Storage에 저장하는 포트."""

    async def ensure_bucket(self) -> None:
        """설정된 private bucket에 접근할 수 있는지 확인한다."""
        ...

    async def put(
        self,
        *,
        object_key: str,
        data: bytes,
        content_type: str,
        checksum: str,
    ) -> None:
        """객체를 저장하고 SHA-256 checksum metadata를 함께 기록한다."""
        ...

    async def get(
        self,
        object_key: str,
    ) -> bytes:
        """private 객체 내용을 읽는다."""
        ...

    async def exists(
        self,
        object_key: str,
    ) -> bool:
        """객체가 존재하는지 확인한다."""
        ...

    async def delete(
        self,
        object_key: str,
    ) -> None:
        """객체를 삭제한다."""
        ...
