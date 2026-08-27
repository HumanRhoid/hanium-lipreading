"""인식 애플리케이션과 외부 구현 사이의 포트."""

from typing import Protocol

from src.backend.recognition.domain import (
    ModelManifest,
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)


class RecognitionRepository(Protocol):
    """인식 세션과 최종 발화를 저장하는 포트."""

    async def create_session(self, mode: RecognitionMode) -> int: ...

    async def complete_session(
        self,
        session_id: int,
        output: RecognitionOutput,
    ) -> int: ...

    async def end_session(self, session_id: int) -> None: ...


class RecognitionGateway(Protocol):
    """구체적인 ML 프레임워크와 분리된 비동기 추론 포트."""

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
