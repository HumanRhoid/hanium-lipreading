"""Redis 추론 Job과 기존 RecognitionGateway 호출을 연결하는 Worker."""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Protocol

import cv2
import numpy as np

from src.ml.preprocess.normalize import (
    FIXED_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

from src.backend.recognition.ports import (
    InferenceJobRecord,
    InferenceJobWorkerQueue,
    ObjectStorage,
    RecognitionGateway,
)

logger = logging.getLogger(__name__)

_SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".webm"})


class StoredVideoPreprocessor(Protocol):
    """Object Storage 영상 bytes를 기존 inference 입력 JPEG 프레임으로 변환한다."""

    async def preprocess(
        self,
        *,
        video_data: bytes,
        object_key: str,
    ) -> tuple[bytes, ...]: ...


Clock = Callable[[], datetime]
VideoProcessor = Callable[[Path, object, int], np.ndarray | None]
LandmarkerFactory = Callable[[], object]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_video_processor(
    video_path: Path,
    landmarker: object,
    frames: int,
) -> np.ndarray | None:
    """기존 ML 영상 전처리 함수를 지연 import해 그대로 재사용한다."""

    from src.ml.preprocess.vid2npy import process_video

    return process_video(
        video_path,
        landmarker,
        frames,
    )


def _default_landmarker_factory() -> object:
    """기존 입술 landmark 생성 함수를 지연 import한다."""

    from src.ml.preprocess.lip_crop import create_landmarker

    return create_landmarker()


class MlStoredVideoPreprocessor:
    """기존 ML 전처리를 재사용해 RecognitionGateway 입력 JPEG를 만든다."""

    def __init__(
        self,
        *,
        processor: VideoProcessor = _default_video_processor,
        landmarker_factory: LandmarkerFactory = _default_landmarker_factory,
    ) -> None:
        self._processor = processor
        self._landmarker_factory = landmarker_factory

    async def preprocess(
        self,
        *,
        video_data: bytes,
        object_key: str,
    ) -> tuple[bytes, ...]:
        """업로드 영상을 기존 전처리 후 gateway 입력 JPEG 프레임으로 변환한다."""

        if not video_data:
            raise ValueError("video_data는 비어 있을 수 없습니다.")

        suffix = PurePosixPath(object_key.strip()).suffix.lower()

        if suffix not in _SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError("추론 영상은 MP4 또는 WebM이어야 합니다.")

        return await asyncio.to_thread(
            self._preprocess_sync,
            video_data,
            suffix,
        )

    def _preprocess_sync(
        self,
        video_data: bytes,
        suffix: str,
    ) -> tuple[bytes, ...]:
        temporary_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(video_data)
                temporary_path = Path(temporary_file.name)

            landmarker = self._landmarker_factory()

            try:
                normalized = self._processor(
                    temporary_path,
                    landmarker,
                    FIXED_FRAME_COUNT,
                )
            finally:
                close = getattr(
                    landmarker,
                    "close",
                    None,
                )

                if callable(close):
                    close()

            if normalized is None:
                raise RuntimeError("업로드 영상에서 입술 ROI를 추출할 수 없습니다.")

            # 전처리 결과는 모델 입력 규격(크롭)이지 전송 규격이 아니다.
            expected_shape = (
                FIXED_FRAME_COUNT,
                TARGET_HEIGHT,
                TARGET_WIDTH,
                3,
            )

            if tuple(normalized.shape) != expected_shape:
                raise RuntimeError(
                    "기존 영상 전처리 결과가 inference 입력 규격과 일치하지 않습니다."
                )

            encoded_frames: list[bytes] = []

            for frame in normalized:
                encoded, payload = cv2.imencode(
                    ".jpg",
                    frame,
                )

                if not encoded:
                    raise RuntimeError(
                        "전처리된 프레임을 inference 입력 JPEG로 변환할 수 없습니다."
                    )

                encoded_frames.append(payload.tobytes())

            return tuple(encoded_frames)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class InferenceWorker:
    """Redis Job을 소비해 기존 전처리와 RecognitionGateway 호출까지 수행한다."""

    def __init__(
        self,
        *,
        queue: InferenceJobWorkerQueue,
        object_storage: ObjectStorage,
        preprocessor: StoredVideoPreprocessor,
        gateway: RecognitionGateway,
        consumer_name: str,
        block_ms: int = 1000,
        clock: Clock = _utc_now,
    ) -> None:
        consumer_name = consumer_name.strip()

        if not consumer_name:
            raise ValueError("consumer_name은 비어 있을 수 없습니다.")

        if block_ms <= 0:
            raise ValueError("block_ms는 양수여야 합니다.")

        self._queue = queue
        self._object_storage = object_storage
        self._preprocessor = preprocessor
        self._gateway = gateway
        self._consumer_name = consumer_name
        self._block_ms = block_ms
        self._clock = clock

    async def run_forever(self) -> None:
        """Consumer Group을 준비하고 Job을 계속 기다리며 처리한다."""

        await self._queue.ensure_consumer_group()

        while True:
            await self.process_once()

    async def process_once(self) -> bool:
        """Job 하나를 기존 inference 호출까지 처리한다."""

        delivery = await self._queue.read_next(
            consumer_name=self._consumer_name,
            block_ms=self._block_ms,
        )

        if delivery is None:
            return False

        job = delivery.job

        await self._queue.mark_processing(
            job_id=job.job_id,
            updated_at=self._clock(),
        )

        try:
            video_data = await self._object_storage.get(job.object_key)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "추론 Worker 영상 조회 실패: error_type=%s",
                type(exc).__name__,
            )
            await self._finish_failed(
                job=job,
                stream_entry_id=delivery.stream_entry_id,
                error_code="OBJECT_STORAGE_READ_FAILED",
            )
            return True

        try:
            frames = await self._preprocessor.preprocess(
                video_data=video_data,
                object_key=job.object_key,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "추론 Worker 영상 전처리 실패: error_type=%s",
                type(exc).__name__,
            )
            await self._finish_failed(
                job=job,
                stream_entry_id=delivery.stream_entry_id,
                error_code="VIDEO_PREPROCESSING_FAILED",
            )
            return True

        try:
            # 실제 모델 내부 구현은 gateway 뒤쪽의 predictor가 담당한다.
            # Worker 책임은 기존 inference 인터페이스를 호출하는 여기까지다.
            await self._gateway.predict(
                frames,
                job.mode,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "추론 Worker inference 호출 실패: error_type=%s",
                type(exc).__name__,
            )
            await self._finish_failed(
                job=job,
                stream_entry_id=delivery.stream_entry_id,
                error_code="INFERENCE_FAILED",
            )
            return True

        await self._queue.mark_succeeded(
            job_id=job.job_id,
            updated_at=self._clock(),
        )

        await self._queue.acknowledge(
            stream_entry_id=delivery.stream_entry_id,
        )

        return True

    async def _finish_failed(
        self,
        *,
        job: InferenceJobRecord,
        stream_entry_id: str,
        error_code: str,
    ) -> None:
        """실패 상태를 먼저 저장한 뒤에만 Redis delivery를 ACK한다."""

        await self._queue.mark_failed(
            job_id=job.job_id,
            error_code=error_code,
            updated_at=self._clock(),
        )

        await self._queue.acknowledge(
            stream_entry_id=stream_entry_id,
        )
