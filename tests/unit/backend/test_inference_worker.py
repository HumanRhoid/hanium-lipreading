"""InferenceWorker와 업로드 영상 전처리 연결 단위 테스트."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.backend.recognition.domain import (
    Prediction,
    RecognitionMode,
)
from src.backend.recognition.inference_worker import (
    InferenceWorker,
    MlStoredVideoPreprocessor,
)
from src.backend.recognition.ports import (
    InferenceJobDelivery,
    InferenceJobRecord,
)
from src.ml.preprocess.normalize import (
    FIXED_FRAME_COUNT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

JOB_ID = "11111111-2222-4333-8444-555555555555"
STREAM_ENTRY_ID = "1700000000000-0"
OBJECT_KEY = "11111111-2222-4333-8444-555555555555/2026/08/video.webm"

CREATED_AT = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
PROCESSING_AT = datetime(2026, 8, 27, 10, 1, tzinfo=UTC)
TERMINAL_AT = datetime(2026, 8, 27, 10, 2, tzinfo=UTC)


def make_job() -> InferenceJobRecord:
    return InferenceJobRecord(
        job_id=JOB_ID,
        utterance_id=123,
        video_id=45,
        object_key=OBJECT_KEY,
        mode=RecognitionMode.CLOSED,
        status="QUEUED",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
        error_code=None,
    )


def make_delivery() -> InferenceJobDelivery:
    return InferenceJobDelivery(
        stream_entry_id=STREAM_ENTRY_ID,
        job=make_job(),
    )


class FakeQueue:
    def __init__(
        self,
        *,
        delivery: InferenceJobDelivery | None = None,
        mark_succeeded_error: Exception | None = None,
    ) -> None:
        self.delivery = delivery
        self.mark_succeeded_error = mark_succeeded_error
        self.calls: list[tuple[str, object]] = []

    async def ensure_consumer_group(self) -> None:
        self.calls.append(("ensure_consumer_group", None))

    async def read_next(
        self,
        *,
        consumer_name: str,
        block_ms: int,
    ) -> InferenceJobDelivery | None:
        self.calls.append(
            (
                "read_next",
                {
                    "consumer_name": consumer_name,
                    "block_ms": block_ms,
                },
            )
        )
        await asyncio.sleep(0)
        return self.delivery

    async def mark_processing(
        self,
        *,
        job_id: str,
        updated_at: datetime,
    ) -> InferenceJobRecord:
        self.calls.append(
            (
                "mark_processing",
                {
                    "job_id": job_id,
                    "updated_at": updated_at,
                },
            )
        )
        job = make_job()
        return InferenceJobRecord(
            job_id=job.job_id,
            utterance_id=job.utterance_id,
            video_id=job.video_id,
            object_key=job.object_key,
            mode=job.mode,
            status="PROCESSING",
            created_at=job.created_at,
            updated_at=updated_at,
            error_code=None,
        )

    async def mark_succeeded(
        self,
        *,
        job_id: str,
        updated_at: datetime,
    ) -> InferenceJobRecord:
        self.calls.append(
            (
                "mark_succeeded",
                {
                    "job_id": job_id,
                    "updated_at": updated_at,
                },
            )
        )

        if self.mark_succeeded_error is not None:
            raise self.mark_succeeded_error

        job = make_job()
        return InferenceJobRecord(
            job_id=job.job_id,
            utterance_id=job.utterance_id,
            video_id=job.video_id,
            object_key=job.object_key,
            mode=job.mode,
            status="SUCCEEDED",
            created_at=job.created_at,
            updated_at=updated_at,
            error_code=None,
        )

    async def mark_failed(
        self,
        *,
        job_id: str,
        error_code: str,
        updated_at: datetime,
    ) -> InferenceJobRecord:
        self.calls.append(
            (
                "mark_failed",
                {
                    "job_id": job_id,
                    "error_code": error_code,
                    "updated_at": updated_at,
                },
            )
        )
        job = make_job()
        return InferenceJobRecord(
            job_id=job.job_id,
            utterance_id=job.utterance_id,
            video_id=job.video_id,
            object_key=job.object_key,
            mode=job.mode,
            status="FAILED",
            created_at=job.created_at,
            updated_at=updated_at,
            error_code=error_code,
        )

    async def acknowledge(
        self,
        *,
        stream_entry_id: str,
    ) -> None:
        self.calls.append(("acknowledge", stream_entry_id))


class FakeObjectStorage:
    def __init__(
        self,
        *,
        data: bytes = b"stored-video",
        error: Exception | None = None,
    ) -> None:
        self.data = data
        self.error = error
        self.keys: list[str] = []

    async def get(
        self,
        object_key: str,
    ) -> bytes:
        self.keys.append(object_key)

        if self.error is not None:
            raise self.error

        return self.data


class FakePreprocessor:
    def __init__(
        self,
        *,
        frames: tuple[bytes, ...] = (b"frame-1", b"frame-2"),
        error: BaseException | None = None,
    ) -> None:
        self.frames = frames
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def preprocess(
        self,
        *,
        video_data: bytes,
        object_key: str,
    ) -> tuple[bytes, ...]:
        self.calls.append(
            {
                "video_data": video_data,
                "object_key": object_key,
            }
        )

        if self.error is not None:
            raise self.error

        return self.frames


class FakeGateway:
    ready = True
    manifest = None

    def __init__(
        self,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def start(self) -> None:
        pass

    async def predict(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> Prediction:
        self.calls.append(
            {
                "frames": frames,
                "mode": mode,
            }
        )

        if self.error is not None:
            raise self.error

        return Prediction(
            text="테스트",
            confidence=0.9,
            phrase_code="TEST",
        )

    async def close(self) -> None:
        pass


class FakeResultRepository:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.saved: list[dict[str, object]] = []

    async def save_inference_result(
        self,
        *,
        utterance_id: int,
        prediction: Prediction,
        model_version: str | None,
    ):
        if self.error is not None:
            raise self.error

        self.saved.append(
            {
                "utterance_id": utterance_id,
                "prediction": prediction,
                "model_version": model_version,
            }
        )



class FakeClock:
    def __init__(self) -> None:
        self.values = iter((PROCESSING_AT, TERMINAL_AT))

    def __call__(self) -> datetime:
        return next(self.values)


class FakeLandmarker:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


async def test_ml_preprocessor_reuses_existing_video_processor_and_encodes_jpeg():
    captured: dict[str, object] = {}
    landmarker = FakeLandmarker()

    def processor(
        video_path: Path,
        received_landmarker: object,
        frames: int,
    ) -> np.ndarray:
        captured["path"] = video_path
        captured["suffix"] = video_path.suffix
        captured["data"] = video_path.read_bytes()
        captured["landmarker"] = received_landmarker
        captured["frames"] = frames

        return np.zeros(
            (
                FIXED_FRAME_COUNT,
                TARGET_HEIGHT,
                TARGET_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

    preprocessor = MlStoredVideoPreprocessor(
        processor=processor,
        landmarker_factory=lambda: landmarker,
    )

    frames = await preprocessor.preprocess(
        video_data=b"webm-video",
        object_key=OBJECT_KEY,
    )

    assert len(frames) == FIXED_FRAME_COUNT
    assert all(frame.startswith(b"\xff\xd8") for frame in frames)

    assert captured["suffix"] == ".webm"
    assert captured["data"] == b"webm-video"
    assert captured["landmarker"] is landmarker
    assert captured["frames"] == FIXED_FRAME_COUNT
    assert landmarker.closed is True

    temporary_path = captured["path"]
    assert isinstance(temporary_path, Path)
    assert temporary_path.exists() is False


@pytest.mark.parametrize(
    "object_key",
    [
        "video.avi",
        "video.mov",
        "video",
        "   ",
    ],
)
async def test_ml_preprocessor_rejects_unsupported_upload_suffix(
    object_key: str,
):
    preprocessor = MlStoredVideoPreprocessor(
        processor=lambda _path, _landmarker, _frames: None,
        landmarker_factory=FakeLandmarker,
    )

    with pytest.raises(
        ValueError,
        match="MP4 또는 WebM",
    ):
        await preprocessor.preprocess(
            video_data=b"video",
            object_key=object_key,
        )


async def test_ml_preprocessor_rejects_empty_video():
    preprocessor = MlStoredVideoPreprocessor(
        processor=lambda _path, _landmarker, _frames: None,
        landmarker_factory=FakeLandmarker,
    )

    with pytest.raises(
        ValueError,
        match="video_data",
    ):
        await preprocessor.preprocess(
            video_data=b"",
            object_key=OBJECT_KEY,
        )


async def test_ml_preprocessor_fails_when_existing_processor_cannot_find_lips():
    landmarker = FakeLandmarker()

    preprocessor = MlStoredVideoPreprocessor(
        processor=lambda _path, _landmarker, _frames: None,
        landmarker_factory=lambda: landmarker,
    )

    with pytest.raises(
        RuntimeError,
        match="입술 ROI",
    ):
        await preprocessor.preprocess(
            video_data=b"video",
            object_key=OBJECT_KEY,
        )

    assert landmarker.closed is True


async def test_ml_preprocessor_rejects_wrong_existing_preprocess_shape():
    wrong_shape = np.zeros(
        (1, 360, 640, 3),   # 크롭 규격(60, 96, 192, 3)이 아니기만 하면 된다
        dtype=np.uint8,
    )

    preprocessor = MlStoredVideoPreprocessor(
        processor=lambda _path, _landmarker, _frames: wrong_shape,
        landmarker_factory=FakeLandmarker,
    )

    with pytest.raises(
        RuntimeError,
        match="입력 규격",
    ):
        await preprocessor.preprocess(
            video_data=b"video",
            object_key=OBJECT_KEY,
        )


async def test_process_once_returns_false_when_queue_is_empty():
    queue = FakeQueue(delivery=None)
    storage = FakeObjectStorage()
    preprocessor = FakePreprocessor()
    gateway = FakeGateway()

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
    )

    processed = await worker.process_once()

    assert processed is False
    assert storage.keys == []
    assert preprocessor.calls == []
    assert gateway.calls == []


async def test_process_once_preprocesses_video_calls_existing_gateway_and_acks():
    queue = FakeQueue(delivery=make_delivery())
    storage = FakeObjectStorage(data=b"video-from-minio")
    preprocessor = FakePreprocessor(
        frames=(b"jpeg-1", b"jpeg-2"),
    )
    gateway = FakeGateway()
    repository = FakeResultRepository()

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=repository,
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    processed = await worker.process_once()

    assert processed is True
    assert storage.keys == [OBJECT_KEY]
    assert preprocessor.calls == [
        {
            "video_data": b"video-from-minio",
            "object_key": OBJECT_KEY,
        }
    ]
    assert gateway.calls == [
        {
            "frames": (b"jpeg-1", b"jpeg-2"),
            "mode": RecognitionMode.CLOSED,
        }
    ]
    assert repository.saved == [
        {
            "utterance_id": 123,
            "prediction": Prediction(
                text="테스트",
                confidence=0.9,
                phrase_code="TEST",
            ),
            "model_version": None,
        }
    ]

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_succeeded",
        "acknowledge",
    ]


async def test_storage_failure_marks_failed_without_preprocess_or_inference():
    queue = FakeQueue(delivery=make_delivery())
    storage = FakeObjectStorage(error=RuntimeError("minio unavailable"))
    preprocessor = FakePreprocessor()
    gateway = FakeGateway()

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    processed = await worker.process_once()

    assert processed is True
    assert preprocessor.calls == []
    assert gateway.calls == []

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_failed",
        "acknowledge",
    ]

    failed_payload = queue.calls[2][1]
    assert failed_payload["error_code"] == "OBJECT_STORAGE_READ_FAILED"


async def test_preprocess_failure_marks_failed_without_inference():
    queue = FakeQueue(delivery=make_delivery())
    storage = FakeObjectStorage()
    preprocessor = FakePreprocessor(error=RuntimeError("preprocess failed"))
    gateway = FakeGateway()

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    processed = await worker.process_once()

    assert processed is True
    assert gateway.calls == []

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_failed",
        "acknowledge",
    ]

    failed_payload = queue.calls[2][1]
    assert failed_payload["error_code"] == "VIDEO_PREPROCESSING_FAILED"


async def test_inference_gateway_failure_marks_failed_and_acks():
    queue = FakeQueue(delivery=make_delivery())
    storage = FakeObjectStorage()
    preprocessor = FakePreprocessor()
    gateway = FakeGateway(error=RuntimeError("model failed"))

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    processed = await worker.process_once()

    assert processed is True

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_failed",
        "acknowledge",
    ]

    failed_payload = queue.calls[2][1]
    assert failed_payload["error_code"] == "INFERENCE_FAILED"


async def test_result_persistence_failure_marks_failed_and_acks():
    queue = FakeQueue(delivery=make_delivery())
    repository = FakeResultRepository(error=RuntimeError("database unavailable"))

    worker = InferenceWorker(
        queue=queue,
        object_storage=FakeObjectStorage(),
        preprocessor=FakePreprocessor(),
        gateway=FakeGateway(),
        repository=repository,
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    processed = await worker.process_once()

    assert processed is True
    assert repository.saved == []
    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_failed",
        "acknowledge",
    ]

    failed_payload = queue.calls[2][1]
    assert failed_payload["error_code"] == "RESULT_PERSISTENCE_FAILED"


async def test_cancellation_during_gateway_call_does_not_mark_failed_or_ack():
    queue = FakeQueue(delivery=make_delivery())
    storage = FakeObjectStorage()
    preprocessor = FakePreprocessor()
    gateway = FakeGateway(error=asyncio.CancelledError())

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.process_once()

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
    ]


async def test_terminal_status_failure_does_not_ack():
    queue = FakeQueue(
        delivery=make_delivery(),
        mark_succeeded_error=RuntimeError("redis unavailable"),
    )
    storage = FakeObjectStorage()
    preprocessor = FakePreprocessor()
    gateway = FakeGateway()

    worker = InferenceWorker(
        queue=queue,
        object_storage=storage,
        preprocessor=preprocessor,
        gateway=gateway,
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
        clock=FakeClock(),
    )

    with pytest.raises(
        RuntimeError,
        match="redis unavailable",
    ):
        await worker.process_once()

    assert [name for name, _payload in queue.calls] == [
        "read_next",
        "mark_processing",
        "mark_succeeded",
    ]


async def test_run_forever_ensures_consumer_group_before_reading():
    queue = FakeQueue(delivery=None)
    worker = InferenceWorker(
        queue=queue,
        object_storage=FakeObjectStorage(),
        preprocessor=FakePreprocessor(),
        gateway=FakeGateway(),
        repository=FakeResultRepository(),
        consumer_name="worker-1",
        block_ms=100,
    )

    task = asyncio.create_task(worker.run_forever())

    while len(queue.calls) < 2:
        await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert queue.calls[0] == ("ensure_consumer_group", None)
    assert queue.calls[1][0] == "read_next"


@pytest.mark.parametrize(
    ("consumer_name", "block_ms", "message"),
    [
        ("   ", 100, "consumer_name"),
        ("worker-1", 0, "block_ms"),
        ("worker-1", -1, "block_ms"),
    ],
)
def test_constructor_rejects_invalid_worker_settings(
    consumer_name: str,
    block_ms: int,
    message: str,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        InferenceWorker(
            queue=FakeQueue(),
            object_storage=FakeObjectStorage(),
            preprocessor=FakePreprocessor(),
            gateway=FakeGateway(),
            repository=FakeResultRepository(),
            consumer_name=consumer_name,
            block_ms=block_ms,
        )
