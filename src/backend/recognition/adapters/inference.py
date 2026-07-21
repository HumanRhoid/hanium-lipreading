"""동기 predictor와 이를 제한 실행하는 비동기 gateway."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from src.backend.recognition.domain import (
    INPUT_FRAME_COUNT,
    INPUT_FRAME_FPS,
    INPUT_FRAME_HEIGHT,
    INPUT_FRAME_WIDTH,
    ModelManifest,
    Prediction,
    RecognitionMode,
)
from src.backend.recognition.errors import InferenceBusyError, ModelNotReadyError


class SyncPredictor(Protocol):
    """전용 worker에서만 호출할 동기 전처리·모델 파이프라인."""

    @property
    def manifest(self) -> ModelManifest: ...

    @property
    def ready(self) -> bool: ...

    def start(self) -> None: ...

    def predict(
        self,
        frames: Sequence[bytes],
        mode: RecognitionMode,
    ) -> Prediction: ...

    def close(self) -> None: ...


class _AsyncCapacity:
    def __init__(self, limit: int):
        self._limit = limit
        self._active = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active -= 1


class BoundedLocalRecognitionGateway:
    """호출 취소와 무관하게 실제 동기 작업 종료 후 capacity를 반환한다."""

    def __init__(self, predictor: SyncPredictor, *, max_concurrency: int):
        self._predictor = predictor
        self._capacity = _AsyncCapacity(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="lipreading-inference",
        )
        self._running: set[asyncio.Task[Prediction]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._start_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._closed = False

    @property
    def manifest(self) -> ModelManifest:
        return self._predictor.manifest

    @property
    def ready(self) -> bool:
        return (
            self._started
            and not self._closing
            and not self._closed
            and self._predictor.ready
        )

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[None]) -> None:
        """호출자가 취소된 뒤 발생한 background 예외도 회수한다."""

        if not task.cancelled():
            task.exception()

    def _on_prediction_done(self, task: asyncio.Task[Prediction]) -> None:
        self._running.discard(task)
        if not task.cancelled():
            task.exception()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("종료 중이거나 종료된 추론 gateway입니다")
            if self._start_task is None:
                self._start_task = asyncio.create_task(self._start())
                self._start_task.add_done_callback(self._consume_task_exception)
            task = self._start_task

        await asyncio.shield(task)

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._predictor.start)
        self._started = True

    async def predict(
        self, frames: tuple[bytes, ...], mode: RecognitionMode
    ) -> Prediction:
        async with self._lifecycle_lock:
            if not self.ready:
                raise ModelNotReadyError("인식 모델이 준비되지 않았습니다")
            if not await self._capacity.try_acquire():
                raise InferenceBusyError("추론 worker가 사용 중입니다")

            async def execute() -> Prediction:
                try:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        self._predictor.predict,
                        frames,
                        mode,
                    )
                finally:
                    await self._capacity.release()

            task = asyncio.create_task(execute())
            self._running.add(task)
            task.add_done_callback(self._on_prediction_done)

        return await asyncio.shield(task)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close())
                self._close_task.add_done_callback(self._consume_task_exception)
            task = self._close_task

        await asyncio.shield(task)

    async def _close(self) -> None:
        start_task = self._start_task
        if start_task is not None:
            await asyncio.gather(start_task, return_exceptions=True)
        if self._running:
            await asyncio.gather(*tuple(self._running), return_exceptions=True)
        try:
            if self._started:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._predictor.close)
        finally:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            finally:
                self._started = False
                self._closed = True


class UnavailableRecognitionGateway:
    """실제 모델 bundle이 없을 때 사용하는 명시적 비가용 gateway."""

    ready = False
    manifest = None

    async def start(self) -> None:
        pass

    async def predict(
        self, frames: tuple[bytes, ...], mode: RecognitionMode
    ) -> Prediction:
        raise ModelNotReadyError("인식 모델이 준비되지 않았습니다")

    async def close(self) -> None:
        pass


class FakeSyncPredictor:
    """local/test에서 결정적인 데모 결과를 반환한다."""

    manifest = ModelManifest(
        bundle_version="fake-v1",
        supported_modes=frozenset(RecognitionMode),
        frame_width=INPUT_FRAME_WIDTH,
        frame_height=INPUT_FRAME_HEIGHT,
        fps=INPUT_FRAME_FPS,
        input_frame_count=INPUT_FRAME_COUNT,
        label_map_version="demo-v1",
    )

    def __init__(self):
        self.ready = False

    def start(self) -> None:
        self.ready = True

    def predict(self, frames: Sequence[bytes], mode: RecognitionMode) -> Prediction:
        if not self.ready:
            raise ModelNotReadyError("fake predictor가 시작되지 않았습니다")
        if len(frames) != INPUT_FRAME_COUNT:
            raise ValueError(f"정확히 {INPUT_FRAME_COUNT}프레임이 필요합니다")
        if mode == RecognitionMode.CLOSED:
            return Prediction("물 주세요", 0.91, "REQUEST_WATER")
        return Prediction("안녕하세요", 0.8)

    def close(self) -> None:
        self.ready = False
