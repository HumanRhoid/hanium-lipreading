"""WebSocket JPEG 프레임 검증기."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from src.backend.recognition.domain import INPUT_FRAME_HEIGHT, INPUT_FRAME_WIDTH
from src.backend.recognition.errors import FrameValidationBusyError, InvalidFrameError

_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


class JpegFrameValidator:
    """OpenCV 디코딩을 event loop 밖의 전용 worker에서 실행한다."""

    def __init__(
        self,
        *,
        width: int = INPUT_FRAME_WIDTH,
        height: int = INPUT_FRAME_HEIGHT,
    ):
        self._width = width
        self._height = height
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lipreading-jpeg",
        )
        self._running: set[asyncio.Future[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    @staticmethod
    def _consume_close_exception(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    def _on_validation_done(self, task: asyncio.Future[None]) -> None:
        self._running.discard(task)
        if not task.cancelled():
            task.exception()

    async def validate(self, payload: bytes) -> None:
        """JPEG 형식과 디코딩 결과의 해상도를 검증한다."""

        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("JPEG 검증기가 종료 중이거나 이미 종료됐습니다")
            if self._running:
                raise FrameValidationBusyError("JPEG 검증 worker가 사용 중입니다")

            loop = asyncio.get_running_loop()
            task = asyncio.ensure_future(
                loop.run_in_executor(self._executor, self._validate_sync, payload)
            )
            self._running.add(task)
            task.add_done_callback(self._on_validation_done)

        await asyncio.shield(task)

    def _validate_sync(self, payload: bytes) -> None:
        width, height = self._read_dimensions(payload)
        if width != self._width or height != self._height:
            raise InvalidFrameError(
                f"JPEG 해상도는 {self._width}x{self._height}여야 합니다"
            )

        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidFrameError("JPEG을 디코딩할 수 없습니다")

        decoded_height, decoded_width = image.shape[:2]
        if decoded_width != width or decoded_height != height:
            raise InvalidFrameError("JPEG header와 디코딩 해상도가 다릅니다")

    @staticmethod
    def _read_dimensions(payload: bytes) -> tuple[int, int]:
        """메모리 할당 전 SOF segment에서 JPEG 해상도를 읽는다."""

        if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
            raise InvalidFrameError("JPEG header가 올바르지 않습니다")

        position = 2
        payload_size = len(payload)
        while position < payload_size:
            if payload[position] != 0xFF:
                raise InvalidFrameError("JPEG marker가 올바르지 않습니다")
            while position < payload_size and payload[position] == 0xFF:
                position += 1
            if position >= payload_size:
                break

            marker = payload[position]
            position += 1
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if marker in {0xD9, 0xDA}:
                break
            if position + 2 > payload_size:
                break

            segment_length = int.from_bytes(payload[position : position + 2], "big")
            if segment_length < 2 or position + segment_length > payload_size:
                break
            if marker in _START_OF_FRAME_MARKERS:
                if segment_length < 7:
                    break
                height = int.from_bytes(payload[position + 3 : position + 5], "big")
                width = int.from_bytes(payload[position + 5 : position + 7], "big")
                if width <= 0 or height <= 0:
                    break
                return width, height
            position += segment_length

        raise InvalidFrameError("JPEG 해상도 header를 확인할 수 없습니다")

    async def close(self) -> None:
        """실행 중인 디코딩을 정리한 뒤 worker를 종료한다."""

        async with self._lifecycle_lock:
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close())
                self._close_task.add_done_callback(self._consume_close_exception)
            task = self._close_task

        await asyncio.shield(task)

    async def _close(self) -> None:
        if self._running:
            await asyncio.gather(*tuple(self._running), return_exceptions=True)
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        finally:
            self._closed = True
