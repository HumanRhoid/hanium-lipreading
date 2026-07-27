"""최종 인식 실행과 buffered session 영속성 순서를 조정한다."""

import asyncio
import logging
from enum import Enum

from src.backend.recognition.domain import (
    INPUT_FRAME_COUNT,
    INPUT_FRAME_FPS,
    INPUT_FRAME_HEIGHT,
    INPUT_FRAME_WIDTH,
    ModelManifest,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import (
    ModelNotReadyError,
    SessionBusyError,
    SessionClosedError,
    UnsupportedRecognitionModeError,
)
from src.backend.recognition.frame_policy import (
    normalize_video_frames,
    validate_video_limits,
)
from src.backend.recognition.ports import (
    RecognitionGateway,
    RecognitionRepository,
    TextCorrector,
)

logger = logging.getLogger(__name__)


class _TerminalState(Enum):
    ACTIVE = "active"
    STOPPING = "stopping"
    DISCONNECTING = "disconnecting"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class _AsyncCapacity:
    """대기열 없이 즉시 성공·실패하는 세션 capacity."""

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
            if self._active <= 0:
                raise RuntimeError("세션 capacity가 중복 반환됐습니다")
            self._active -= 1


class NoopTextCorrector:
    """LLM 교정기가 준비되기 전 원문을 변경하지 않는다."""

    async def correct(self, text: str) -> str | None:
        return None


class RecognitionService:
    """DB transaction과 stop-triggered 추론 수명을 분리하는 서비스."""

    def __init__(
        self,
        *,
        repository: RecognitionRepository,
        gateway: RecognitionGateway,
        corrector: TextCorrector,
        max_active_sessions: int = 1,
    ):
        self._repository = repository
        self._gateway = gateway
        self._corrector = corrector
        self._session_capacity = _AsyncCapacity(max_active_sessions)
        self._manifest: ModelManifest | None = getattr(gateway, "manifest", None)
        self._session_open_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._validate_model_manifest()

    def _validate_model_manifest(self) -> None:
        """모델 입력 계약과 공개 스트림 계약의 불일치를 조기에 차단한다."""

        if self._manifest is None:
            return
        expected_values = {
            "frame_width": INPUT_FRAME_WIDTH,
            "frame_height": INPUT_FRAME_HEIGHT,
            "fps": INPUT_FRAME_FPS,
            "input_frame_count": INPUT_FRAME_COUNT,
            "input_codec": "image/jpeg",
        }
        mismatches = [
            field_name
            for field_name, expected in expected_values.items()
            if getattr(self._manifest, field_name) != expected
        ]
        if mismatches:
            fields = ", ".join(mismatches)
            raise ValueError(
                f"모델 manifest가 서버 입력 계약과 일치하지 않습니다: {fields}"
            )

    def _ensure_mode_supported(self, mode: RecognitionMode) -> None:
        if self._manifest is not None and mode not in self._manifest.supported_modes:
            raise UnsupportedRecognitionModeError(
                "현재 모델이 요청한 인식 모드를 지원하지 않습니다"
            )

    def _on_open_cleanup_done(self, task: asyncio.Task[None]) -> None:
        self._session_open_cleanup_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "인식 스트림 시작 실패 정리 중 오류: error_type=%s",
                type(exception).__name__,
            )

    async def _release_failed_open(self) -> None:
        """호출자가 반복 취소돼도 capacity 반환 작업은 끝까지 유지한다."""

        cleanup = asyncio.create_task(self._session_capacity.release())
        self._session_open_cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._on_open_cleanup_done)
        await asyncio.shield(cleanup)

    async def open_session(
        self,
        mode: RecognitionMode,
    ) -> "BufferedRecognitionSession":
        """모델과 세션 capacity를 대기 없이 검증하고 세션을 연다."""

        self._ensure_mode_supported(mode)
        if not self._gateway.ready:
            raise ModelNotReadyError("인식 모델이 준비되지 않았습니다")
        if not await self._session_capacity.try_acquire():
            raise SessionBusyError("활성 인식 세션이 이미 사용 중입니다")

        try:
            session_id = await self._repository.create_session(mode)
        except BaseException:
            await self._release_failed_open()
            raise

        return BufferedRecognitionSession(
            service=self,
            session_id=session_id,
            mode=mode,
        )

    async def start_session(self, mode: RecognitionMode) -> int:
        return await self._repository.create_session(mode)

    async def recognize(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> RecognitionOutput:
        self._ensure_mode_supported(mode)
        prediction = await self._gateway.predict(frames, mode)
        corrected_text = await self._corrector.correct(prediction.text)
        return RecognitionOutput(
            raw_text=prediction.text,
            corrected_text=corrected_text,
            confidence=prediction.confidence,
            phrase_code=prediction.phrase_code,
        )

    async def complete_session(
        self,
        session_id: int,
        output: RecognitionOutput,
    ) -> int:
        return await self._repository.complete_session(session_id, output)

    async def end_session(self, session_id: int) -> None:
        await self._repository.end_session(session_id)

    async def _release_session(self) -> None:
        await self._session_capacity.release()


class BufferedRecognitionSession:
    """검증된 JPEG를 bounded buffer에 모아 stop에서 한 번 추론한다."""

    def __init__(
        self,
        *,
        service: RecognitionService,
        session_id: int,
        mode: RecognitionMode,
    ):
        self._service = service
        self._session_id = session_id
        self._mode = mode
        self._frames: list[bytes] = []
        self._total_bytes = 0
        self._terminal_lock = asyncio.Lock()
        self._terminal_task: asyncio.Task[None] | None = None
        self._terminal_state = _TerminalState.ACTIVE
        self._terminal_result: RecognitionOutput | None = None
        self._terminal_failure: BaseException | None = None
        self._accepting = True

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def buffered_frame_count(self) -> int:
        return len(self._frames)

    @property
    def buffered_bytes(self) -> int:
        return self._total_bytes

    async def push_frame(self, frame: bytes) -> None:
        """검증이 끝난 JPEG를 상한 안에서 메모리에 보관한다."""

        if not self._accepting:
            raise SessionClosedError("이미 종료 중인 인식 세션입니다")

        next_frame_count = len(self._frames) + 1
        next_total_bytes = self._total_bytes + len(frame)
        validate_video_limits(
            frame_count=next_frame_count,
            total_bytes=next_total_bytes,
        )
        self._frames.append(frame)
        self._total_bytes = next_total_bytes

    async def stop(self) -> RecognitionOutput:
        """buffer 전체를 정규화해 추론·교정·저장을 정확히 한 번 수행한다."""

        async with self._terminal_lock:
            if self._terminal_state is _TerminalState.ACTIVE:
                self._accepting = False
                self._terminal_state = _TerminalState.STOPPING
                self._terminal_task = asyncio.create_task(self._finish_stop())
            elif self._terminal_state is _TerminalState.DISCONNECTING:
                raise SessionClosedError("연결 종료 중인 인식 세션입니다")
            elif self._terminal_state is _TerminalState.ABORTED:
                raise SessionClosedError("이미 연결이 종료된 인식 세션입니다")

            task = self._terminal_task

        if task is not None:
            await asyncio.shield(task)
        return self._stop_outcome()

    async def disconnect(self) -> None:
        """stop 전에는 폐기하고 stop 뒤에는 terminal task를 끝까지 기다린다."""

        async with self._terminal_lock:
            if self._terminal_state is _TerminalState.ACTIVE:
                self._accepting = False
                self._clear_buffer()
                self._terminal_state = _TerminalState.DISCONNECTING
                self._terminal_task = asyncio.create_task(self._finish_disconnect())
            task = self._terminal_task

        if task is not None:
            await asyncio.shield(task)

    def _stop_outcome(self) -> RecognitionOutput:
        if self._terminal_state is _TerminalState.COMPLETED:
            if self._terminal_result is None:
                raise RuntimeError("최종 인식 결과가 누락됐습니다")
            return self._terminal_result
        if self._terminal_failure is not None:
            raise self._terminal_failure
        raise SessionClosedError("이미 종료된 인식 세션입니다")

    async def _finish_stop(self) -> None:
        failure: BaseException | None = None
        try:
            frames = self._take_normalized_frames()
            output = await self._service.recognize(frames, self._mode)
            await self._service.complete_session(self._session_id, output)
            self._terminal_result = output
            self._terminal_state = _TerminalState.COMPLETED
        except BaseException as exc:
            failure = exc
            try:
                await self._service.end_session(self._session_id)
            except BaseException as cleanup_exc:
                logger.error(
                    "실패한 인식 세션 종료 정리 실패: error_type=%s",
                    type(cleanup_exc).__name__,
                )
            self._terminal_failure = failure
            self._terminal_state = _TerminalState.FAILED
        finally:
            self._clear_buffer()
            await self._service._release_session()

    async def _finish_disconnect(self) -> None:
        try:
            await self._service.end_session(self._session_id)
            self._terminal_state = _TerminalState.ABORTED
        except BaseException as exc:
            self._terminal_failure = exc
            self._terminal_state = _TerminalState.FAILED
            logger.error(
                "인식 세션 종료 정리 실패: error_type=%s",
                type(exc).__name__,
            )
        finally:
            self._clear_buffer()
            await self._service._release_session()

    def _take_normalized_frames(self) -> tuple[bytes, ...]:
        buffered_frames = self._frames
        self._frames = []
        self._total_bytes = 0
        try:
            return normalize_video_frames(buffered_frames)
        finally:
            buffered_frames.clear()

    def _clear_buffer(self) -> None:
        self._frames.clear()
        self._total_bytes = 0
