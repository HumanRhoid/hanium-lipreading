"""stop-triggered 최종 인식 WebSocket의 wire contract adapter."""

import asyncio
import json
import logging
from typing import NoReturn

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.backend.recognition.domain import (
    MAX_FRAME_BYTES,
    RecognitionMode,
)
from src.backend.recognition.errors import (
    FrameValidationBusyError,
    InferenceBusyError,
    InsufficientFramesError,
    InvalidFrameError,
    ModelNotReadyError,
    SessionBusyError,
    SessionClosedError,
    UnsupportedRecognitionModeError,
    VideoTooLargeError,
    VideoTooLongError,
)
from src.backend.recognition.schemas import (
    ErrorEvent,
    ReadyEvent,
    ResultEvent,
    StartCommand,
    StopCommand,
    StoppedEvent,
    StrictSchema,
    parse_client_command,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["recognition"])


def _log_safe_error(context: str, exc: Exception) -> None:
    """예외 메시지·DB parameter 없이 타입만 기록한다."""

    logger.error("%s: error_type=%s", context, type(exc).__name__)


class _WireError(Exception):
    """외부에 노출할 안정적인 오류 코드와 close code."""

    def __init__(self, code: str, message: str, close_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


def _invalid_message(message: str = "올바른 명령 형식이 아닙니다.") -> _WireError:
    return _WireError("INVALID_MESSAGE", message, 1008)


def _parse_command(payload: str):
    """지원하지 않는 mode와 일반 schema 오류를 구분한다."""

    try:
        return parse_client_command(payload)
    except ValidationError as exc:
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            raise _invalid_message() from exc

        supported_modes = {mode.value for mode in RecognitionMode}
        if (
            isinstance(value, dict)
            and value.get("type") == "start"
            and "mode" in value
            and isinstance(value["mode"], str)
            and value["mode"] not in supported_modes
        ):
            raise _WireError(
                "UNSUPPORTED_MODE",
                "지원하지 않는 인식 모드입니다.",
                1008,
            ) from exc
        raise _invalid_message() from exc


def _text_message(message: dict) -> str:
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if text is None or message.get("bytes") is not None:
        raise _invalid_message("현재 상태에서 text 명령이 필요합니다.")
    return text


def _event_payload(event: StrictSchema) -> str:
    return event.model_dump_json(exclude_none=True)


async def _send_event(
    websocket: WebSocket,
    event: StrictSchema,
    send_lock: asyncio.Lock,
    timeout: float,
) -> bool:
    """lock 대기까지 포함해 한 outbound event의 시간을 제한한다."""

    async with asyncio.timeout(timeout):
        async with send_lock:
            await websocket.send_text(_event_payload(event))
    return True


async def _send_event_best_effort(
    websocket: WebSocket,
    event: StrictSchema,
    send_lock: asyncio.Lock,
    timeout: float,
    *,
    context: str,
) -> bool:
    """종료 event 전송 실패가 다음 close나 세션 정리를 막지 않게 한다."""

    try:
        return await _send_event(websocket, event, send_lock, timeout)
    except Exception as exc:
        _log_safe_error(context, exc)
        return False


async def _close_best_effort(
    websocket: WebSocket,
    close_code: int,
    send_lock: asyncio.Lock,
    timeout: float,
) -> None:
    """peer 상태와 무관하게 close를 제한시간 안에서 한 번 시도한다."""

    try:
        async with asyncio.timeout(timeout):
            async with send_lock:
                await websocket.close(code=close_code)
    except Exception as exc:
        _log_safe_error("인식 WebSocket close 실패", exc)


async def _send_error_and_close(
    websocket: WebSocket,
    error: _WireError,
    send_lock: asyncio.Lock,
    timeout: float,
) -> None:
    await _send_event_best_effort(
        websocket,
        ErrorEvent(code=error.code, message=error.message),
        send_lock,
        timeout,
        context="인식 WebSocket 오류 응답 실패",
    )
    await _close_best_effort(websocket, error.close_code, send_lock, timeout)


def _to_wire_error(exc: Exception) -> _WireError:
    if isinstance(exc, UnsupportedRecognitionModeError):
        return _WireError(
            "UNSUPPORTED_MODE",
            "지원하지 않는 인식 모드입니다.",
            1008,
        )
    if isinstance(exc, ModelNotReadyError):
        return _WireError(
            "MODEL_NOT_READY",
            "인식 모델이 준비되지 않았습니다.",
            1013,
        )
    if isinstance(
        exc,
        (FrameValidationBusyError, InferenceBusyError, SessionBusyError),
    ):
        return _WireError(
            "SERVER_BUSY",
            "서버가 다른 인식 요청을 처리 중입니다.",
            1013,
        )
    if isinstance(exc, InvalidFrameError):
        return _WireError(
            "INVALID_FRAME",
            "JPEG 프레임을 확인할 수 없습니다.",
            1003,
        )
    if isinstance(exc, VideoTooLongError):
        return _WireError(
            "VIDEO_TOO_LONG",
            "영상 프레임 수가 허용 범위를 넘었습니다.",
            1008,
        )
    if isinstance(exc, VideoTooLargeError):
        return _WireError(
            "VIDEO_TOO_LARGE",
            "영상 전체 크기가 허용 범위를 넘었습니다.",
            1009,
        )
    if isinstance(exc, SessionClosedError):
        return _invalid_message("이미 종료 중인 인식 스트림입니다.")
    return _WireError(
        "INTERNAL_ERROR",
        "인식 처리 중 서버 오류가 발생했습니다.",
        1011,
    )


async def _raise_start_timeout(websocket: WebSocket, timeout: float) -> dict:
    timeout_context = asyncio.timeout(timeout)
    try:
        async with timeout_context:
            return await websocket.receive()
    except TimeoutError as exc:
        if not timeout_context.expired():
            raise
        raise _invalid_message("연결 후 시작 명령을 시간 내에 보내야 합니다.") from exc


async def _receive_with_limits(
    websocket: WebSocket,
    *,
    idle_timeout: float,
    session_deadline: float,
) -> dict:
    """입력 유휴 시간과 연결 전체 수명을 함께 제한한다."""

    loop = asyncio.get_running_loop()
    remaining = session_deadline - loop.time()
    if remaining <= 0:
        raise _WireError(
            "SESSION_LIMIT_REACHED",
            "최대 이용 시간을 초과해 연결을 종료합니다.",
            1008,
        )

    max_session_is_next = remaining <= idle_timeout
    timeout_context = asyncio.timeout(min(idle_timeout, remaining))
    try:
        async with timeout_context:
            return await websocket.receive()
    except TimeoutError as exc:
        if not timeout_context.expired():
            raise
        if max_session_is_next:
            raise _WireError(
                "SESSION_LIMIT_REACHED",
                "최대 이용 시간을 초과해 연결을 종료합니다.",
                1008,
            ) from exc
        raise _WireError(
            "STREAM_IDLE_TIMEOUT",
            "일정 시간 동안 입력이 없어 연결을 종료합니다.",
            1008,
        ) from exc


async def _close_insufficient_frames(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    timeout: float,
) -> None:
    await _send_event_best_effort(
        websocket,
        ErrorEvent(
            code="INSUFFICIENT_FRAMES",
            message="인식에 필요한 프레임이 부족합니다.",
        ),
        send_lock,
        timeout,
        context="인식 WebSocket 프레임 부족 응답 실패",
    )
    await _send_event_best_effort(
        websocket,
        StoppedEvent(),
        send_lock,
        timeout,
        context="인식 WebSocket 종료 event 전송 실패",
    )
    await _close_best_effort(websocket, 1000, send_lock, timeout)


async def _disconnect_session_safely(session) -> None:
    """wire I/O와 독립적으로 buffered session과 capacity를 정리한다."""

    try:
        await session.disconnect()
    except Exception as exc:
        _log_safe_error("인식 WebSocket 세션 정리 실패", exc)


def _observe_cleanup_task(task: asyncio.Task[None]) -> None:
    """handler 취소 뒤 계속되는 cleanup task의 예외를 회수한다."""

    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log_safe_error("인식 WebSocket 비동기 세션 정리 실패", exc)


def _raise_disconnect(message: dict) -> NoReturn:
    raise WebSocketDisconnect(message.get("code", 1000))


@router.websocket("/recognition/stream")
async def recognition_stream(websocket: WebSocket) -> None:
    """한 연결의 JPEG 전체를 stop에서 한 번 인식하고 저장한다."""

    settings = websocket.app.state.settings
    send_lock = asyncio.Lock()
    if getattr(websocket.app.state, "draining", True):
        try:
            async with asyncio.timeout(settings.send_timeout_seconds):
                await websocket.accept()
        except Exception as exc:
            _log_safe_error("draining WebSocket accept 실패", exc)
            return
        await _close_best_effort(
            websocket,
            1012,
            send_lock,
            settings.send_timeout_seconds,
        )
        return

    origin = websocket.headers.get("origin")
    if origin not in settings.allowed_origins:
        await _close_best_effort(
            websocket,
            1008,
            send_lock,
            settings.send_timeout_seconds,
        )
        return

    try:
        async with asyncio.timeout(settings.send_timeout_seconds):
            await websocket.accept()
    except Exception as exc:
        _log_safe_error("인식 WebSocket accept 실패", exc)
        await _close_best_effort(
            websocket,
            1011,
            send_lock,
            settings.send_timeout_seconds,
        )
        return

    session = None
    cleanup_task: asyncio.Task[None] | None = None

    def begin_session_cleanup() -> asyncio.Task[None] | None:
        nonlocal cleanup_task
        if session is not None and cleanup_task is None:
            cleanup_task = asyncio.create_task(_disconnect_session_safely(session))
            cleanup_task.add_done_callback(_observe_cleanup_task)
        return cleanup_task

    try:
        message = await _raise_start_timeout(
            websocket,
            settings.start_timeout_seconds,
        )
        command = _parse_command(_text_message(message))
        if not isinstance(command, StartCommand):
            raise _invalid_message("첫 명령은 start여야 합니다.")

        service = websocket.app.state.recognition_service
        session = await service.open_session(command.mode)
        session_deadline = (
            asyncio.get_running_loop().time() + settings.max_session_seconds
        )
        await _send_event(
            websocket,
            ReadyEvent(),
            send_lock,
            settings.send_timeout_seconds,
        )

        while True:
            message = await _receive_with_limits(
                websocket,
                idle_timeout=settings.stream_idle_timeout_seconds,
                session_deadline=session_deadline,
            )
            if message["type"] == "websocket.disconnect":
                _raise_disconnect(message)

            frame = message.get("bytes")
            if frame is not None:
                if len(frame) > MAX_FRAME_BYTES:
                    raise _WireError(
                        "FRAME_TOO_LARGE",
                        "프레임 크기가 허용 범위를 넘었습니다.",
                        1009,
                    )
                await websocket.app.state.frame_validator.validate(frame)
                await session.push_frame(frame)
                continue

            command = _parse_command(_text_message(message))
            if not isinstance(command, StopCommand):
                raise _invalid_message("시작 후에는 stop 명령만 보낼 수 있습니다.")

            try:
                final = await session.stop()
            except InsufficientFramesError:
                begin_session_cleanup()
                await _close_insufficient_frames(
                    websocket,
                    send_lock,
                    settings.send_timeout_seconds,
                )
                return

            await _send_event(
                websocket,
                ResultEvent(
                    text=final.display_text,
                    final=True,
                    confidence=final.confidence,
                ),
                send_lock,
                settings.send_timeout_seconds,
            )
            await _send_event_best_effort(
                websocket,
                StoppedEvent(),
                send_lock,
                settings.send_timeout_seconds,
                context="인식 WebSocket 종료 event 전송 실패",
            )
            await _close_best_effort(
                websocket,
                1000,
                send_lock,
                settings.send_timeout_seconds,
            )
            return
    except WebSocketDisconnect:
        begin_session_cleanup()
    except _WireError as exc:
        begin_session_cleanup()
        await _send_error_and_close(
            websocket,
            exc,
            send_lock,
            settings.send_timeout_seconds,
        )
    except Exception as exc:
        begin_session_cleanup()
        wire_error = _to_wire_error(exc)
        if wire_error.code == "INTERNAL_ERROR":
            _log_safe_error("인식 WebSocket 처리 실패", exc)
        await _send_error_and_close(
            websocket,
            wire_error,
            send_lock,
            settings.send_timeout_seconds,
        )
    finally:
        task = begin_session_cleanup()
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # cleanup task는 handler 취소와 무관하게 끝까지 실행된다.
                raise
