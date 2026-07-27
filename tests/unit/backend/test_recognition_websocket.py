"""인식 WebSocket의 wire contract와 동시성 정책 테스트."""

import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager

import pytest

from src.backend.core.config import Settings
from src.backend.main import create_app
from src.backend.recognition.domain import (
    MAX_FRAME_BYTES,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_FRAMES,
    Prediction,
    RecognitionMode,
)
from src.backend.recognition.errors import (
    FrameValidationBusyError,
    InferenceBusyError,
    InvalidFrameError,
)


class ASGIWebSocket:
    """실제 포트를 열지 않고 WebSocket ASGI event를 주고받는 테스트 client."""

    def __init__(
        self,
        app,
        *,
        origin: str = "http://localhost:5173",
        block_outbound: Callable[[dict], bool] | None = None,
    ):
        self._app = app
        self._origin = origin
        self._block_outbound = block_outbound
        self._to_app: asyncio.Queue[dict] = asyncio.Queue()
        self._from_app: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.outbound_started = asyncio.Event()
        self.release_outbound = asyncio.Event()

    async def _send_from_app(self, event: dict) -> None:
        if self._block_outbound is not None and self._block_outbound(event):
            self.outbound_started.set()
            await self.release_outbound.wait()
        await self._from_app.put(event)

    async def start(self) -> dict:
        """ASGI WebSocket handshake를 시작하고 첫 server event를 반환한다."""

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 50000),
            "root_path": "",
            "path": "/api/v1/recognition/stream",
            "raw_path": b"/api/v1/recognition/stream",
            "query_string": b"",
            "headers": [(b"origin", self._origin.encode())],
            "subprotocols": [],
            "state": {},
        }
        self._task = asyncio.create_task(
            self._app(scope, self._to_app.get, self._send_from_app)
        )
        await self._to_app.put({"type": "websocket.connect"})
        return await self.receive_event()

    async def connect(self) -> None:
        event = await self.start()
        assert event["type"] == "websocket.accept"

    async def send_text(self, payload: str) -> None:
        await self._to_app.put({"type": "websocket.receive", "text": payload})

    async def send_json(self, payload: dict) -> None:
        await self.send_text(json.dumps(payload))

    async def send_bytes(self, payload: bytes) -> None:
        await self._to_app.put({"type": "websocket.receive", "bytes": payload})

    async def receive_event(self, *, timeout: float = 1.0) -> dict:
        return await asyncio.wait_for(self._from_app.get(), timeout=timeout)

    async def receive_json(self, *, timeout: float = 1.0) -> dict:
        event = await self.receive_event(timeout=timeout)
        assert event["type"] == "websocket.send"
        if "text" in event:
            return json.loads(event["text"])
        return json.loads(event["bytes"])

    async def receive_close(self, expected_code: int) -> None:
        event = await self.receive_event()
        assert event["type"] == "websocket.close"
        assert event["code"] == expected_code
        await self.wait_closed()

    async def disconnect(self, *, wait: bool = True) -> None:
        if self._task is None or self._task.done():
            return
        await self._to_app.put({"type": "websocket.disconnect", "code": 1000})
        if wait:
            await self.wait_closed()

    async def wait_closed(self, *, timeout: float = 1.0) -> None:
        if self._task is not None:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)

    async def force_disconnect(self) -> None:
        """실패한 assertion이 남긴 ASGI task를 다음 테스트로 유출하지 않는다."""

        if self._task is None or self._task.done():
            return
        await self._to_app.put({"type": "websocket.disconnect", "code": 1001})
        try:
            await self.wait_closed(timeout=0.2)
        except TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class FakeDatabase:
    def __init__(self):
        self.closed = False

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class FakeRepository:
    def __init__(self):
        self.created: list[tuple[int, RecognitionMode]] = []
        self.completed: list[tuple[int, object]] = []
        self.ended: list[int] = []
        self._next_session_id = 1

    async def create_session(self, mode: RecognitionMode) -> int:
        session_id = self._next_session_id
        self._next_session_id += 1
        self.created.append((session_id, mode))
        return session_id

    async def complete_session(self, session_id: int, output) -> int:
        self.completed.append((session_id, output))
        return len(self.completed)

    async def end_session(self, session_id: int) -> None:
        self.ended.append(session_id)


class FakeGateway:
    def __init__(
        self,
        *,
        ready: bool = True,
        prediction: Prediction | None = None,
        error: Exception | None = None,
    ):
        self.ready = ready
        self.prediction = prediction or Prediction(
            text="물 주세요",
            confidence=0.91,
            phrase_code="REQUEST_WATER",
        )
        self.error = error
        self.calls: list[tuple[tuple[bytes, ...], RecognitionMode]] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def predict(
        self, frames: tuple[bytes, ...], mode: RecognitionMode
    ) -> Prediction:
        self.calls.append((frames, mode))
        if self.error is not None:
            raise self.error
        return self.prediction

    async def close(self) -> None:
        self.closed = True


class BlockingGateway(FakeGateway):
    """최종 추론을 block해 terminal task와 capacity를 관찰하게 한다."""

    def __init__(self):
        super().__init__()
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def predict(
        self, frames: tuple[bytes, ...], mode: RecognitionMode
    ) -> Prediction:
        self.calls.append((frames, mode))
        call_number = len(self.calls)
        if call_number == 1:
            self.first_started.set()
            native_work = asyncio.create_task(self.release_first.wait())
            try:
                await asyncio.shield(native_work)
            except asyncio.CancelledError:
                # native/GPU 작업은 caller 취소 후에도 끝날 때까지 자원을 점유한다.
                await native_work
                raise
        return Prediction(text=f"결과 {call_number}", confidence=0.9)


class FakeFrameValidator:
    def __init__(
        self,
        *,
        invalid_payload: bytes | None = None,
        error: Exception | None = None,
    ):
        self.invalid_payload = invalid_payload
        self.error = error
        self.frames: list[bytes] = []
        self.closed = False

    async def validate(self, payload: bytes, *args, **kwargs) -> None:
        self.frames.append(payload)
        if self.error is not None:
            raise self.error
        if payload == self.invalid_payload:
            raise InvalidFrameError("JPEG을 디코딩할 수 없습니다")

    async def close(self) -> None:
        self.closed = True


def build_settings(**overrides) -> Settings:
    values = {
        "app_env": "test",
        "database_url": (
            "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"
        ),
        "inference_backend": "fake",
        "max_active_sessions": 1,
        "max_inference_concurrency": 1,
        "start_timeout_seconds": 0.1,
        "send_timeout_seconds": 0.1,
        "stream_idle_timeout_seconds": 1.0,
        "max_session_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def build_app(
    *,
    settings: Settings | None = None,
    repository: FakeRepository | None = None,
    gateway: FakeGateway | None = None,
    validator: FakeFrameValidator | None = None,
):
    resources = {
        "database": FakeDatabase(),
        "repository": repository or FakeRepository(),
        "gateway": gateway or FakeGateway(),
        "validator": validator or FakeFrameValidator(),
    }
    app = create_app(
        settings=settings or build_settings(),
        database=resources["database"],
        repository=resources["repository"],
        gateway=resources["gateway"],
        frame_validator=resources["validator"],
    )
    return app, resources


@asynccontextmanager
async def connected_client(app):
    client = ASGIWebSocket(app)
    await client.connect()
    try:
        yield client
    finally:
        await client.force_disconnect()


async def assert_error_and_close(
    client: ASGIWebSocket, *, code: str, close_code: int
) -> dict:
    event = await client.receive_json()
    assert event["type"] == "error"
    assert event["code"] == code
    assert set(event) == {"type", "code", "message"}
    assert event["message"]
    await client.receive_close(close_code)
    return event


async def wait_until(predicate, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def outbound_event_is(event_type: str) -> Callable[[dict], bool]:
    def matches(event: dict) -> bool:
        if event["type"] != "websocket.send" or "text" not in event:
            return False
        return json.loads(event["text"])["type"] == event_type

    return matches


async def test_start_frames_single_final_stop_happy_path() -> None:
    repository = FakeRepository()
    gateway = FakeGateway()
    app, _ = build_app(repository=repository, gateway=gateway)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start"})
            assert await client.receive_json() == {"type": "ready"}

            frames = tuple(f"frame-{index}".encode() for index in range(30))
            for frame in frames:
                await client.send_bytes(frame)

            with pytest.raises(TimeoutError):
                await client.receive_json(timeout=0.01)
            assert gateway.calls == []
            assert repository.completed == []

            await client.send_json({"type": "stop"})
            assert await client.receive_json() == {
                "type": "result",
                "text": "물 주세요",
                "final": True,
                "confidence": 0.91,
            }
            assert await client.receive_json() == {"type": "stopped"}
            await client.receive_close(1000)

    assert gateway.calls == [(frames, RecognitionMode.CLOSED)]
    assert repository.created == [(1, RecognitionMode.CLOSED)]
    assert len(repository.completed) == 1
    session_id, output = repository.completed[0]
    assert session_id == 1
    assert output.display_text == "물 주세요"
    assert repository.ended == []


async def test_origin_outside_allowlist_is_rejected_before_accept() -> None:
    app, _ = build_app()

    async with app.router.lifespan_context(app):
        client = ASGIWebSocket(app, origin="https://untrusted.example")
        event = await client.start()

        assert event["type"] == "websocket.close"
        assert event["code"] == 1008
        await client.wait_closed()


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"type":"pause"}',
        '{"type":"start","mode":"CLOSED","device_info":"browser"}',
        '{"type":"start","mode":[]}',
        '{"type":"start","mode":{}}',
    ],
)
async def test_invalid_or_extra_command_fields_are_rejected(payload: str) -> None:
    app, _ = build_app()

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_text(payload)
            await assert_error_and_close(
                client, code="INVALID_MESSAGE", close_code=1008
            )


async def test_binary_frame_before_start_is_rejected() -> None:
    app, _ = build_app()

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_bytes(b"frame")
            await assert_error_and_close(
                client, code="INVALID_MESSAGE", close_code=1008
            )


async def test_second_start_command_ends_the_created_session() -> None:
    repository = FakeRepository()
    app, _ = build_app(repository=repository)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "CLOSED"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_json({"type": "start", "mode": "CLOSED"})
            await assert_error_and_close(
                client, code="INVALID_MESSAGE", close_code=1008
            )

    assert repository.ended == [1]


async def test_unsupported_mode_has_a_distinct_stable_error() -> None:
    app, _ = build_app()

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "CONTINUOUS"})
            await assert_error_and_close(
                client, code="UNSUPPORTED_MODE", close_code=1008
            )


async def test_start_command_timeout_is_an_invalid_message() -> None:
    app, _ = build_app(settings=build_settings(start_timeout_seconds=0.02))

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await assert_error_and_close(
                client, code="INVALID_MESSAGE", close_code=1008
            )


async def test_draining_application_rejects_new_connection_for_restart() -> None:
    app, _ = build_app()

    async with app.router.lifespan_context(app):
        app.state.draining = True
        client = ASGIWebSocket(app)
        try:
            await client.connect()
            await client.receive_close(1012)
        finally:
            await client.force_disconnect()


async def test_idle_timeout_ends_session_and_returns_capacity() -> None:
    repository = FakeRepository()
    app, _ = build_app(
        settings=build_settings(
            stream_idle_timeout_seconds=0.02,
            max_session_seconds=1.0,
        ),
        repository=repository,
    )

    async with app.router.lifespan_context(app):
        first = ASGIWebSocket(app)
        second = ASGIWebSocket(app)
        try:
            await first.connect()
            await first.send_json({"type": "start", "mode": "CLOSED"})
            assert await first.receive_json() == {"type": "ready"}
            event = await assert_error_and_close(
                first, code="STREAM_IDLE_TIMEOUT", close_code=1008
            )
            assert "입력" in event["message"]
            assert repository.ended == [1]

            await second.connect()
            await second.send_json({"type": "start", "mode": "CLOSED"})
            assert await second.receive_json() == {"type": "ready"}
            await second.disconnect()
        finally:
            await first.force_disconnect()
            await second.force_disconnect()

    assert repository.ended == [1, 2]


async def test_max_session_timeout_has_a_stable_error_and_cleans_up() -> None:
    repository = FakeRepository()
    app, _ = build_app(
        settings=build_settings(
            stream_idle_timeout_seconds=1.0,
            max_session_seconds=0.02,
        ),
        repository=repository,
    )

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            event = await assert_error_and_close(
                client, code="SESSION_LIMIT_REACHED", close_code=1008
            )
            assert "최대 이용 시간" in event["message"]

    assert repository.ended == [1]


async def test_oversized_frame_is_rejected_before_decode() -> None:
    validator = FakeFrameValidator()
    app, _ = build_app(validator=validator)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"x" * (MAX_FRAME_BYTES + 1))
            await assert_error_and_close(
                client, code="FRAME_TOO_LARGE", close_code=1009
            )

    assert validator.frames == []


async def test_251st_frame_is_rejected_as_video_too_long() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator()
    app, _ = build_app(repository=repository, validator=validator)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start"})
            assert await client.receive_json() == {"type": "ready"}
            for _ in range(MAX_VIDEO_FRAMES + 1):
                await client.send_bytes(b"frame")

            await assert_error_and_close(
                client,
                code="VIDEO_TOO_LONG",
                close_code=1008,
            )

    assert len(validator.frames) == MAX_VIDEO_FRAMES + 1
    assert repository.completed == []
    assert repository.ended == [1]


async def test_first_byte_over_64_mib_is_rejected_as_video_too_large() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator()
    app, _ = build_app(repository=repository, validator=validator)
    shared_frame = b"x" * (MAX_VIDEO_BYTES // 249)
    final_frame = b"x" * (MAX_VIDEO_BYTES - len(shared_frame) * 248)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start"})
            assert await client.receive_json() == {"type": "ready"}
            for _ in range(248):
                await client.send_bytes(shared_frame)
            await client.send_bytes(final_frame)
            await client.send_bytes(b"x")

            await assert_error_and_close(
                client,
                code="VIDEO_TOO_LARGE",
                close_code=1009,
            )

    assert len(validator.frames) == 250
    assert repository.completed == []
    assert repository.ended == [1]


async def test_invalid_jpeg_is_rejected_and_session_is_ended() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator(invalid_payload=b"broken")
    app, _ = build_app(repository=repository, validator=validator)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"broken")
            await assert_error_and_close(client, code="INVALID_FRAME", close_code=1003)

    assert repository.ended == [1]


async def test_stop_below_thirty_frames_reports_insufficient_frames() -> None:
    repository = FakeRepository()
    app, _ = build_app(repository=repository)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "CLOSED"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"frame-1")
            await client.send_bytes(b"frame-2")
            await client.send_json({"type": "stop"})

            event = await client.receive_json()
            assert event["type"] == "error"
            assert event["code"] == "INSUFFICIENT_FRAMES"
            assert await client.receive_json() == {"type": "stopped"}
            await client.receive_close(1000)

    assert repository.completed == []
    assert repository.ended == [1]


async def test_model_unavailable_rejects_start_without_creating_session() -> None:
    repository = FakeRepository()
    gateway = FakeGateway(ready=False)
    app, _ = build_app(repository=repository, gateway=gateway)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "CLOSED"})
            await assert_error_and_close(
                client, code="MODEL_NOT_READY", close_code=1013
            )

    assert repository.created == []


async def test_inference_capacity_error_closes_session_as_server_busy() -> None:
    repository = FakeRepository()
    gateway = FakeGateway(error=InferenceBusyError("worker 사용 중"))
    app, _ = build_app(repository=repository, gateway=gateway)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "CLOSED"})
            assert await client.receive_json() == {"type": "ready"}
            for _ in range(30):
                await client.send_bytes(b"frame")
            await client.send_json({"type": "stop"})
            await assert_error_and_close(client, code="SERVER_BUSY", close_code=1013)

    assert repository.ended == [1]


async def test_frame_validation_capacity_error_closes_as_server_busy() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator(
        error=FrameValidationBusyError("JPEG 검증 worker 사용 중")
    )
    app, _ = build_app(repository=repository, validator=validator)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"frame")
            await assert_error_and_close(client, code="SERVER_BUSY", close_code=1013)

    assert repository.ended == [1]


async def test_internal_error_log_does_not_contain_recognition_text(caplog) -> None:
    sensitive_text = "인식 문장 물 주세요"
    gateway = FakeGateway(error=RuntimeError(sensitive_text))
    app, _ = build_app(gateway=gateway)

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            for _ in range(30):
                await client.send_bytes(b"frame")
            await client.send_json({"type": "stop"})
            await assert_error_and_close(
                client,
                code="INTERNAL_ERROR",
                close_code=1011,
            )

    assert "RuntimeError" in caplog.text
    assert sensitive_text not in caplog.text


async def test_disconnect_cleanup_error_is_sanitized_and_does_not_escape(
    caplog,
) -> None:
    sensitive_parameter = "DB parameter 물 주세요"

    class FailingEndRepository(FakeRepository):
        async def end_session(self, session_id: int) -> None:
            raise RuntimeError(sensitive_parameter)

    validator = FakeFrameValidator(invalid_payload=b"broken")
    app, _ = build_app(
        repository=FailingEndRepository(),
        validator=validator,
    )

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"broken")
            await assert_error_and_close(
                client,
                code="INVALID_FRAME",
                close_code=1003,
            )

    assert "RuntimeError" in caplog.text
    assert sensitive_parameter not in caplog.text


async def test_blocked_error_send_does_not_delay_session_cleanup() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator(invalid_payload=b"broken")
    app, _ = build_app(
        settings=build_settings(send_timeout_seconds=0.2),
        repository=repository,
        validator=validator,
    )

    async with app.router.lifespan_context(app):
        client = ASGIWebSocket(app, block_outbound=outbound_event_is("error"))
        try:
            await client.connect()
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"broken")
            await asyncio.wait_for(client.outbound_started.wait(), timeout=1)

            # 오류 응답 I/O와 세션 capacity 반환은 서로 독립적이어야 한다.
            await wait_until(lambda: repository.ended == [1], timeout=0.05)
            await client.receive_close(1003)
        finally:
            client.release_outbound.set()
            await client.force_disconnect()


async def test_blocked_stopped_event_is_bounded_and_close_is_still_attempted() -> None:
    repository = FakeRepository()
    app, _ = build_app(
        settings=build_settings(send_timeout_seconds=0.02),
        repository=repository,
    )

    async with app.router.lifespan_context(app):
        client = ASGIWebSocket(app, block_outbound=outbound_event_is("stopped"))
        try:
            await client.connect()
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            for _ in range(30):
                await client.send_bytes(b"frame")

            await client.send_json({"type": "stop"})
            assert (await client.receive_json())["final"] is True
            await asyncio.wait_for(client.outbound_started.wait(), timeout=1)
            assert len(repository.completed) == 1
            await client.receive_close(1000)
        finally:
            client.release_outbound.set()
            await client.force_disconnect()


async def test_blocked_close_is_bounded_and_handler_finishes() -> None:
    repository = FakeRepository()
    validator = FakeFrameValidator(invalid_payload=b"broken")
    app, _ = build_app(
        settings=build_settings(send_timeout_seconds=0.02),
        repository=repository,
        validator=validator,
    )

    async with app.router.lifespan_context(app):
        client = ASGIWebSocket(
            app,
            block_outbound=lambda event: event["type"] == "websocket.close",
        )
        try:
            await client.connect()
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            await client.send_bytes(b"broken")
            assert (await client.receive_json())["code"] == "INVALID_FRAME"
            await asyncio.wait_for(client.outbound_started.wait(), timeout=1)

            await client.wait_closed(timeout=0.2)
            assert repository.ended == [1]
        finally:
            client.release_outbound.set()
            await client.force_disconnect()


async def test_frames_are_buffered_until_stop_then_normalized_once() -> None:
    repository = FakeRepository()
    gateway = BlockingGateway()
    validator = FakeFrameValidator()
    app, _ = build_app(
        repository=repository,
        gateway=gateway,
        validator=validator,
    )

    async with app.router.lifespan_context(app):
        async with connected_client(app) as client:
            await client.send_json({"type": "start", "mode": "OPEN"})
            assert await client.receive_json() == {"type": "ready"}
            frames = tuple(f"frame-{index}".encode() for index in range(40))
            for frame in frames:
                await client.send_bytes(frame)
            await wait_until(lambda: len(validator.frames) == 40)

            assert gateway.first_started.is_set() is False
            with pytest.raises(TimeoutError):
                await client.receive_json(timeout=0.01)

            await client.send_json({"type": "stop"})
            await asyncio.wait_for(gateway.first_started.wait(), timeout=1)
            gateway.release_first.set()
            final = await client.receive_json()
            assert (final["text"], final["final"]) == ("결과 1", True)
            assert await client.receive_json() == {"type": "stopped"}
            await client.receive_close(1000)

    expected_indices = tuple(index * 39 // 29 for index in range(30))
    assert gateway.calls == [
        (tuple(frames[index] for index in expected_indices), RecognitionMode.OPEN)
    ]


async def test_disconnect_after_stop_keeps_terminal_work_and_session_slot() -> None:
    repository = FakeRepository()
    gateway = BlockingGateway()
    app, _ = build_app(repository=repository, gateway=gateway)

    async with app.router.lifespan_context(app):
        first = ASGIWebSocket(app)
        second = ASGIWebSocket(app)
        third = ASGIWebSocket(app)
        try:
            await first.connect()
            await first.send_json({"type": "start", "mode": "OPEN"})
            assert await first.receive_json() == {"type": "ready"}
            for _ in range(30):
                await first.send_bytes(b"frame")
            await first.send_json({"type": "stop"})
            await asyncio.wait_for(gateway.first_started.wait(), timeout=1)
            await first.disconnect(wait=False)

            await second.connect()
            await second.send_json({"type": "start", "mode": "OPEN"})
            await assert_error_and_close(second, code="SERVER_BUSY", close_code=1013)

            gateway.release_first.set()
            await first.wait_closed()
            await wait_until(lambda: len(repository.completed) == 1)

            await third.connect()
            await third.send_json({"type": "start", "mode": "OPEN"})
            assert await third.receive_json() == {"type": "ready"}
            await third.disconnect()
        finally:
            gateway.release_first.set()
            await first.force_disconnect()
            await second.force_disconnect()
            await third.force_disconnect()

    assert repository.created == [
        (1, RecognitionMode.OPEN),
        (2, RecognitionMode.OPEN),
    ]
    assert len(repository.completed) == 1
    assert repository.ended == [2]
