"""BufferedRecognitionSession의 단일 추론과 수명주기 테스트."""

import asyncio

import pytest

from src.backend.recognition import errors
from src.backend.recognition.domain import (
    MAX_VIDEO_BYTES,
    MAX_VIDEO_FRAMES,
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.ports import (
    RecognitionGateway,
    RecognitionRepository,
    TextCorrector,
)
from src.backend.recognition.service import RecognitionService


class RecordingRepository:
    """DB 호출 순서와 최종 결과만 기록하는 repository fake."""

    def __init__(self):
        self.created_modes: list[RecognitionMode] = []
        self.completed: list[tuple[int, RecognitionOutput]] = []
        self.ended: list[int] = []
        self.events: list[tuple[str, object]] = []

    async def create_session(self, mode: RecognitionMode) -> int:
        session_id = len(self.created_modes) + 1
        self.created_modes.append(mode)
        self.events.append(("created", session_id))
        return session_id

    async def complete_session(
        self,
        session_id: int,
        output: RecognitionOutput,
    ) -> int:
        self.completed.append((session_id, output))
        self.events.append(("completed", output))
        return len(self.completed)

    async def end_session(self, session_id: int) -> None:
        self.ended.append(session_id)
        self.events.append(("ended", session_id))


class RecordingCorrector:
    """호출 횟수를 기록하고 모델 원문을 바꾸지 않는 교정기 fake."""

    def __init__(self):
        self.calls: list[str] = []

    async def correct(self, text: str) -> None:
        self.calls.append(text)
        return None


class ImmediateGateway:
    """호출된 최종 입력을 기록하고 즉시 결과를 반환한다."""

    ready = True
    manifest = None

    def __init__(self):
        self.calls: list[tuple[tuple[bytes, ...], RecognitionMode]] = []

    async def predict(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> Prediction:
        self.calls.append((frames, mode))
        return Prediction(text=f"결과-{len(self.calls)}", confidence=0.9)


class BlockingGateway(ImmediateGateway):
    """단일 추론을 명시적으로 해제할 때까지 대기시키는 gateway fake."""

    def __init__(self):
        super().__init__()
        self.called = asyncio.Event()
        self.release = asyncio.Event()

    async def predict(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> Prediction:
        self.calls.append((frames, mode))
        self.called.set()
        await self.release.wait()
        return Prediction(text="최종 결과", confidence=0.8)


class FailingGateway(ImmediateGateway):
    async def predict(
        self,
        frames: tuple[bytes, ...],
        mode: RecognitionMode,
    ) -> Prediction:
        self.calls.append((frames, mode))
        raise RuntimeError("predict failed")


def make_service(
    repository: RecognitionRepository,
    gateway: RecognitionGateway,
    *,
    corrector: TextCorrector | None = None,
    max_active_sessions: int = 1,
) -> tuple[RecognitionService, TextCorrector]:
    app_corrector = corrector or RecordingCorrector()
    return (
        RecognitionService(
            repository=repository,
            gateway=gateway,
            corrector=app_corrector,
            max_active_sessions=max_active_sessions,
        ),
        app_corrector,
    )


async def push_frames(stream, count: int) -> tuple[bytes, ...]:
    frames = tuple(f"frame-{index}".encode() for index in range(count))
    for frame in frames:
        await stream.push_frame(frame)
    return frames


async def test_active_session_limit_rejects_immediately_and_reopens_after_close():
    repository = RecordingRepository()
    service, _ = make_service(repository, ImmediateGateway())

    first = await service.open_session(RecognitionMode.CLOSED)

    with pytest.raises(errors.SessionBusyError):
        await service.open_session(RecognitionMode.OPEN)

    await first.disconnect()
    second = await service.open_session(RecognitionMode.OPEN)
    await second.disconnect()

    assert repository.created_modes == [RecognitionMode.CLOSED, RecognitionMode.OPEN]


async def test_frames_are_only_buffered_until_stop_then_processed_once():
    repository = RecordingRepository()
    gateway = ImmediateGateway()
    corrector = RecordingCorrector()
    service, _ = make_service(repository, gateway, corrector=corrector)
    session = await service.open_session(RecognitionMode.CLOSED)
    frames = await push_frames(session, 80)

    assert session.buffered_frame_count == 80
    assert session.buffered_bytes == sum(map(len, frames))
    assert gateway.calls == []
    assert corrector.calls == []
    assert repository.completed == []

    output = await session.stop()

    expected_indices = tuple(index * 79 // 59 for index in range(60))
    assert gateway.calls == [
        (tuple(frames[index] for index in expected_indices), RecognitionMode.CLOSED)
    ]
    assert corrector.calls == ["결과-1"]
    assert repository.completed == [(1, output)]
    assert repository.ended == []
    assert session.buffered_frame_count == 0
    assert session.buffered_bytes == 0


async def test_repeated_and_concurrent_stop_share_one_terminal_task():
    repository = RecordingRepository()
    gateway = BlockingGateway()
    corrector = RecordingCorrector()
    service, _ = make_service(repository, gateway, corrector=corrector)
    session = await service.open_session(RecognitionMode.OPEN)
    await push_frames(session, 60)

    first = asyncio.create_task(session.stop())
    second = asyncio.create_task(session.stop())
    await gateway.called.wait()

    assert len(gateway.calls) == 1
    assert repository.completed == []
    gateway.release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result is second_result
    assert corrector.calls == ["최종 결과"]
    assert repository.completed == [(1, first_result)]


async def test_insufficient_stop_skips_inference_ends_session_and_releases_buffer():
    repository = RecordingRepository()
    gateway = ImmediateGateway()
    service, _ = make_service(repository, gateway)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 29)

    with pytest.raises(errors.InsufficientFramesError):
        await session.stop()

    assert gateway.calls == []
    assert repository.completed == []
    assert repository.ended == [1]
    assert session.buffered_frame_count == 0
    assert session.buffered_bytes == 0

    reopened = await service.open_session(RecognitionMode.CLOSED)
    await reopened.disconnect()


async def test_session_rejects_the_251st_frame_without_growing_the_buffer():
    repository = RecordingRepository()
    service, _ = make_service(repository, ImmediateGateway())
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, MAX_VIDEO_FRAMES)

    with pytest.raises(errors.VideoTooLongError):
        await session.push_frame(b"one-too-many")

    assert session.buffered_frame_count == MAX_VIDEO_FRAMES
    await session.disconnect()


async def test_session_rejects_the_first_byte_over_64_mib():
    repository = RecordingRepository()
    service, _ = make_service(repository, ImmediateGateway())
    session = await service.open_session(RecognitionMode.CLOSED)
    shared_frame = b"x" * (MAX_VIDEO_BYTES // 249)
    final_frame = b"x" * (MAX_VIDEO_BYTES - len(shared_frame) * 248)

    for _ in range(248):
        await session.push_frame(shared_frame)
    await session.push_frame(final_frame)
    assert session.buffered_bytes == MAX_VIDEO_BYTES

    with pytest.raises(errors.VideoTooLargeError):
        await session.push_frame(b"x")

    assert session.buffered_frame_count == 249
    assert session.buffered_bytes == MAX_VIDEO_BYTES
    await session.disconnect()


async def test_disconnect_before_stop_discards_without_inference_or_utterance():
    repository = RecordingRepository()
    gateway = ImmediateGateway()
    corrector = RecordingCorrector()
    service, _ = make_service(repository, gateway, corrector=corrector)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 60)

    await session.disconnect()

    assert gateway.calls == []
    assert corrector.calls == []
    assert repository.completed == []
    assert repository.ended == [1]
    assert session.buffered_frame_count == 0
    assert session.buffered_bytes == 0
    with pytest.raises(errors.SessionClosedError):
        await session.stop()


async def test_cancelled_stop_keeps_capacity_until_terminal_work_finishes():
    repository = RecordingRepository()
    gateway = BlockingGateway()
    service, _ = make_service(repository, gateway)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 60)

    stopping = asyncio.create_task(session.stop())
    await gateway.called.wait()
    stopping.cancel()
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    with pytest.raises(errors.SessionBusyError):
        await service.open_session(RecognitionMode.OPEN)

    gateway.release.set()
    await session.disconnect()

    assert len(gateway.calls) == 1
    assert len(repository.completed) == 1
    reopened = await service.open_session(RecognitionMode.OPEN)
    await reopened.disconnect()


async def test_disconnect_after_stop_waits_for_the_same_terminal_task():
    repository = RecordingRepository()
    gateway = BlockingGateway()
    service, _ = make_service(repository, gateway)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 60)

    stopping = asyncio.create_task(session.stop())
    await gateway.called.wait()
    disconnecting = asyncio.create_task(session.disconnect())
    await asyncio.sleep(0)

    assert disconnecting.done() is False
    assert repository.completed == []
    gateway.release.set()
    output = await stopping
    await disconnecting

    assert repository.completed == [(1, output)]
    assert repository.ended == []


async def test_inference_failure_ends_session_clears_buffer_and_releases_capacity():
    repository = RecordingRepository()
    gateway = FailingGateway()
    service, _ = make_service(repository, gateway)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 60)

    with pytest.raises(RuntimeError, match="predict failed"):
        await session.stop()

    assert len(gateway.calls) == 1
    assert repository.completed == []
    assert repository.ended == [1]
    assert session.buffered_frame_count == 0
    assert session.buffered_bytes == 0

    reopened = await service.open_session(RecognitionMode.OPEN)
    await reopened.disconnect()


async def test_frames_are_rejected_after_terminal_processing_starts():
    repository = RecordingRepository()
    gateway = BlockingGateway()
    service, _ = make_service(repository, gateway)
    session = await service.open_session(RecognitionMode.CLOSED)
    await push_frames(session, 60)
    stopping = asyncio.create_task(session.stop())
    await gateway.called.wait()

    with pytest.raises(errors.SessionClosedError):
        await session.push_frame(b"late")

    gateway.release.set()
    await stopping
