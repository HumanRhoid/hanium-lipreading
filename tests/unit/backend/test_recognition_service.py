"""추론, 교정과 영속성을 조정하는 애플리케이션 서비스 테스트."""

import asyncio
from dataclasses import replace

import pytest

from src.backend.recognition.domain import (
    INPUT_FRAME_COUNT,
    INPUT_FRAME_HEIGHT,
    INPUT_FRAME_WIDTH,
    ModelManifest,
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)
from src.backend.recognition.errors import UnsupportedRecognitionModeError
from src.backend.recognition.service import RecognitionService


class FakeRepository:
    def __init__(self):
        self.created_modes = []
        self.completed = []
        self.ended = []

    async def create_session(self, mode):
        self.created_modes.append(mode)
        return 42

    async def complete_session(self, session_id, output):
        self.completed.append((session_id, output))
        return 7

    async def end_session(self, session_id):
        self.ended.append(session_id)


class FakeGateway:
    ready = True

    def __init__(self, manifest=None):
        self.manifest = manifest

    async def predict(self, frames, mode):
        assert frames == (b"frame",)
        assert mode == RecognitionMode.CLOSED
        return Prediction(
            text="물 주세오",
            confidence=0.87,
            phrase_code="REQUEST_WATER",
        )


class FakeCorrector:
    async def correct(self, text):
        assert text == "물 주세오"
        return "물 주세요"


def valid_manifest(**overrides):
    manifest = ModelManifest(
        bundle_version="fake-v1",
        supported_modes={RecognitionMode.CLOSED, RecognitionMode.OPEN},
        frame_width=INPUT_FRAME_WIDTH,
        frame_height=INPUT_FRAME_HEIGHT,
        fps=25,
        input_frame_count=INPUT_FRAME_COUNT,
        label_map_version="demo-v1",
    )
    return replace(manifest, **overrides)


async def test_service_keeps_raw_and_corrected_text_separate():
    service = RecognitionService(
        repository=FakeRepository(),
        gateway=FakeGateway(),
        corrector=FakeCorrector(),
    )

    output = await service.recognize((b"frame",), RecognitionMode.CLOSED)

    assert output == RecognitionOutput(
        raw_text="물 주세오",
        corrected_text="물 주세요",
        confidence=0.87,
        phrase_code="REQUEST_WATER",
    )
    assert output.display_text == "물 주세요"


async def test_service_delegates_session_lifecycle_to_repository():
    repository = FakeRepository()
    service = RecognitionService(
        repository=repository,
        gateway=FakeGateway(),
        corrector=FakeCorrector(),
    )
    output = RecognitionOutput(raw_text="안녕하세요")

    session_id = await service.start_session(RecognitionMode.OPEN)
    utterance_id = await service.complete_session(session_id, output)
    await service.end_session(session_id)

    assert session_id == 42
    assert utterance_id == 7
    assert repository.created_modes == [RecognitionMode.OPEN]
    assert repository.completed == [(42, output)]
    assert repository.ended == [42]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_width", INPUT_FRAME_WIDTH + 1),
        ("frame_height", INPUT_FRAME_HEIGHT + 1),
        ("fps", 24),
        ("input_frame_count", INPUT_FRAME_COUNT - 1),
        ("input_codec", "video/mp4"),
    ],
)
def test_service_rejects_model_manifest_that_differs_from_api_contract(field, value):
    with pytest.raises(ValueError, match="모델 manifest"):
        RecognitionService(
            repository=FakeRepository(),
            gateway=FakeGateway(valid_manifest(**{field: value})),
            corrector=FakeCorrector(),
        )


async def test_service_rejects_mode_not_supported_by_model_before_creating_session():
    repository = FakeRepository()
    service = RecognitionService(
        repository=repository,
        gateway=FakeGateway(
            valid_manifest(supported_modes=frozenset({RecognitionMode.CLOSED}))
        ),
        corrector=FakeCorrector(),
    )

    with pytest.raises(UnsupportedRecognitionModeError):
        await service.open_session(RecognitionMode.OPEN)

    assert repository.created_modes == []


async def test_repeated_cancellation_does_not_leak_capacity_after_create_failure():
    class CancelThenSucceedRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.first_started = asyncio.Event()
            self.first_cancelled = asyncio.Event()
            self.calls = 0

        async def create_session(self, mode):
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.first_cancelled.set()
            self.created_modes.append(mode)
            return 42

    repository = CancelThenSucceedRepository()
    service = RecognitionService(
        repository=repository,
        gateway=FakeGateway(),
        corrector=FakeCorrector(),
        max_active_sessions=1,
    )
    opening = asyncio.create_task(service.open_session(RecognitionMode.CLOSED))
    await repository.first_started.wait()

    # 반환 lock을 잡아 첫 취소가 cleanup에서 대기하게 만든 뒤 다시 취소한다.
    await service._session_capacity._lock.acquire()
    try:
        opening.cancel()
        await repository.first_cancelled.wait()
        await asyncio.sleep(0)
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
    finally:
        service._session_capacity._lock.release()

    for _ in range(100):
        if not service._session_open_cleanup_tasks:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("반복 취소 후 capacity 반환 작업이 끝나지 않았습니다")

    stream = await service.open_session(RecognitionMode.CLOSED)
    await stream.disconnect()
