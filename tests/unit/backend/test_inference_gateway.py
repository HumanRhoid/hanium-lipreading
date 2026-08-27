"""동기 모델을 격리하는 제한된 추론 gateway 계약 테스트."""

import asyncio
import threading
import time

import pytest

from src.backend.recognition.adapters.inference import (
    BoundedLocalRecognitionGateway,
    FakeSyncPredictor,
    UnavailableRecognitionGateway,
)
from src.backend.recognition.domain import (
    MODEL_INPUT_FRAME_COUNT,
    STREAM_FRAME_HEIGHT,
    STREAM_FRAME_WIDTH,
    ModelManifest,
    Prediction,
    RecognitionMode,
)
from src.backend.recognition.errors import InferenceBusyError, ModelNotReadyError


class BlockingPredictor:
    def __init__(self):
        self.ready = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.start_count = 0
        self.close_count = 0
        self.manifest = ModelManifest(
            bundle_version="blocking-v1",
            supported_modes={RecognitionMode.CLOSED, RecognitionMode.OPEN},
            frame_width=STREAM_FRAME_WIDTH,
            frame_height=STREAM_FRAME_HEIGHT,
            fps=25,
            input_frame_count=MODEL_INPUT_FRAME_COUNT,
        )

    def start(self):
        self.start_count += 1
        self.ready = True

    def predict(self, frames, mode):
        self.started.set()
        self.release.wait(timeout=5)
        return Prediction(text="물 주세요", confidence=0.9, phrase_code="REQUEST_WATER")

    def close(self):
        self.close_count += 1
        self.ready = False


async def wait_for_thread_event(event):
    ready = await asyncio.to_thread(event.wait, 1)
    assert ready is True


async def test_gateway_runs_predictor_outside_event_loop():
    class SlowPredictor(BlockingPredictor):
        def predict(self, frames, mode):
            time.sleep(0.05)
            return Prediction(text="안녕하세요")

    gateway = BoundedLocalRecognitionGateway(SlowPredictor(), max_concurrency=1)
    await gateway.start()

    inference = asyncio.create_task(gateway.predict((b"jpeg",), RecognitionMode.OPEN))
    started_at = time.perf_counter()
    await asyncio.sleep(0.01)
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.04
    assert (await inference).text == "안녕하세요"
    await gateway.close()


async def test_gateway_rejects_when_capacity_is_in_use():
    predictor = BlockingPredictor()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)
    await gateway.start()
    first = asyncio.create_task(gateway.predict((b"first",), RecognitionMode.CLOSED))
    await wait_for_thread_event(predictor.started)

    with pytest.raises(InferenceBusyError):
        await gateway.predict((b"second",), RecognitionMode.CLOSED)

    predictor.release.set()
    assert (await first).text == "물 주세요"
    await gateway.close()


async def test_cancelled_caller_does_not_release_running_worker_capacity():
    predictor = BlockingPredictor()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)
    await gateway.start()
    first = asyncio.create_task(gateway.predict((b"first",), RecognitionMode.CLOSED))
    await wait_for_thread_event(predictor.started)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(InferenceBusyError):
        await gateway.predict((b"second",), RecognitionMode.CLOSED)

    predictor.release.set()
    for _ in range(100):
        try:
            result = await gateway.predict((b"third",), RecognitionMode.CLOSED)
        except InferenceBusyError:
            await asyncio.sleep(0.01)
            continue
        break
    else:
        pytest.fail("동기 worker 종료 후에도 capacity가 반환되지 않았습니다")

    assert result.text == "물 주세요"
    await gateway.close()


async def test_gateway_starts_and_closes_predictor_once():
    predictor = BlockingPredictor()
    predictor.release.set()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)

    await gateway.start()
    await gateway.close()
    await gateway.close()

    assert predictor.start_count == 1
    assert predictor.close_count == 1


async def test_concurrent_start_shares_work_even_if_one_caller_is_cancelled():
    class SlowStartPredictor(BlockingPredictor):
        def __init__(self):
            super().__init__()
            self.start_entered = threading.Event()
            self.start_release = threading.Event()

        def start(self):
            self.start_count += 1
            self.start_entered.set()
            self.start_release.wait(timeout=5)
            self.ready = True

    predictor = SlowStartPredictor()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)
    first = asyncio.create_task(gateway.start())
    second = asyncio.create_task(gateway.start())
    await wait_for_thread_event(predictor.start_entered)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    predictor.start_release.set()
    await second

    assert predictor.start_count == 1
    assert gateway.ready is True
    await gateway.close()


async def test_cancelled_close_caller_does_not_interrupt_shared_cleanup():
    predictor = BlockingPredictor()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)
    await gateway.start()
    inference = asyncio.create_task(
        gateway.predict((b"frame",), RecognitionMode.CLOSED)
    )
    await wait_for_thread_event(predictor.started)

    first_close = asyncio.create_task(gateway.close())
    await asyncio.sleep(0)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    predictor.release.set()
    assert (await inference).text == "물 주세요"
    await gateway.close()

    assert predictor.close_count == 1
    assert gateway.ready is False


async def test_gateway_exposes_predictor_manifest():
    predictor = BlockingPredictor()
    gateway = BoundedLocalRecognitionGateway(predictor, max_concurrency=1)

    assert gateway.manifest is predictor.manifest

    await gateway.close()


async def test_unavailable_gateway_never_returns_fake_result():
    gateway = UnavailableRecognitionGateway()
    await gateway.start()

    assert gateway.ready is False
    with pytest.raises(ModelNotReadyError):
        await gateway.predict((b"jpeg",), RecognitionMode.CLOSED)


@pytest.mark.parametrize(
    "mode,expected_text,expected_code",
    [
        (RecognitionMode.CLOSED, "물 주세요", "REQUEST_WATER"),
        (RecognitionMode.OPEN, "안녕하세요", None),
    ],
)
def test_fake_predictor_supports_both_api_modes(mode, expected_text, expected_code):
    predictor = FakeSyncPredictor()
    predictor.start()

    prediction = predictor.predict((b"jpeg",) * MODEL_INPUT_FRAME_COUNT, mode)

    assert prediction.text == expected_text
    assert prediction.phrase_code == expected_code


def test_fake_predictor_manifest_matches_public_input_contract():
    manifest = FakeSyncPredictor().manifest

    assert manifest.frame_width == STREAM_FRAME_WIDTH
    assert manifest.frame_height == STREAM_FRAME_HEIGHT
    assert manifest.fps == 25
    assert manifest.input_frame_count == MODEL_INPUT_FRAME_COUNT
    assert manifest.supported_modes == frozenset(RecognitionMode)


def test_fake_predictor_rejects_input_that_was_not_normalized_to_sixty_frames():
    predictor = FakeSyncPredictor()
    predictor.start()

    with pytest.raises(ValueError, match="정확히 60프레임"):
        predictor.predict((b"jpeg",) * 59, RecognitionMode.CLOSED)
