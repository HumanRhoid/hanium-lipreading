"""JPEG 프레임 검증 어댑터의 계약 테스트."""

import asyncio
import threading

import cv2
import numpy as np
import pytest

from src.backend.recognition.adapters.media import JpegFrameValidator
from src.backend.recognition.errors import FrameValidationBusyError, InvalidFrameError


def encode_jpeg(*, width: int = 640, height: int = 360) -> bytes:
    """테스트용 단색 JPEG를 생성한다."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded is True
    return buffer.tobytes()


async def test_validator_accepts_640_by_360_jpeg():
    validator = JpegFrameValidator()

    try:
        await validator.validate(encode_jpeg())
    finally:
        await validator.close()


async def test_validator_rejects_bytes_that_are_not_jpeg():
    validator = JpegFrameValidator()

    try:
        with pytest.raises(InvalidFrameError):
            await validator.validate(b"not-a-jpeg")
    finally:
        await validator.close()


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (320, 360),
        (640, 480),
    ],
)
async def test_validator_rejects_jpeg_with_unexpected_resolution(width, height):
    validator = JpegFrameValidator()

    try:
        with pytest.raises(InvalidFrameError):
            await validator.validate(encode_jpeg(width=width, height=height))
    finally:
        await validator.close()


async def test_validator_decodes_jpeg_outside_event_loop_thread(monkeypatch):
    validator = JpegFrameValidator()
    event_loop_thread_id = threading.get_ident()
    decoder_thread_ids = []
    original_imdecode = cv2.imdecode

    def tracked_imdecode(*args, **kwargs):
        decoder_thread_ids.append(threading.get_ident())
        return original_imdecode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecode", tracked_imdecode)

    try:
        await validator.validate(encode_jpeg())
    finally:
        await validator.close()

    assert decoder_thread_ids
    assert event_loop_thread_id not in decoder_thread_ids


async def test_validator_rejects_large_header_dimensions_before_decode(monkeypatch):
    payload = bytearray(encode_jpeg())
    marker_position = next(
        index
        for index in range(len(payload) - 1)
        if payload[index] == 0xFF and payload[index + 1] in {0xC0, 0xC2}
    )
    segment_start = marker_position + 2
    payload[segment_start + 3 : segment_start + 5] = (5000).to_bytes(2, "big")
    payload[segment_start + 5 : segment_start + 7] = (5000).to_bytes(2, "big")
    decode_called = False

    def fail_if_decoded(*args, **kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("고해상도 JPEG을 decode하면 안 됩니다")

    monkeypatch.setattr(cv2, "imdecode", fail_if_decoded)
    validator = JpegFrameValidator()
    try:
        with pytest.raises(InvalidFrameError):
            await validator.validate(bytes(payload))
    finally:
        await validator.close()

    assert decode_called is False


async def test_cancelled_close_caller_does_not_interrupt_decoder_cleanup(monkeypatch):
    validator = JpegFrameValidator()
    decode_started = threading.Event()
    decode_release = threading.Event()
    shutdown_calls = 0
    original_shutdown = validator._executor.shutdown

    def blocking_validation(payload):
        decode_started.set()
        decode_release.wait(timeout=5)

    def tracked_shutdown(*args, **kwargs):
        nonlocal shutdown_calls
        shutdown_calls += 1
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(validator, "_validate_sync", blocking_validation)
    monkeypatch.setattr(validator._executor, "shutdown", tracked_shutdown)
    validation = asyncio.create_task(validator.validate(b"payload"))
    ready = await asyncio.to_thread(decode_started.wait, 1)
    assert ready is True

    first_close = asyncio.create_task(validator.close())
    await asyncio.sleep(0)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    decode_release.set()
    await validation
    await validator.close()

    assert shutdown_calls == 1
    with pytest.raises(RuntimeError, match="종료"):
        await validator.validate(b"payload")


async def test_cancelled_validation_holds_capacity_without_queueing(monkeypatch):
    validator = JpegFrameValidator()
    validation_started = threading.Event()
    validation_release = threading.Event()
    validation_calls = 0

    def blocking_validation(payload):
        nonlocal validation_calls
        validation_calls += 1
        validation_started.set()
        validation_release.wait(timeout=5)

    monkeypatch.setattr(validator, "_validate_sync", blocking_validation)
    first = asyncio.create_task(validator.validate(b"first"))
    ready = await asyncio.to_thread(validation_started.wait, 1)
    assert ready is True

    first.cancel()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    with pytest.raises(FrameValidationBusyError):
        await validator.validate(b"second")
    assert validation_calls == 1

    native_validation = next(iter(validator._running))
    validation_release.set()
    await native_validation
    await asyncio.sleep(0)
    await validator.validate(b"third")
    await validator.close()

    assert validation_calls == 2
