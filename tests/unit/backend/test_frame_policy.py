"""영상 단위 상한과 균등 프레임 정규화 정책 테스트."""

import pytest

from src.backend.recognition.domain import (
    MODEL_INPUT_FRAME_COUNT,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_FRAMES,
    MIN_VIDEO_FRAMES,
)
from src.backend.recognition.errors import (
    InsufficientFramesError,
    VideoTooLargeError,
    VideoTooLongError,
)
from src.backend.recognition.frame_policy import (
    normalize_video_frames,
    uniform_frame_indices,
    validate_video_limits,
)


def test_v1_video_limits_are_fixed_contract_values():
    assert MODEL_INPUT_FRAME_COUNT == 60
    assert MIN_VIDEO_FRAMES == 60
    assert MAX_VIDEO_FRAMES == 250
    assert MAX_VIDEO_BYTES == 64 * 1024 * 1024


@pytest.mark.parametrize("frame_count", [60, 61, 250])
def test_uniform_indices_follow_the_documented_floor_formula(frame_count):
    indices = uniform_frame_indices(frame_count)

    assert indices == tuple(index * (frame_count - 1) // 59 for index in range(60))
    assert len(indices) == MODEL_INPUT_FRAME_COUNT
    assert indices[0] == 0
    assert indices[-1] == frame_count - 1
    assert all(left < right for left, right in zip(indices, indices[1:]))


def test_normalization_selects_frames_from_the_full_video():
    frames = tuple(f"frame-{index}".encode() for index in range(250))

    normalized = normalize_video_frames(frames)

    expected_indices = tuple(index * 249 // 59 for index in range(60))
    assert normalized == tuple(frames[index] for index in expected_indices)


def test_exactly_sixty_frames_are_kept_without_reordering():
    frames = tuple(f"frame-{index}".encode() for index in range(60))

    assert normalize_video_frames(frames) == frames


@pytest.mark.parametrize("frame_count", [0, 29])
def test_normalization_rejects_too_few_frames(frame_count):
    with pytest.raises(InsufficientFramesError):
        normalize_video_frames([b"frame"] * frame_count)


def test_video_limits_accept_both_exact_upper_boundaries():
    validate_video_limits(
        frame_count=MAX_VIDEO_FRAMES,
        total_bytes=MAX_VIDEO_BYTES,
    )


def test_video_limits_reject_the_251st_frame():
    with pytest.raises(VideoTooLongError):
        validate_video_limits(
            frame_count=MAX_VIDEO_FRAMES + 1,
            total_bytes=MAX_VIDEO_BYTES,
        )


def test_video_limits_reject_the_first_byte_over_64_mib():
    with pytest.raises(VideoTooLargeError):
        validate_video_limits(
            frame_count=MAX_VIDEO_FRAMES,
            total_bytes=MAX_VIDEO_BYTES + 1,
        )


@pytest.mark.parametrize(
    ("frame_count", "total_bytes"),
    [(-1, 0), (0, -1)],
)
def test_video_limits_reject_negative_internal_counters(frame_count, total_bytes):
    with pytest.raises(ValueError):
        validate_video_limits(frame_count=frame_count, total_bytes=total_bytes)
