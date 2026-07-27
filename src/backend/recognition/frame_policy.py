"""v1 영상 입력 상한과 모델 입력 프레임 정규화 정책."""

from collections.abc import Sequence

from src.backend.recognition.domain import (
    INPUT_FRAME_COUNT,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_FRAMES,
    MIN_VIDEO_FRAMES,
)
from src.backend.recognition.errors import (
    InsufficientFramesError,
    VideoTooLargeError,
    VideoTooLongError,
)


def validate_video_limits(*, frame_count: int, total_bytes: int) -> None:
    """현재 영상 크기가 v1 bounded-buffer 계약 안인지 확인한다."""

    if frame_count < 0 or total_bytes < 0:
        raise ValueError("프레임 수와 누적 byte는 음수일 수 없습니다")
    if frame_count > MAX_VIDEO_FRAMES:
        raise VideoTooLongError(
            f"영상은 최대 {MAX_VIDEO_FRAMES}프레임까지 전송할 수 있습니다"
        )
    if total_bytes > MAX_VIDEO_BYTES:
        raise VideoTooLargeError(
            f"영상은 최대 {MAX_VIDEO_BYTES}byte까지 전송할 수 있습니다"
        )


def uniform_frame_indices(
    frame_count: int,
    *,
    output_count: int = INPUT_FRAME_COUNT,
) -> tuple[int, ...]:
    """처음과 마지막을 포함해 전체 구간의 index를 균등하게 고른다."""

    if frame_count < output_count:
        raise InsufficientFramesError(
            f"인식에는 최소 {output_count}프레임이 필요합니다"
        )
    if output_count < 2:
        raise ValueError("출력 프레임 수는 2 이상이어야 합니다")

    last_index = frame_count - 1
    denominator = output_count - 1
    return tuple(index * last_index // denominator for index in range(output_count))


def normalize_video_frames(frames: Sequence[bytes]) -> tuple[bytes, ...]:
    """v1 영상 전체를 모델 계약인 정확히 30프레임으로 정규화한다."""

    frame_count = len(frames)
    validate_video_limits(
        frame_count=frame_count,
        total_bytes=sum(len(frame) for frame in frames),
    )
    if frame_count < MIN_VIDEO_FRAMES:
        raise InsufficientFramesError(
            f"인식에는 최소 {MIN_VIDEO_FRAMES}프레임이 필요합니다"
        )

    indices = uniform_frame_indices(frame_count)
    return tuple(frames[index] for index in indices)
