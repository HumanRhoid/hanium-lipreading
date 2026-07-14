import numpy as np

from src.preprocess.normalize import normalize_frames, to_grayscale_clahe


def test_normalize_frames_returns_none_for_empty_input():
    assert normalize_frames([]) is None


def test_to_grayscale_clahe_keeps_three_equal_channels():
    frame = np.array(
        [
            [[0, 40, 200], [200, 40, 0]],
            [[10, 120, 250], [250, 120, 10]],
        ],
        dtype=np.uint8,
    )

    result = to_grayscale_clahe(frame)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])


def test_normalize_frames_resizes_and_pads_with_last_frame():
    frames = [
        np.full((4, 6, 3), fill_value=20, dtype=np.uint8),
        np.full((4, 6, 3), fill_value=180, dtype=np.uint8),
    ]

    result = normalize_frames(
        frames,
        target_width=3,
        target_height=2,
        fixed_frame_count=4,
    )

    assert result.shape == (4, 2, 3, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[2], result[1])
    assert np.array_equal(result[3], result[1])


def test_normalize_frames_truncates_extra_frames():
    frames = [
        np.full((4, 6, 3), fill_value=value, dtype=np.uint8) for value in (20, 100, 180)
    ]

    result = normalize_frames(
        frames,
        target_width=3,
        target_height=2,
        fixed_frame_count=2,
    )
    expected = normalize_frames(
        frames[:2],
        target_width=3,
        target_height=2,
        fixed_frame_count=2,
    )

    assert np.array_equal(result, expected)
