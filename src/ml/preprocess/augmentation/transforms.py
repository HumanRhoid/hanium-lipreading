"""입술 ROI 영상 클립에 적용하는 개별 데이터 증강 함수.

모든 함수는 다음 형식의 NumPy 배열을 입력받는다.

- shape: (T, H, W, C)
- dtype: uint8
- pixel range: 0~255

T는 프레임 수이며, 현재 전처리 파이프라인에서는
기본적으로 (60, 96, 192, 3) 형태의 배열이 입력된다.
"""

from __future__ import annotations

import cv2
import numpy as np


def validate_clip(clip: np.ndarray) -> None:
    """증강 입력 클립의 형태와 자료형을 검사한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.

    Raises:
        TypeError: 입력이 NumPy 배열이 아니거나 dtype이 uint8이 아닌 경우.
        ValueError: 배열 차원, 채널 수 또는 프레임 수가 올바르지 않은 경우.
    """
    if not isinstance(clip, np.ndarray):
        raise TypeError("clip은 NumPy 배열이어야 합니다.")

    if clip.ndim != 4:
        raise ValueError(
            f"clip의 shape은 (T, H, W, C) 형태여야 합니다. 현재 shape: {clip.shape}"
        )

    if clip.shape[0] == 0:
        raise ValueError("clip에는 최소 1개 이상의 프레임이 있어야 합니다.")

    if clip.shape[-1] not in (1, 3):
        raise ValueError(
            f"clip의 채널 수는 1 또는 3이어야 합니다. 현재 채널 수: {clip.shape[-1]}"
        )

    if clip.dtype != np.uint8:
        raise TypeError(f"clip의 dtype은 uint8이어야 합니다. 현재 dtype: {clip.dtype}")


def adjust_brightness(clip: np.ndarray, delta: float) -> np.ndarray:
    """클립 전체의 밝기를 동일한 값만큼 변경한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        delta: 픽셀에 더할 밝기 값. 예: -20~20.

    Returns:
        밝기가 조정된 uint8 영상 배열.
    """
    validate_clip(clip)

    result = clip.astype(np.float32) + float(delta)
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(clip: np.ndarray, factor: float) -> np.ndarray:
    """클립 전체의 대비를 동일한 비율로 변경한다.

    factor가 1이면 원본과 같고, 1보다 작으면 대비가 감소하며,
    1보다 크면 대비가 증가한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        factor: 대비 배율. 예: 0.8~1.2.

    Returns:
        대비가 조정된 uint8 영상 배열.

    Raises:
        ValueError: factor가 음수인 경우.
    """
    validate_clip(clip)

    if factor < 0:
        raise ValueError("contrast factor는 0 이상이어야 합니다.")

    clip_float = clip.astype(np.float32)

    # 각 프레임의 평균 밝기를 중심으로 대비를 조절한다.
    frame_means = clip_float.mean(axis=(1, 2), keepdims=True)
    result = (clip_float - frame_means) * float(factor) + frame_means

    return np.clip(result, 0, 255).astype(np.uint8)


def add_gaussian_noise(
    clip: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """클립에 가우시안 노이즈를 추가한다.

    노이즈 강도(std)는 클립 전체에서 동일하지만,
    실제 노이즈 패턴은 프레임마다 다르게 생성된다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        std: 가우시안 노이즈의 표준편차. 예: 2~8.
        rng: 재현 가능한 난수 생성을 위한 NumPy Generator.

    Returns:
        가우시안 노이즈가 추가된 uint8 영상 배열.

    Raises:
        ValueError: std가 음수인 경우.
    """
    validate_clip(clip)

    if std < 0:
        raise ValueError("noise std는 0 이상이어야 합니다.")

    noise = rng.normal(
        loc=0.0,
        scale=float(std),
        size=clip.shape,
    ).astype(np.float32)

    result = clip.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gaussian_blur(
    clip: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 0.0,
) -> np.ndarray:
    """클립의 모든 프레임에 동일한 가우시안 블러를 적용한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        kernel_size: 블러 커널 크기. 양의 홀수여야 한다.
        sigma: 가우시안 표준편차. 0이면 OpenCV가 자동 계산한다.

    Returns:
        블러가 적용된 uint8 영상 배열.

    Raises:
        ValueError: kernel_size가 양의 홀수가 아니거나 sigma가 음수인 경우.
    """
    validate_clip(clip)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size는 양의 홀수여야 합니다.")

    if sigma < 0:
        raise ValueError("blur sigma는 0 이상이어야 합니다.")

    blurred_frames = [
        cv2.GaussianBlur(
            frame,
            (kernel_size, kernel_size),
            sigmaX=float(sigma),
        )
        for frame in clip
    ]

    return np.stack(blurred_frames).astype(np.uint8)


def rotate_clip(
    clip: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    """클립의 모든 프레임을 동일한 각도로 회전한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        angle_degrees: 회전 각도. 양수는 반시계 방향이다.

    Returns:
        회전된 uint8 영상 배열.
    """
    validate_clip(clip)

    _, height, width, _ = clip.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center=center,
        angle=float(angle_degrees),
        scale=1.0,
    )

    rotated_frames = [
        cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        for frame in clip
    ]

    return np.stack(rotated_frames).astype(np.uint8)


def shift_clip(
    clip: np.ndarray,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    """클립의 모든 프레임을 동일한 픽셀만큼 이동한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        shift_x: 가로 이동 픽셀. 양수는 오른쪽이다.
        shift_y: 세로 이동 픽셀. 양수는 아래쪽이다.

    Returns:
        이동된 uint8 영상 배열.
    """
    validate_clip(clip)

    _, height, width, _ = clip.shape

    matrix = np.array(
        [
            [1.0, 0.0, float(shift_x)],
            [0.0, 1.0, float(shift_y)],
        ],
        dtype=np.float32,
    )

    shifted_frames = [
        cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        for frame in clip
    ]

    return np.stack(shifted_frames).astype(np.uint8)


def zoom_clip(
    clip: np.ndarray,
    scale: float,
) -> np.ndarray:
    """클립의 모든 프레임을 중심 기준으로 동일하게 확대 또는 축소한다.

    출력 해상도는 입력과 동일하게 유지된다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        scale: 확대·축소 배율.
            - 1.0: 원본 크기
            - 1.1: 10% 확대
            - 0.9: 10% 축소

    Returns:
        확대 또는 축소된 uint8 영상 배열.

    Raises:
        ValueError: scale이 0 이하인 경우.
    """
    validate_clip(clip)

    if scale <= 0:
        raise ValueError("zoom scale은 0보다 커야 합니다.")

    _, height, width, _ = clip.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center=center,
        angle=0.0,
        scale=float(scale),
    )

    zoomed_frames = [
        cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        for frame in clip
    ]

    return np.stack(zoomed_frames).astype(np.uint8)


def apply_jpeg_compression(
    clip: np.ndarray,
    quality: int,
) -> np.ndarray:
    """각 프레임을 JPEG로 인코딩·디코딩하여 압축 손실을 모사한다.

    Args:
        clip: (T, H, W, C) 형태의 uint8 영상 배열.
        quality: JPEG 품질. 1~100 범위이며 낮을수록 압축 손실이 크다.

    Returns:
        JPEG 압축 손실이 적용된 uint8 영상 배열.

    Raises:
        ValueError: quality가 1~100 범위를 벗어나거나 인코딩에 실패한 경우.
    """
    validate_clip(clip)

    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality는 1에서 100 사이여야 합니다.")

    compressed_frames: list[np.ndarray] = []

    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,
        int(quality),
    ]

    for frame in clip:
        encode_success, encoded = cv2.imencode(
            ".jpg",
            frame,
            encode_params,
        )

        if not encode_success:
            raise ValueError("JPEG 프레임 인코딩에 실패했습니다.")

        decode_flag = cv2.IMREAD_GRAYSCALE if frame.shape[-1] == 1 else cv2.IMREAD_COLOR

        decoded = cv2.imdecode(encoded, decode_flag)

        if decoded is None:
            raise ValueError("JPEG 프레임 디코딩에 실패했습니다.")

        if frame.shape[-1] == 1 and decoded.ndim == 2:
            decoded = decoded[..., np.newaxis]

        compressed_frames.append(decoded)

    return np.stack(compressed_frames).astype(np.uint8)
