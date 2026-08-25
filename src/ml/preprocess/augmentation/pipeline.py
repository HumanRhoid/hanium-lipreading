"""입술 ROI 영상 클립용 데이터 증강 파이프라인.

전처리 완료 후 저장된 NumPy 배열을 입력받아
학습 시 온라인 데이터 증강을 적용한다.

입력 규격:
- shape: (T, H, W, C)
- dtype: uint8
- pixel range: 0~255

현재 프로젝트 기본 입력:
- shape: (60, 96, 192, 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .transforms import (
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    apply_gaussian_blur,
    apply_jpeg_compression,
    rotate_clip,
    shift_clip,
    validate_clip,
    zoom_clip,
)


@dataclass(frozen=True)
class AugmentationConfig:
    """데이터 증강의 적용 확률과 강도 범위를 관리한다."""

    brightness_probability: float = 0.5
    brightness_delta_range: tuple[float, float] = (-20.0, 20.0)

    contrast_probability: float = 0.5
    contrast_factor_range: tuple[float, float] = (0.8, 1.2)

    gaussian_noise_probability: float = 0.3
    gaussian_noise_std_range: tuple[float, float] = (2.0, 8.0)

    gaussian_blur_probability: float = 0.3
    gaussian_blur_kernel_sizes: tuple[int, ...] = (3,)
    gaussian_blur_sigma_range: tuple[float, float] = (0.1, 1.0)

    rotation_probability: float = 0.4
    rotation_degrees_range: tuple[float, float] = (-10.0, 10.0)

    shift_probability: float = 0.4
    shift_x_ratio_range: tuple[float, float] = (-0.05, 0.05)
    shift_y_ratio_range: tuple[float, float] = (-0.05, 0.05)

    zoom_probability: float = 0.4
    zoom_scale_range: tuple[float, float] = (0.9, 1.1)

    jpeg_probability: float = 0.3
    jpeg_quality_range: tuple[int, int] = (60, 95)

    def __post_init__(self) -> None:
        """설정값이 안전한 범위인지 검사한다."""
        probability_fields = {
            "brightness_probability": self.brightness_probability,
            "contrast_probability": self.contrast_probability,
            "gaussian_noise_probability": self.gaussian_noise_probability,
            "gaussian_blur_probability": self.gaussian_blur_probability,
            "rotation_probability": self.rotation_probability,
            "shift_probability": self.shift_probability,
            "zoom_probability": self.zoom_probability,
            "jpeg_probability": self.jpeg_probability,
        }

        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}은 0과 1 사이여야 합니다.")

        self._validate_range(
            "brightness_delta_range",
            self.brightness_delta_range,
        )
        self._validate_range(
            "contrast_factor_range",
            self.contrast_factor_range,
        )
        self._validate_range(
            "gaussian_noise_std_range",
            self.gaussian_noise_std_range,
        )
        self._validate_range(
            "gaussian_blur_sigma_range",
            self.gaussian_blur_sigma_range,
        )
        self._validate_range(
            "rotation_degrees_range",
            self.rotation_degrees_range,
        )
        self._validate_range(
            "shift_x_ratio_range",
            self.shift_x_ratio_range,
        )
        self._validate_range(
            "shift_y_ratio_range",
            self.shift_y_ratio_range,
        )
        self._validate_range(
            "zoom_scale_range",
            self.zoom_scale_range,
        )
        self._validate_range(
            "jpeg_quality_range",
            self.jpeg_quality_range,
        )

        if self.contrast_factor_range[0] < 0:
            raise ValueError("contrast factor의 최솟값은 0 이상이어야 합니다.")

        if self.gaussian_noise_std_range[0] < 0:
            raise ValueError("Gaussian Noise std의 최솟값은 0 이상이어야 합니다.")

        if self.gaussian_blur_sigma_range[0] < 0:
            raise ValueError("Gaussian Blur sigma의 최솟값은 0 이상이어야 합니다.")

        if self.zoom_scale_range[0] <= 0:
            raise ValueError("zoom scale의 최솟값은 0보다 커야 합니다.")

        if not self.gaussian_blur_kernel_sizes:
            raise ValueError("Gaussian Blur 커널 크기가 최소 하나 필요합니다.")

        for kernel_size in self.gaussian_blur_kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("Gaussian Blur 커널 크기는 양의 홀수여야 합니다.")

        jpeg_min, jpeg_max = self.jpeg_quality_range
        if jpeg_min < 1 or jpeg_max > 100:
            raise ValueError("JPEG quality 범위는 1에서 100 사이여야 합니다.")

    @staticmethod
    def _validate_range(
        name: str,
        value_range: tuple[float, float] | tuple[int, int],
    ) -> None:
        minimum, maximum = value_range
        if minimum > maximum:
            raise ValueError(f"{name}의 최솟값은 최댓값보다 클 수 없습니다.")


@dataclass
class AugmentationResult:
    """증강된 클립과 적용 정보를 함께 반환한다."""

    clip: np.ndarray
    applied: dict[str, dict[str, Any]] = field(default_factory=dict)


class VideoAugmentation:
    """입술 ROI 영상 클립에 여러 증강을 확률적으로 적용한다."""

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            config: 증강 확률과 범위 설정.
            seed: 동일한 증강 결과를 재현하기 위한 난수 시드.
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        clip: np.ndarray,
        return_details: bool = False,
    ) -> np.ndarray | AugmentationResult:
        """클립에 증강을 적용한다.

        Args:
            clip: (T, H, W, C) 형태의 uint8 배열.
            return_details:
                False이면 증강된 배열만 반환한다.
                True이면 적용된 증강 정보도 함께 반환한다.

        Returns:
            증강된 배열 또는 AugmentationResult.
        """
        validate_clip(clip)

        # 원본 배열이 변경되지 않도록 복사본에서 작업한다.
        augmented = clip.copy()
        applied: dict[str, dict[str, Any]] = {}

        augmented = self._apply_brightness(augmented, applied)
        augmented = self._apply_contrast(augmented, applied)
        augmented = self._apply_gaussian_noise(augmented, applied)
        augmented = self._apply_gaussian_blur(augmented, applied)
        augmented = self._apply_rotation(augmented, applied)
        augmented = self._apply_shift(augmented, applied)
        augmented = self._apply_zoom(augmented, applied)
        augmented = self._apply_jpeg_compression(augmented, applied)

        if return_details:
            return AugmentationResult(
                clip=augmented,
                applied=applied,
            )

        return augmented

    def _should_apply(self, probability: float) -> bool:
        """주어진 확률에 따라 증강 적용 여부를 결정한다."""
        return bool(self.rng.random() < probability)

    def _sample_float(
        self,
        value_range: tuple[float, float],
    ) -> float:
        """실수 범위에서 값을 하나 선택한다."""
        minimum, maximum = value_range
        return float(self.rng.uniform(minimum, maximum))

    def _sample_int(
        self,
        value_range: tuple[int, int],
    ) -> int:
        """양 끝을 포함하는 정수 범위에서 값을 하나 선택한다."""
        minimum, maximum = value_range
        return int(self.rng.integers(minimum, maximum + 1))

    def _apply_brightness(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.brightness_probability):
            return clip

        delta = self._sample_float(self.config.brightness_delta_range)
        applied["brightness"] = {"delta": delta}

        return adjust_brightness(
            clip,
            delta=delta,
        )

    def _apply_contrast(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.contrast_probability):
            return clip

        factor = self._sample_float(self.config.contrast_factor_range)
        applied["contrast"] = {"factor": factor}

        return adjust_contrast(
            clip,
            factor=factor,
        )

    def _apply_gaussian_noise(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.gaussian_noise_probability):
            return clip

        std = self._sample_float(self.config.gaussian_noise_std_range)
        applied["gaussian_noise"] = {"std": std}

        return add_gaussian_noise(
            clip,
            std=std,
            rng=self.rng,
        )

    def _apply_gaussian_blur(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.gaussian_blur_probability):
            return clip

        kernel_size = int(self.rng.choice(self.config.gaussian_blur_kernel_sizes))
        sigma = self._sample_float(self.config.gaussian_blur_sigma_range)

        applied["gaussian_blur"] = {
            "kernel_size": kernel_size,
            "sigma": sigma,
        }

        return apply_gaussian_blur(
            clip,
            kernel_size=kernel_size,
            sigma=sigma,
        )

    def _apply_rotation(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.rotation_probability):
            return clip

        angle = self._sample_float(self.config.rotation_degrees_range)
        applied["rotation"] = {
            "angle_degrees": angle,
        }

        return rotate_clip(
            clip,
            angle_degrees=angle,
        )

    def _apply_shift(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.shift_probability):
            return clip

        _, height, width, _ = clip.shape

        shift_x_ratio = self._sample_float(self.config.shift_x_ratio_range)
        shift_y_ratio = self._sample_float(self.config.shift_y_ratio_range)

        shift_x = shift_x_ratio * width
        shift_y = shift_y_ratio * height

        applied["shift"] = {
            "shift_x": shift_x,
            "shift_y": shift_y,
            "shift_x_ratio": shift_x_ratio,
            "shift_y_ratio": shift_y_ratio,
        }

        return shift_clip(
            clip,
            shift_x=shift_x,
            shift_y=shift_y,
        )

    def _apply_zoom(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.zoom_probability):
            return clip

        scale = self._sample_float(self.config.zoom_scale_range)
        applied["zoom"] = {"scale": scale}

        return zoom_clip(
            clip,
            scale=scale,
        )

    def _apply_jpeg_compression(
        self,
        clip: np.ndarray,
        applied: dict[str, dict[str, Any]],
    ) -> np.ndarray:
        if not self._should_apply(self.config.jpeg_probability):
            return clip

        quality = self._sample_int(self.config.jpeg_quality_range)
        applied["jpeg_compression"] = {
            "quality": quality,
        }

        return apply_jpeg_compression(
            clip,
            quality=quality,
        )
