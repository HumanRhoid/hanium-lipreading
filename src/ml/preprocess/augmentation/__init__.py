"""입술 ROI 영상 데이터 증강 모듈.

학습 코드에서는 다음과 같이 사용할 수 있습니다.

    from src.preprocess.augmentation import (
        AugmentationConfig,
        AugmentationResult,
        VideoAugmentation,
    )
"""

from .pipeline import (
    AugmentationConfig,
    AugmentationResult,
    VideoAugmentation,
)

__all__ = [
    "AugmentationConfig",
    "AugmentationResult",
    "VideoAugmentation",
]
