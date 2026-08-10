"""폐쇄형 문구 인식을 위한 분류 헤드."""

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """시간축 특징을 문구별 로짓으로 변환한다.

    ``features``는 ``[B, T, D]`` 형태이며 ``padding_mask``에서 ``True``인
    시점은 평균 풀링에서 제외한다. 반환값은 softmax를 적용하지 않은
    ``[B, N]`` 로짓이다.

    Args:
        input_dim: 시간 모델이 출력하는 특징 차원 ``D``.
        num_classes: 분류할 문구 개수 ``N``.
        dropout: FC 계층 앞에 적용할 dropout 확률.
    """

    def __init__(self, input_dim=512, num_classes=None, dropout=0.2):
        super().__init__()
        self._validate_config(input_dim, num_classes, dropout)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    @staticmethod
    def _validate_config(input_dim, num_classes, dropout):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

    def _validate_input(self, features, padding_mask):
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 3:
            raise ValueError(
                "expected features shape [B, T, D], "
                f"but received {tuple(features.shape)}"
            )

        batch_size, time_steps, feature_dim = features.shape
        if batch_size < 1 or time_steps < 1:
            raise ValueError("batch and time dimensions must be positive")
        if feature_dim != self.input_dim:
            raise ValueError(
                f"expected feature dimension {self.input_dim}, "
                f"but received {feature_dim}"
            )
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")

        if padding_mask is None:
            return
        if not isinstance(padding_mask, torch.Tensor):
            raise TypeError("padding_mask must be a torch.Tensor")
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must use torch.bool dtype")
        if padding_mask.shape != (batch_size, time_steps):
            raise ValueError(
                "expected padding_mask shape "
                f"({batch_size}, {time_steps}), but received {tuple(padding_mask.shape)}"
            )
        if padding_mask.device != features.device:
            raise ValueError("features and padding_mask must be on the same device")
        if padding_mask.all(dim=1).any():
            raise ValueError("each sample must contain at least one valid time step")

    @staticmethod
    def _temporal_mean(features, padding_mask):
        if padding_mask is None:
            return features.mean(dim=1)

        valid_steps = (~padding_mask).unsqueeze(-1)
        valid_count = valid_steps.sum(dim=1)
        return (features * valid_steps).sum(dim=1) / valid_count

    def forward(self, features, padding_mask=None):
        """시간축 특징을 평균 풀링한 뒤 클래스 로짓을 반환한다."""
        self._validate_input(features, padding_mask)
        pooled_features = self._temporal_mean(features, padding_mask)
        return self.classifier(self.dropout(pooled_features))
