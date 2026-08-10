"""립리딩을 위한 시간 모델."""

from torch import nn


class TemporalBiGRU(nn.Module):
    """양방향 GRU로 프레임 사이의 장기 시간 관계를 학습한다.

    단기 시간 관계만 보는 3D-Conv와 달리 발화 전체의 순서를 해석한다.
    입력 shape은 ``[배치, 시간, input_dim]``이고 출력 shape은
    ``[배치, 시간, hidden_dim * 2]``이다. 시간축 길이는 유지한다.

    Args:
        input_dim: 백본이 출력하는 특징 차원.
        hidden_dim: GRU 은닉 상태의 차원. 양방향이라 출력은 두 배가 된다.
        num_layer: GRU 층 수.
        dropout: 층 사이에 적용할 dropout 확률. 단층이면 무시된다.
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_layer=2, dropout=0.2):
        super().__init__()
        self._validate_config(input_dim, hidden_dim, num_layer, dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.output_dim = hidden_dim * 2

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layer > 1 else 0.0,
        )

    @staticmethod
    def _validate_config(input_dim, hidden_dim, num_layer, dropout):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if isinstance(num_layer, bool) or not isinstance(num_layer, int):
            raise TypeError("num_layer must be an integer")
        if num_layer < 1:
            raise ValueError("num_layer must be positive")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

    def _validate_input(self, features):
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

    def forward(self, features):
        """프레임 특징 시퀀스를 시간 문맥이 반영된 특징으로 변환한다."""
        self._validate_input(features)
        outputs, _ = self.gru(features)
        return outputs
