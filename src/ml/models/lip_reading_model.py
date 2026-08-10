"""백본·시간 모델·분류 헤드를 잇는 종단 간 립리딩 모델."""

from torch import nn

from .backbone import LipReadingBackbone
from .classification_head import ClassificationHead
from .temporal import TemporalBiGRU


class LipReadingModel(nn.Module):
    """백본·시간 모델·분류 헤드를 잇는 종단 간 립리딩 모델."""

    def __init__(self, num_classes, hidden_dim=256, num_layer=2, dropout=0.2):
        super().__init__()
        self.backbone = LipReadingBackbone()
        self.temporal = TemporalBiGRU(
            input_dim=self.backbone.feature_dim,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            dropout=dropout,
        )
        self.head = ClassificationHead(
            input_dim=self.temporal.output_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, frames):
        features = self.backbone(frames)
        features = self.temporal(features)
        return self.head(features)
