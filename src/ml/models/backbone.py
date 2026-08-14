"""립리딩을 위한 시공간 시각 백본."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """표준 ResNet 기본 잔차 블록."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, inputs):
        identity = self.downsample(inputs)

        features = self.conv1(inputs)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.conv2(features)
        features = self.bn2(features)

        features = features + identity
        return self.relu(features)


class LipReadingBackbone(nn.Module):
    """입술 프레임마다 512차원 특징을 추출한다.
    입력 shape은 ``[배치, 3, 시간, 80, 112]``이고 출력 shape은
    ``[배치, 시간, 512]``이다. 시간축 길이는 축소하지 않는다.
    """

    input_channels = 3
    input_height = 80
    input_width = 112
    feature_dim = 512

    def __init__(self, pretrained=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                self.input_channels,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        self._resnet_channels = 64
        self.layer1 = self._make_layer(out_channels=64, block_count=2)
        self.layer2 = self._make_layer(out_channels=128, block_count=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, block_count=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, block_count=2, stride=2)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()
        if pretrained:
            self.load_imagenet_weights()

    def load_imagenet_weights(self):
        """ImageNet 학습 가중치를 layer1~layer4에 옮긴다.

        3D stem과 첫 2D 합성곱(``conv1``)은 대응하는 층이 없어 제외된다.
        ``conv1``만 원본 색을 직접 다루므로, 흑백 입력이어도 옮겨지는
        층들은 형태와 질감을 처리하는 부분이라 그대로 쓸 수 있다.
        """
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError:
            print("[알림] torchvision이 없어 사전학습 가중치를 건너뜁니다.")
            return 0

        source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        transferable = {
            key: value
            for key, value in source.items()
            if key.startswith(("layer1.", "layer2.", "layer3.", "layer4."))
        }

        missing, unexpected = self.load_state_dict(transferable, strict=False)
        if unexpected:
            raise RuntimeError(f"구조가 어긋난 가중치가 있습니다: {unexpected[:3]}")

        moved = sum(value.numel() for value in transferable.values())
        print(f"ImageNet 가중치 이식: 텐서 {len(transferable)}개 · {moved / 1e6:.2f}M")
        return len(transferable)

    def _make_layer(self, out_channels, block_count, stride=1):
        blocks = [BasicBlock(self._resnet_channels, out_channels, stride=stride)]
        self._resnet_channels = out_channels * BasicBlock.expansion
        blocks.extend(
            BasicBlock(self._resnet_channels, out_channels)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_input(self, frames):
        if not isinstance(frames, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if frames.ndim != 5:
            raise ValueError(
                "expected input shape [B, C, T, H, W], "
                f"but received {tuple(frames.shape)}"
            )

        batch_size, channels, frame_count, height, width = frames.shape
        if channels != self.input_channels:
            raise ValueError(f"expected 3 input channels, but received {channels}")
        if (height, width) != (self.input_height, self.input_width):
            raise ValueError(
                f"expected spatial size (80, 112), but received ({height}, {width})"
            )
        if batch_size < 1 or frame_count < 1:
            raise ValueError("batch and time dimensions must be positive")
        if not frames.is_floating_point():
            raise TypeError("input tensor must use a floating-point dtype")

    def forward(self, frames):
        self._validate_input(frames)
        batch_size, _, frame_count, _, _ = frames.shape

        features = self.stem(frames)
        _, channels, _, height, width = features.shape
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        features = features.view(batch_size * frame_count, channels, height, width)

        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.spatial_pool(features)
        features = torch.flatten(features, start_dim=1)

        return features.view(batch_size, frame_count, self.feature_dim)
