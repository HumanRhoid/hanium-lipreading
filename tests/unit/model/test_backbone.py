"""립리딩 시각 백본의 계약 테스트."""

import pytest

torch = pytest.importorskip(
    "torch",
    reason="PyTorch가 설치된 환경에서만 백본 계약 테스트를 실행합니다.",
)


@pytest.fixture(scope="module")
def backbone():
    from src.model import LipReadingBackbone

    torch.manual_seed(0)
    return LipReadingBackbone()


def test_backbone_returns_one_feature_vector_per_frame(backbone):
    backbone.eval()
    frames = torch.randn(2, 3, 3, 80, 112)

    with torch.inference_mode():
        features = backbone(frames)

    assert features.shape == (2, 3, 512)


def test_backbone_matches_agreed_3d_stem_and_resnet18_depth(backbone):
    stem_conv = backbone.stem[0]

    assert isinstance(stem_conv, torch.nn.Conv3d)
    assert stem_conv.in_channels == 3
    assert stem_conv.out_channels == 64
    assert stem_conv.kernel_size == (5, 7, 7)
    assert stem_conv.stride == (1, 2, 2)
    assert stem_conv.padding == (2, 3, 3)
    assert [
        len(layer)
        for layer in (
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
    ] == [2, 2, 2, 2]


@pytest.mark.parametrize("frame_count", [1, 5])
def test_backbone_preserves_variable_frame_count(backbone, frame_count):
    backbone.eval()
    frames = torch.randn(1, 3, frame_count, 80, 112)

    with torch.inference_mode():
        features = backbone(frames)

    assert features.shape == (1, frame_count, 512)


def test_backbone_propagates_gradients_through_3d_stem_and_resnet(backbone):
    backbone.train()
    frames = torch.randn(1, 3, 2, 80, 112, requires_grad=True)

    features = backbone(frames)
    features.mean().backward()

    assert features.shape == (1, 2, 512)
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()
    assert backbone.stem[0].weight.grad is not None
    assert backbone.layer4[-1].conv2.weight.grad is not None


def test_backbone_rejects_non_tensor_input(backbone):
    with pytest.raises(TypeError, match="input must be a torch.Tensor"):
        backbone([1, 2, 3])


def test_backbone_rejects_wrong_rank(backbone):
    frames = torch.randn(3, 4, 80, 112)

    with pytest.raises(ValueError, match="expected input shape"):
        backbone(frames)


def test_backbone_rejects_wrong_channel_count(backbone):
    frames = torch.randn(1, 1, 3, 80, 112)

    with pytest.raises(ValueError, match="expected 3 input channels"):
        backbone(frames)


@pytest.mark.parametrize("height,width", [(64, 112), (80, 96)])
def test_backbone_rejects_wrong_spatial_size(backbone, height, width):
    frames = torch.randn(1, 3, 3, height, width)

    with pytest.raises(ValueError, match="expected spatial size"):
        backbone(frames)


@pytest.mark.parametrize(
    "frames",
    [
        torch.empty(0, 3, 3, 80, 112),
        torch.empty(1, 3, 0, 80, 112),
    ],
)
def test_backbone_rejects_empty_batch_or_time(backbone, frames):
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        backbone(frames)


def test_backbone_rejects_integer_pixels(backbone):
    frames = torch.zeros(1, 3, 3, 80, 112, dtype=torch.uint8)

    with pytest.raises(TypeError, match="floating-point dtype"):
        backbone(frames)
