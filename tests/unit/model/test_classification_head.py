"""폐쇄형 분류 헤드의 계약 테스트."""

import pytest

torch = pytest.importorskip(
    "torch",
    reason="PyTorch가 설치된 환경에서만 분류 헤드 테스트를 실행합니다.",
)


@pytest.fixture
def head():
    from ml.model import ClassificationHead

    torch.manual_seed(0)
    return ClassificationHead(input_dim=8, num_classes=4, dropout=0.0)


def test_head_returns_one_logit_vector_per_sample(head):
    features = torch.randn(3, 5, 8)

    logits = head(features)

    assert logits.shape == (3, 4)


def test_head_uses_mean_pooling_without_padding(head):
    head.eval()
    features = torch.randn(2, 4, 8)

    with torch.inference_mode():
        logits = head(features)
        expected = head.classifier(features.mean(dim=1))

    torch.testing.assert_close(logits, expected)


def test_head_excludes_padded_steps_from_mean(head):
    head.eval()
    features = torch.tensor(
        [
            [[1.0] * 8, [3.0] * 8, [100.0] * 8],
            [[2.0] * 8, [4.0] * 8, [6.0] * 8],
        ]
    )
    padding_mask = torch.tensor(
        [[False, False, True], [False, False, False]],
        dtype=torch.bool,
    )

    with torch.inference_mode():
        logits = head(features, padding_mask=padding_mask)
        expected_pooled = torch.tensor([[2.0] * 8, [4.0] * 8])
        expected = head.classifier(expected_pooled)

    torch.testing.assert_close(logits, expected)


def test_head_propagates_gradients(head):
    features = torch.randn(2, 3, 8, requires_grad=True)

    head(features).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert head.classifier.weight.grad is not None


@pytest.mark.parametrize(
    "kwargs,error_type,message",
    [
        ({"input_dim": 0, "num_classes": 3}, ValueError, "input_dim"),
        ({"input_dim": 8, "num_classes": 1}, ValueError, "num_classes"),
        ({"input_dim": 8, "num_classes": 3, "dropout": 1.0}, ValueError, "dropout"),
    ],
)
def test_head_rejects_invalid_configuration(kwargs, error_type, message):
    from ml.model import ClassificationHead

    with pytest.raises(error_type, match=message):
        ClassificationHead(**kwargs)


def test_head_rejects_wrong_feature_rank(head):
    with pytest.raises(ValueError, match="expected features shape"):
        head(torch.randn(2, 8))


def test_head_rejects_wrong_feature_dimension(head):
    with pytest.raises(ValueError, match="expected feature dimension 8"):
        head(torch.randn(2, 3, 7))


def test_head_rejects_wrong_mask_shape(head):
    features = torch.randn(2, 3, 8)
    padding_mask = torch.zeros(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="expected padding_mask shape"):
        head(features, padding_mask=padding_mask)


def test_head_rejects_non_boolean_mask(head):
    features = torch.randn(2, 3, 8)
    padding_mask = torch.zeros(2, 3)

    with pytest.raises(TypeError, match="torch.bool"):
        head(features, padding_mask=padding_mask)


def test_head_rejects_sample_with_only_padding(head):
    features = torch.randn(2, 3, 8)
    padding_mask = torch.tensor(
        [[False, True, True], [True, True, True]],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="at least one valid time step"):
        head(features, padding_mask=padding_mask)
