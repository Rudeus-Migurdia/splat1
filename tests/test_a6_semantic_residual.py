import pytest

torch = pytest.importorskip("torch")

from train_a6_semantic_residual import (
    A6LowRankSemanticField,
    render_split_target,
    split_agreement_confidence,
    weighted_segment_contrastive_loss,
)


def test_zero_residual_preserves_a6_weighted_render():
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    field = A6LowRankSemanticField(base, torch.tensor([True, True]), 2, False)
    rendered, valid = field.render(
        torch.tensor([[0, 1]]),
        torch.tensor([[0.75, 0.25]]),
    )
    expected = torch.nn.functional.normalize(torch.tensor([[0.75, 0.25]]), dim=-1)
    assert valid.tolist() == [True]
    assert torch.allclose(rendered, expected)


def test_semantic_opacity_can_suppress_inconsistent_membership():
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    field = A6LowRankSemanticField(base, torch.tensor([True, True]), 2, True)
    with torch.no_grad():
        field.opacity_log_scale.weight[:, 0] = torch.tensor([0.0, -4.0])
    rendered, _ = field.render(
        torch.tensor([[0, 1]]),
        torch.tensor([[0.5, 0.5]]),
    )
    assert rendered[0, 0] > 0.99
    assert rendered[0, 1] < 0.02


def test_opposite_split_target_requires_supported_gaussians():
    target, valid = render_split_target(
        torch.tensor([[0, 1], [1, -1]]),
        torch.ones(2, 2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([2.0, 0.0]),
    )
    assert valid.tolist() == [True, False]
    assert torch.allclose(target[0], torch.tensor([1.0, 0.0]))


def test_split_agreement_filters_unreliable_observations():
    observation = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    split_target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    confidence, cosine = split_agreement_confidence(
        observation,
        split_target,
        torch.tensor([True, True]),
        0.5,
    )
    assert cosine.tolist() == pytest.approx([1.0, 0.0])
    assert confidence.tolist() == pytest.approx([1.0, 0.0])


def test_segment_contrastive_loss_respects_reliability_weights():
    prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    segment_features = torch.eye(2)
    loss = weighted_segment_contrastive_loss(
        prediction,
        torch.tensor([0, 1]),
        segment_features,
        torch.tensor([1.0, 0.0]),
        0.1,
    )
    assert loss < 1e-3
