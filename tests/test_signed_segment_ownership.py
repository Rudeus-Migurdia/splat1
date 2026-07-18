import pytest


torch = pytest.importorskip("torch")

from prepare_semantic_field import (
    aggregate_owned_view_observations,
    aggregate_view_observations,
    apply_signed_segment_ownership,
    kl_constrained_importance_ratios,
    segment_view_importance,
    signed_segment_ownership,
)


def test_compact_view_aggregation_omits_pixel_reverse_index():
    ids, weights, sums, reverse = aggregate_view_observations(
        torch.tensor([[0, 1], [0, -1]]),
        torch.tensor([[0.5, 0.25], [0.5, 0.0]]),
        torch.tensor([0, 1]),
        torch.eye(2),
        return_pixel_indices=False,
    )

    assert ids.tolist() == [0, 1]
    assert weights.tolist() == pytest.approx([1.0, 0.25])
    assert torch.allclose(sums, torch.tensor([[0.5, 0.5], [0.25, 0.0]]))
    assert reverse is None


def test_owned_view_aggregation_matches_dense_aggregation():
    point_ids = torch.tensor([[0, 1], [0, 1]])
    point_weights = torch.tensor([[0.6, 0.0], [0.0, 0.4]])
    segment_ids = torch.tensor([0, 1])
    dominant_segment = torch.tensor([0, 1])
    features = torch.eye(2)

    dense = aggregate_view_observations(
        point_ids,
        point_weights,
        segment_ids,
        features,
        return_pixel_indices=False,
    )
    owned = aggregate_owned_view_observations(
        point_ids,
        point_weights,
        dominant_segment,
        features,
    )

    for dense_value, owned_value in zip(dense[:3], owned[:3]):
        assert torch.equal(dense_value, owned_value)
    assert owned[3] is None


def test_signed_ownership_uses_foreground_minus_all_competing_mass():
    point_ids = torch.tensor(
        [
            [0, 1, -1],
            [0, -1, -1],
            [0, 1, 2],
            [2, -1, -1],
        ]
    )
    point_weights = torch.tensor(
        [
            [0.6, 0.2, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.4, 0.5],
            [0.5, 0.0, 0.0],
        ]
    )
    segment_ids = torch.tensor([0, 0, 1, 0])

    dominant, confidence, dominant_mass, total = signed_segment_ownership(
        point_ids, point_weights, segment_ids, num_gaussians=4
    )

    assert dominant.tolist() == [0, 1, 0, -1]
    assert dominant_mass.tolist() == pytest.approx([0.8, 0.4, 0.5, 0.0])
    assert total.tolist() == pytest.approx([1.1, 0.6, 1.0, 0.0])
    assert confidence.tolist() == pytest.approx(
        [(0.8 - 0.3) / 1.1, (0.4 - 0.2) / 0.6, 0.0, 0.0]
    )


def test_signed_ownership_only_keeps_the_winning_segment():
    point_ids = torch.tensor([[0, 1], [0, 1]])
    point_weights = torch.tensor([[0.6, 0.2], [0.3, 0.4]])
    segment_ids = torch.tensor([0, 1])
    dominant, confidence, _, _ = signed_segment_ownership(
        point_ids, point_weights, segment_ids, num_gaussians=2
    )

    gated = apply_signed_segment_ownership(
        point_ids, point_weights, segment_ids, dominant, confidence
    )

    assert gated[0, 0].item() == pytest.approx(0.6 * ((0.6 - 0.3) / 0.9))
    assert gated[0, 1].item() == 0.0
    assert gated[1, 0].item() == 0.0
    assert gated[1, 1].item() == pytest.approx(0.4 * ((0.4 - 0.2) / 0.6))


def test_signed_ownership_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        signed_segment_ownership(
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros((2, 1)),
            torch.zeros(2, dtype=torch.long),
            num_gaussians=2,
        )


def test_importance_ratios_respect_kl_and_ratio_constraints():
    behavior = torch.tensor([0.5, 0.3, 0.2])
    utility = torch.tensor([4.0, 0.0, -4.0])
    ratios, kl = kl_constrained_importance_ratios(
        behavior,
        utility,
        temperature=1.0,
        max_kl=0.02,
        ratio_clip=1.5,
    )

    target = behavior * ratios
    assert target.sum().item() == pytest.approx(1.0)
    assert kl.item() <= 0.020001
    assert ratios.max().item() <= 1.500001
    assert ratios[0] > 1.0
    assert ratios[2] < 1.0


def test_segment_importance_uses_opposite_split_agreement():
    segment_ids = torch.tensor([0, 0, 1, 1])
    masses = torch.ones(4)
    segment_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    references = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    )
    split_weights = torch.ones(4)
    total_weights = torch.full((4,), 4.0)

    ratios, diagnostics = segment_view_importance(
        segment_ids,
        masses,
        segment_features,
        references,
        split_weights,
        total_weights,
        max_kl=0.02,
        information_weight=0.0,
    )

    assert ratios[0] > 1.0
    assert ratios[1] < 1.0
    assert diagnostics["kl"] <= 0.020001
    assert diagnostics["mean_split_reliability"] == pytest.approx(1.0)


def test_segment_information_is_neutral_without_cross_split_support():
    ratios, diagnostics = segment_view_importance(
        torch.tensor([0, 1]),
        torch.ones(2),
        torch.eye(2),
        torch.eye(2),
        torch.zeros(2),
        torch.ones(2),
        information_weight=1.0,
    )

    assert ratios.tolist() == pytest.approx([1.0, 1.0])
    assert diagnostics["mean_split_reliability"] == 0.0
