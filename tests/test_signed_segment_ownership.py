import pytest


torch = pytest.importorskip("torch")

from prepare_semantic_field import (
    apply_signed_segment_ownership,
    signed_segment_ownership,
)


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

    dominant, confidence, total = signed_segment_ownership(
        point_ids, point_weights, segment_ids, num_gaussians=4
    )

    assert dominant.tolist() == [0, 1, 0, -1]
    assert total.tolist() == pytest.approx([1.1, 0.6, 1.0, 0.0])
    assert confidence.tolist() == pytest.approx(
        [(0.8 - 0.3) / 1.1, (0.4 - 0.2) / 0.6, 0.0, 0.0]
    )


def test_signed_ownership_only_keeps_the_winning_segment():
    point_ids = torch.tensor([[0, 1], [0, 1]])
    point_weights = torch.tensor([[0.6, 0.2], [0.3, 0.4]])
    segment_ids = torch.tensor([0, 1])
    dominant, confidence, _ = signed_segment_ownership(
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
