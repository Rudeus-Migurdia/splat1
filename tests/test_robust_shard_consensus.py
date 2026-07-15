import pytest

torch = pytest.importorskip("torch")

from build_robust_shard_consensus import weighted_geometric_median


def test_geometric_median_rejects_one_view_shard_outlier():
    features = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]
    )
    weights = torch.ones((4, 1))
    center = weighted_geometric_median(features, weights, iterations=12)
    assert center[0, 0] > 0.99
    assert center[0, 1] < 0.05


def test_geometric_median_respects_shard_confidence():
    features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    weights = torch.tensor([[10.0], [1.0], [1.0]])
    center = weighted_geometric_median(features, weights, iterations=12)
    assert center[0, 0] > center[0, 1]


def test_geometric_median_returns_zero_without_support():
    features = torch.zeros((4, 2, 3))
    weights = torch.zeros((4, 2))
    center = weighted_geometric_median(features, weights)
    assert torch.count_nonzero(center) == 0
