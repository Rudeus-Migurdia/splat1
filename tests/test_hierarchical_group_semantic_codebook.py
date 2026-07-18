import pytest


torch = pytest.importorskip("torch")

from build_hierarchical_group_semantic_codebook import (
    aggregate_split_groups,
    fuse_sources,
)


def test_split_group_aggregation_uses_observation_weights():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    weights = torch.tensor([2.0, 1.0, 1.0])
    groups = __import__("numpy").array([0, 0, 1], dtype="int64")
    centers, totals, compactness, valid = aggregate_split_groups(
        features, weights, groups, 2, "cpu", 2
    )
    assert totals.tolist() == pytest.approx([3.0, 1.0])
    assert valid.tolist() == [True, True]
    assert centers[0].float().tolist() == pytest.approx(
        [2.0 / 5.0**0.5, 1.0 / 5.0**0.5], abs=5e-4
    )
    assert compactness[0].item() == pytest.approx(5.0**0.5 / 3.0)


def test_source_fusion_uses_auxiliary_when_old_is_missing():
    old = {
        "features": torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float16),
        "reliability": torch.tensor([0.8, 0.0]),
        "supported": torch.tensor([True, False]),
    }
    auxiliary = {
        "features": torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float16),
        "reliability": torch.tensor([0.9, 0.7]),
        "supported": torch.tensor([True, True]),
    }
    features, reliability, valid, gate = fuse_sources(old, auxiliary, 1.5, 0.05)
    assert valid.tolist() == [True, True]
    assert gate[0] > 0.5
    assert gate[1].item() == 1.0
    assert features[1].tolist() == pytest.approx([0.0, 1.0])
    assert reliability[1].item() == pytest.approx(0.7)
