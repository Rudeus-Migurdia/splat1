import pytest

torch = pytest.importorskip("torch")

from blend_semantic_consensus import blend_consensus_features


def test_blend_consensus_features_uses_normalized_endpoints():
    base = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    candidate = torch.tensor([[0.0, 4.0], [0.0, 2.0]])
    result = blend_consensus_features(base, candidate, 0.5, chunk_size=1)
    expected = torch.tensor([[2**-0.5, 2**-0.5], [0.0, 1.0]]).half()
    assert torch.allclose(result, expected, atol=1e-3)


def test_blend_consensus_features_rejects_invalid_weight():
    with pytest.raises(ValueError):
        blend_consensus_features(torch.eye(2), torch.eye(2), 1.1)
