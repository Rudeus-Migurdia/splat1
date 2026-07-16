import pytest

torch = pytest.importorskip("torch")

from build_novelty_route_mask import semantic_novelty_and_quantization_noise


def test_semantic_novelty_is_compared_to_larger_quantization_noise():
    base = torch.tensor([[1.0, 0.0]])
    candidate = torch.tensor([[0.0, 1.0]])
    base_reconstruction = torch.tensor([[1.0, 0.0]])
    candidate_reconstruction = torch.tensor([[0.1, 0.9]])
    novelty, noise = semantic_novelty_and_quantization_noise(
        base, candidate, base_reconstruction, candidate_reconstruction
    )
    assert novelty.item() > noise.item()
    assert noise.item() > 0.0
