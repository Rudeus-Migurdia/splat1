import pytest

torch = pytest.importorskip("torch")

from compute_gaussian_max_contribution import update_max_contribution


def test_max_contribution_reduces_across_pixels_and_calls():
    output = torch.zeros(4)
    update_max_contribution(
        output,
        torch.tensor([[0, 1, -1], [0, 2, 3]]),
        torch.tensor([[0.2, 0.7, 1.0], [0.8, 0.3, 0.1]]),
        chunk_size=3,
    )
    update_max_contribution(
        output,
        torch.tensor([[1, 2]]),
        torch.tensor([[0.4, 0.9]]),
        chunk_size=1,
    )
    assert torch.allclose(output, torch.tensor([0.8, 0.7, 0.9, 0.1]))
