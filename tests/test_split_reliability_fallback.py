import pytest


torch = pytest.importorskip("torch")

from build_split_reliability_fallback import blend_with_split_reliability


def test_split_reliability_blend_falls_back_and_adopts_candidate():
    baseline = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    candidate = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    valid = torch.ones(2, dtype=torch.bool)
    output, output_valid, gate = blend_with_split_reliability(
        baseline,
        candidate,
        valid,
        valid,
        torch.tensor([0.0, 1.0]),
        valid,
        torch.tensor([0.0, 1.0]),
        valid,
        torch.full((2,), 0.5),
    )

    assert output_valid.tolist() == [True, True]
    assert gate.tolist() == pytest.approx([0.0, 1.0])
    assert torch.allclose(output, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_split_reliability_blend_uses_available_source():
    valid = torch.ones(1, dtype=torch.bool)
    output, _, gate = blend_with_split_reliability(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        valid,
        valid,
        torch.tensor([0.25]),
        valid,
        torch.tensor([1.0]),
        torch.zeros(1, dtype=torch.bool),
        torch.tensor([1.0]),
    )

    assert gate.item() == pytest.approx(0.25)
    assert output.norm(dim=-1).item() == pytest.approx(1.0)
