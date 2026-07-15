import pytest

torch = pytest.importorskip("torch")

from prepare_semantic_field import visibility_truncate_weights


def test_visibility_truncation_keeps_cumulative_render_mass():
    ids = torch.tensor([[0, 1, 2, 3]])
    weights = torch.tensor([[0.60, 0.20, 0.10, 0.05]])
    truncated, retained_mass, retained_count = visibility_truncate_weights(
        ids,
        weights,
        mass_fraction=0.90,
        relative_floor=0.01,
        min_contributors=1,
    )
    assert torch.allclose(truncated, torch.tensor([[0.60, 0.20, 0.10, 0.0]]))
    assert retained_mass.item() == pytest.approx(0.90 / 0.95)
    assert retained_count.item() == 3


def test_visibility_truncation_preserves_minimum_contributors():
    ids = torch.tensor([[0, 1, 2]])
    weights = torch.tensor([[0.80, 0.15, 0.05]])
    truncated, _, retained_count = visibility_truncate_weights(
        ids,
        weights,
        mass_fraction=0.50,
        relative_floor=0.50,
        min_contributors=2,
    )
    assert torch.allclose(truncated, torch.tensor([[0.80, 0.15, 0.0]]))
    assert retained_count.item() == 2


def test_visibility_truncation_is_order_independent_and_ignores_invalid_ids():
    ids = torch.tensor([[2, -1, 0, 1]])
    weights = torch.tensor([[0.10, 1.00, 0.60, 0.20]])
    truncated, _, retained_count = visibility_truncate_weights(
        ids,
        weights,
        mass_fraction=0.80,
        relative_floor=0.0,
        min_contributors=1,
    )
    assert torch.allclose(truncated, torch.tensor([[0.0, 0.0, 0.60, 0.20]]))
    assert retained_count.item() == 2


def test_visibility_defaults_preserve_all_valid_weights():
    ids = torch.tensor([[0, 1, -1]])
    weights = torch.tensor([[0.70, 0.30, 0.50]])
    truncated, retained_mass, retained_count = visibility_truncate_weights(ids, weights)
    assert torch.allclose(truncated, torch.tensor([[0.70, 0.30, 0.0]]))
    assert retained_mass.item() == pytest.approx(1.0)
    assert retained_count.item() == 2
