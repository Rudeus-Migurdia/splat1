import torch

from build_split_reproducible_semantic_modes import select_reproducible_modes
from reweight_sparse_semantic_hypothesis import directional_reweight


def test_selects_only_cross_split_reproducible_separated_mode():
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    split_sums = torch.tensor(
        [
            [[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0], [0.0, 1.5]],
        ]
    )
    split_weights = torch.tensor([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
    split_support = torch.tensor([[3, 3, 3], [3, 3, 1]])

    result = select_reproducible_modes(
        base,
        split_sums,
        split_weights,
        split_support,
        min_views_per_split=2,
        min_compactness=0.9,
        min_cross_split_cosine=0.9,
        max_base_cosine=0.8,
        support_saturation=3,
    )

    assert result["selected"].tolist() == [True, False, False]
    assert torch.allclose(result["candidate"][0], torch.tensor([0.0, 1.0]))
    assert result["reliability"][0] > 0.99
    assert result["reliability"][1:].eq(0).all()


def test_rejects_diffuse_mode_even_when_split_centers_match():
    base = torch.tensor([[1.0, 0.0]])
    split_sums = torch.tensor([[[0.0, 1.5]], [[0.0, 1.5]]])
    split_weights = torch.tensor([[3.0], [3.0]])
    split_support = torch.tensor([[3], [3]])

    result = select_reproducible_modes(
        base,
        split_sums,
        split_weights,
        split_support,
        min_views_per_split=2,
        min_compactness=0.75,
        min_cross_split_cosine=0.9,
        max_base_cosine=0.8,
        support_saturation=3,
    )

    assert not result["selected"].item()
    assert result["compactness"][:, 0].tolist() == [0.5, 0.5]


def test_directional_reweight_suppresses_semantically_unrelated_mode():
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    hypotheses = torch.tensor([[0.8, 0.6], [0.0, 1.0]])
    weighted, agreement = directional_reweight(
        base, hypotheses, torch.tensor([0.5, 0.5])
    )

    assert torch.allclose(agreement, torch.tensor([0.8, 0.0]))
    assert torch.allclose(weighted, torch.tensor([0.4, 0.0]))
