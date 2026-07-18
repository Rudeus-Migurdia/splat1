import torch

from build_fine_part_shared_codebook import select_fine_tokens


def test_select_fine_tokens_enforces_identity_size_and_semantic_gates():
    selected = select_fine_tokens(
        sizes=torch.tensor([2, 3, 16, 33, 8]),
        reliability=torch.tensor([0.9, 0.6, 0.8, 0.9, 0.59]),
        disagreement=torch.tensor([0.2, 0.05, 0.2, 0.2, 0.2]),
        supported=torch.tensor([True, True, True, True, True]),
        min_size=3,
        max_size=32,
        min_reliability=0.6,
        min_disagreement=0.05,
    )
    assert selected.tolist() == [False, True, True, False, False]


def test_select_fine_tokens_requires_split_support():
    selected = select_fine_tokens(
        sizes=torch.tensor([8]),
        reliability=torch.tensor([0.9]),
        disagreement=torch.tensor([0.2]),
        supported=torch.tensor([False]),
        min_size=3,
        max_size=32,
        min_reliability=0.6,
        min_disagreement=0.05,
    )
    assert not selected.item()
