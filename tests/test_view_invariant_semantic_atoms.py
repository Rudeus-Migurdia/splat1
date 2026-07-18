import numpy as np
import torch

from train_view_invariant_semantic_atoms import (
    select_competitors,
    train_atoms,
)


def test_select_competitors_prefers_supported_confusing_neighbor():
    local = np.array([0, 0, -1], dtype=np.int64)
    parts = np.array([0, 0, 1], dtype=np.int64)
    neighbors = np.array([[1, 2], [0, 2], [1, 0]], dtype=np.int64)
    fine = np.array([[1.0, 0.0]], dtype=np.float32)
    part_features = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    part_features /= np.linalg.norm(part_features, axis=-1, keepdims=True)
    competitor, edges = select_competitors(
        local, parts, neighbors, fine, part_features
    )
    assert competitor.tolist() == [1]
    assert edges.tolist() == [2]


def test_atom_training_preserves_positive_and_improves_margin():
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.98, 0.2]])
    second = torch.nn.functional.normalize(second, dim=-1)
    anchor = torch.nn.functional.normalize(first + second, dim=-1)
    competitor = torch.tensor([[0.8, 0.6]])
    competitor = torch.nn.functional.normalize(competitor, dim=-1)
    valid = torch.tensor([True])
    before = (
        torch.minimum((anchor * first).sum(-1), (anchor * second).sum(-1))
        - (anchor * competitor).sum(-1)
    )
    trained, _ = train_atoms(
        first,
        second,
        anchor,
        competitor,
        valid,
        steps=50,
        learning_rate=0.03,
        contrastive_margin=0.2,
        push_weight=1.0,
        anchor_weight=0.5,
    )
    after = (
        torch.minimum((trained * first).sum(-1), (trained * second).sum(-1))
        - (trained * competitor).sum(-1)
    )
    assert after.item() > before.item()
    assert torch.minimum((trained * first).sum(-1), (trained * second).sum(-1)).item() > 0.9
