import numpy as np
import pytest

torch = pytest.importorskip("torch")

from build_seeded_hierarchical_resident_memory import (  # noqa: E402
    assign_codebook_in_chunks,
    complete_resident_sources,
    deterministic_weighted_indices,
    select_group_feature_source,
)
from build_hierarchical_semantic_memory import (  # noqa: E402
    AUXILIARY_SOURCE,
    INVALID_SOURCE,
    OLD_SOURCE,
)


def source(features, reliability, supported):
    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "reliability": torch.tensor(reliability, dtype=torch.float32),
        "supported": torch.tensor(supported, dtype=torch.bool),
    }


def test_source_disagreement_falls_back_to_old_with_reduced_reliability():
    old = source([[1.0, 0.0], [1.0, 0.0]], [0.8, 0.6], [True, True])
    sam = source([[0.0, 1.0], [0.99, 0.1]], [0.95, 0.9], [True, True])

    _, reliability, selected, agreement, conflict = select_group_feature_source(
        old, sam, agreement_floor=0.8, source_margin=0.0
    )

    assert selected.tolist() == [int(OLD_SOURCE), int(AUXILIARY_SOURCE)]
    assert conflict.tolist() == [True, False]
    assert agreement[0] == pytest.approx(0.0)
    assert reliability[0] == pytest.approx(0.0)
    assert reliability[1] == pytest.approx(0.9)


def test_weighted_training_sample_is_seeded_and_unique():
    indices = np.arange(20)
    weights = np.linspace(1.0, 2.0, 20)
    first = deterministic_weighted_indices(indices, weights, maximum=8, seed=71)
    second = deterministic_weighted_indices(indices, weights, maximum=8, seed=71)

    assert first.tolist() == second.tolist()
    assert len(np.unique(first)) == 8


def test_group_assignment_searches_in_bounded_chunks():
    class Index:
        def __init__(self):
            self.batch_sizes = []

        def search(self, values):
            self.batch_sizes.append(values.shape[0])
            return np.arange(values.shape[0], dtype=np.int32)

    index = Index()
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    assigned = assign_codebook_in_chunks(
        index, features, np.array([1, 3, 4, 8, 9]), chunk_size=2
    )

    assert index.batch_sizes == [2, 2, 1]
    assert assigned[[1, 3, 4, 8, 9]].tolist() == [0, 1, 0, 1, 0]


def test_full_consensus_completes_missing_resident_slots_symmetrically():
    features = np.zeros((3, 2), dtype=np.float32)
    reliability = np.zeros(3, dtype=np.float32)
    selected = np.full(3, int(INVALID_SOURCE), dtype=np.uint8)
    old_full = (
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.ones(3),
        torch.tensor([0.8, 0.9, 0.7]),
        torch.tensor([True, True, False]),
    )
    sam_full = (
        torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        torch.ones(3),
        torch.tensor([0.6, 0.5, 0.4]),
        torch.tensor([True, False, False]),
    )
    output, output_reliability, source_ids, old_mask, sam_mask, unresolved = (
        complete_resident_sources(
            features, reliability, selected, old_full, sam_full, 0.05
        )
    )
    assert sam_mask.tolist() == [True, False, False]
    assert old_mask.tolist() == [False, True, False]
    assert unresolved.tolist() == [False, False, True]
    assert source_ids.tolist() == [int(AUXILIARY_SOURCE), int(OLD_SOURCE), int(INVALID_SOURCE)]
    assert output[0].tolist() == pytest.approx([0.0, 1.0])
    assert output[1].tolist() == pytest.approx([1.0, 0.0])
    assert output_reliability.tolist() == pytest.approx([0.03, 0.045, 0.0])
