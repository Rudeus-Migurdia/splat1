import numpy as np
import pytest


pytest.importorskip("torch")

from build_spatial_semantic_support_mask import semantic_neighbor_support


def test_support_ignores_semantically_unrelated_neighbors():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    neighbors = np.array(
        [[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.0, 1.0]]],
        dtype=np.float32,
    )
    neighbors /= np.linalg.norm(neighbors, axis=-1, keepdims=True)
    distance = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    candidate = np.array([[True, True, False, False]])
    support, count = semantic_neighbor_support(
        query, neighbors, distance, candidate, semantic_floor=0.9
    )
    assert count.tolist() == [2]
    assert support[0] == pytest.approx(1.0)


def test_support_reflects_candidate_fraction_among_similar_neighbors():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    neighbors = np.repeat(query[:, None, :], 4, axis=1)
    distance = np.ones((1, 4), dtype=np.float32)
    candidate = np.array([[True, False, True, False]])
    support, _ = semantic_neighbor_support(
        query, neighbors, distance, candidate, semantic_floor=0.9
    )
    assert support[0] == pytest.approx(0.5)
