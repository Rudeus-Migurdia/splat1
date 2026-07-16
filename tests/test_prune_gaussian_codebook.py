import numpy as np

from prune_gaussian_codebook import prune_sparse_codebook_arrays


def test_prune_sparse_codebook_filters_base_and_overflow_ids():
    arrays = prune_sparse_codebook_arrays(
        np.array([1, 2, 3], dtype=np.uint16),
        np.array([True, True, True]),
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([4, 5, 6], dtype=np.uint16),
        np.ones(3, dtype=np.uint8),
        np.full(3, 128, dtype=np.uint8),
        np.array([True, False, True]),
        65535,
    )
    base, valid, points, ids, slots, weights = arrays
    assert base.tolist() == [1, 65535, 3]
    assert valid.tolist() == [True, False, True]
    assert points.tolist() == [0, 2]
    assert ids.tolist() == [4, 6]
    assert slots.tolist() == [1, 1]
    assert weights.tolist() == [128, 128]
