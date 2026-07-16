import numpy as np
import pytest


pytest.importorskip("torch")

from build_gaussian_superpoint_support import (
    BoundedUnionFind,
    compact_components,
    leave_one_out_candidate_support,
)


def test_bounded_union_find_never_exceeds_component_limit():
    union = BoundedUnionFind(6)
    assert union.union(0, 1, 3)
    assert union.union(1, 2, 3)
    assert not union.union(2, 3, 3)
    labels = compact_components(union)
    sizes = np.bincount(labels)
    assert sizes.max() == 3


def test_candidate_support_is_leave_one_out():
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    candidate = np.array([True, True, False, True, False])
    support, sizes, counts = leave_one_out_candidate_support(labels, candidate)
    np.testing.assert_allclose(support, [0.5, 0.5, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sizes, [3, 2])
    np.testing.assert_allclose(counts, [2.0, 1.0])
