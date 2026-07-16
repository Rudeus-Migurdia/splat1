import numpy as np
import pytest


pytest.importorskip("torch")

from build_residual_vocabulary_extension import (
    assign_residual_codes,
    reconstruct_numpy,
)


class FixedIndex:
    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int32)

    def search(self, features):
        assert features.shape[0] == self.ids.shape[0]
        return self.ids


def test_reconstruct_numpy_uses_relative_slot_weights():
    codebook = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([[0, 1]], dtype=np.int32)
    weights = np.array([[1.0, 0.5]], dtype=np.float32)
    output = reconstruct_numpy(codebook, ids, weights)
    expected = np.array([[1.0, 0.5]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(output, expected, atol=1e-6)


def test_residual_code_is_kept_only_when_it_improves_cosine():
    target = np.array([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32)
    current = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    extension = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    ids, coefficients, accepted, cosine = assign_residual_codes(
        target,
        current,
        extension,
        FixedIndex([0, 1]),
        min_gain=1e-5,
    )
    np.testing.assert_array_equal(ids, [0, 1])
    assert coefficients[0] > 0.0 and accepted[0]
    assert coefficients[1] == 0.0 and not accepted[1]
    assert cosine[0] > np.dot(target[0], current[0])
