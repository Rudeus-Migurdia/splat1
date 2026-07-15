import numpy as np
import pytest

from build_joint_semantic_vocabulary import balanced_training_features


class SourceStub:
    def __init__(self, features, valid_mask):
        self.features = np.asarray(features, dtype=np.float32)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        self.feature_dim = self.features.shape[1]

    def read(self, indices):
        return self.features[indices]


def test_joint_vocabulary_samples_both_modes_equally():
    base = SourceStub([[1, 0], [2, 0], [3, 0]], [True, True, True])
    candidate = SourceStub([[0, 1], [0, 2], [0, 3]], [True, True, True])
    features = balanced_training_features(base, candidate, 2, seed=0)
    assert features.shape == (4, 2)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert np.allclose(features[:2, 1], 0.0)
    assert np.allclose(features[2:, 0], 0.0)


def test_joint_vocabulary_rejects_dimension_mismatch():
    base = SourceStub([[1, 0]], [True])
    candidate = SourceStub([[1, 0, 0]], [True])
    with pytest.raises(ValueError):
        balanced_training_features(base, candidate, 1, seed=0)
