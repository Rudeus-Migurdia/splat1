import numpy as np

from build_point_supported_micro_capacity import prefer_micro


def test_micro_is_kept_only_when_point_evidence_prefers_it():
    points = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    fine = np.array([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32)
    micro = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    keep, margin = prefer_micro(points, fine, micro)
    assert keep.tolist() == [True, False]
    assert margin[0] > 0.0 and margin[1] < 0.0
