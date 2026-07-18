import numpy as np

from build_micro_interior_membership import apply_micro_support


def test_only_micro_slot_is_scaled():
    weights = np.array([[100, 200, 255], [50, 60, 128]], dtype=np.uint8)
    output = apply_micro_support(weights, np.array([0.5, 0.0]), slot=2)
    assert output[:, :2].tolist() == weights[:, :2].tolist()
    assert output[:, 2].tolist() == [128, 0]
