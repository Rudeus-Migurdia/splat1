import numpy as np

from build_part_conditioned_signed_membership import (
    per_view_part_ownership,
    split_membership,
)


def test_group_winner_exposes_misassigned_gaussian():
    # Both Gaussians belong to one persistent 3D part. Gaussian 0 strongly anchors
    # segment 0, while Gaussian 1 consistently contributes to the competing mask.
    point_ids = np.array([[0, 1], [0, 1]], dtype=np.int64)
    weights = np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)
    segments = np.array([0, 1], dtype=np.int64)
    foreground, total, observed, diagnostics = per_view_part_ownership(
        point_ids,
        weights,
        segments,
        np.ones(2, dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        1,
    )
    assert diagnostics["positive_margin_parts"] == 1
    assert observed.tolist() == [True, True]
    assert foreground[0] > total[0] * 0.5
    assert foreground[1] < total[1] * 0.5


def test_split_membership_requires_reproducible_support():
    foreground = np.array([[0.8, 0.2], [0.7, 0.1]], dtype=np.float32)
    total = np.ones((2, 2), dtype=np.float32)
    views = np.array([[3, 3], [3, 2]], dtype=np.uint16)
    membership, signed, reliability, supported = split_membership(
        foreground, total, views, 3
    )
    assert supported.tolist() == [True, False]
    assert signed[0] > 0.0
    assert reliability[0] > 0.0
    assert reliability[1] == 0.0
    assert membership.shape == (2, 2)
