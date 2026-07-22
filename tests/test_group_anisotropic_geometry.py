import numpy as np

from build_group_anisotropic_geometry import weighted_group_shape


def test_weighted_group_shape_detects_a_linear_group():
    points = np.array([[x, 0.01 * (-1) ** x, 0.0] for x in range(8)], dtype=np.float32)
    axes, ratios, linearity, planarity = weighted_group_shape(points, np.ones(8))
    assert axes.shape == (3, 3)
    assert ratios[0] == 1.0
    assert ratios[1] < 0.02
    assert linearity > 0.99
    assert planarity < 0.01


def test_weighted_group_shape_keeps_degenerate_groups_isotropic():
    _, ratios, linearity, planarity = weighted_group_shape(
        np.zeros((2, 3), dtype=np.float32), np.ones(2)
    )
    assert np.array_equal(ratios, np.ones(3, dtype=np.float32))
    assert linearity == 0.0
    assert planarity == 0.0
