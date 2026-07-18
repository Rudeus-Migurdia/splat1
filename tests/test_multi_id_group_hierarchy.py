import numpy as np

from build_multi_id_group_hierarchy import (
    build_multi_id_hierarchy,
    hierarchy_route_reliability,
    multi_mode_set_similarity,
)


def test_multi_mode_similarity_is_invariant_to_mode_order():
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0]], dtype=np.float32)
    score = multi_mode_set_similarity(x, y, y, x)
    np.testing.assert_allclose(score, np.ones(1))


def test_hierarchy_keeps_strict_parts_inside_looser_object():
    neighbors = np.array([[1, 2], [0, 2], [3, 1], [2, 1]], dtype=np.int32)
    distances = np.ones_like(neighbors, dtype=np.float32)
    rgb = np.zeros((4, 3), dtype=np.float32)
    scale = np.zeros(4, dtype=np.float32)
    base = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [0.8, 0.6]],
        dtype=np.float32,
    )
    candidate = base.copy()
    result = build_multi_id_hierarchy(
        neighbors,
        distances,
        rgb,
        scale,
        base,
        candidate,
        spatial_radius_factor=2.0,
        rgb_threshold=1.0,
        log_scale_threshold=1.0,
        part_base_threshold=0.95,
        part_set_threshold=0.95,
        object_base_threshold=0.75,
        object_set_threshold=0.75,
        maximum_part_size=2,
        maximum_object_size=4,
        chunk_size=4,
    )
    assert result["part_labels"][0] == result["part_labels"][1]
    assert result["part_labels"][2] == result["part_labels"][3]
    assert result["part_labels"][0] != result["part_labels"][2]
    assert np.unique(result["object_labels"]).size == 1


def test_route_expansion_preserves_original_candidates():
    routes = hierarchy_route_reliability(
        np.array([0, 0, 0, 0, 1]),
        np.array([0, 0, 0, 0, 0]),
        np.array([True, True, False, False, True]),
        min_part_size=3,
        min_object_size=5,
        min_part_density=0.5,
        min_object_density=0.25,
    )
    assert np.all(routes["expansion"][[0, 1, 4]] == 1.0)
    assert routes["expansion"][2] > 0.0
    assert routes["expansion"][3] > 0.0
    assert routes["consensus"][4] == 0.0
