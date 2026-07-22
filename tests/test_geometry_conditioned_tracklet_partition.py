from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from build_geometry_conditioned_tracklet_partition import (
    aggregate_gaussian_geometry,
    geometry_conditioned_partition,
    make_gate,
)


def test_atom_geometry_preserves_centroids_and_unit_normals():
    geometry = aggregate_gaussian_geometry(
        xyz=np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        log_scaling=np.zeros((2, 3), dtype=np.float32),
        quaternion=np.asarray([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
        raw_opacity=np.zeros((2, 1), dtype=np.float32),
        atom_ids=np.asarray([0, 0]),
    )
    np.testing.assert_allclose(geometry["centroid"], [[1, 0, 0]])
    np.testing.assert_allclose(np.linalg.norm(geometry["normal"], axis=1), [1])


def test_partition_merges_supported_same_level_tracklets_only():
    model = {
        "profiles": np.asarray([[0.9, 0.8, 0], [0.8, 0.9, 0], [0.9, 0.8, 0]], dtype=np.float32),
        "descriptors": np.asarray([[1, 0], [1, 0], [1, 0]], dtype=np.float32),
        "support_views": np.asarray([3, 4, 3]),
        "levels": np.asarray([0, 0, 1]),
        "utility": np.ones(3, dtype=np.float32),
    }
    atom_geometry = {
        "centroid": np.asarray([[0, 0, 0], [0.1, 0, 0], [5, 0, 0]], dtype=np.float32),
        "normal": np.asarray([[0, 0, 1]] * 3, dtype=np.float32),
        "radius": np.ones(3, dtype=np.float32),
        "opacity": np.ones(3, dtype=np.float32),
    }
    graph = csr_matrix(np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    args = SimpleNamespace(
        coverage_threshold=0.3,
        partition_minimum_semantic_cosine=0.85,
        partition_minimum_geometry_continuity=0.4,
        partition_minimum_boundary_support=0.2,
        partition_semantic_cost_weight=0.45,
        partition_boundary_cost_weight=0.35,
        partition_geometry_cost_weight=0.2,
        partition_entity_count_penalty=0.42,
    )
    result = geometry_conditioned_partition(model, atom_geometry, graph, args)
    assert result["statistics"]["partition_entities"] == 2
    assert sorted(result["levels"].tolist()) == [0, 1]


def test_gate_requires_all_mechanism_checks():
    args = SimpleNamespace(
        minimum_nll_improvement_over_a48=0.0,
        minimum_nll_improvement_over_uncapped=0.0,
        minimum_split_stability=0.8,
        minimum_stable_entities=53,
        minimum_entity_count_agreement=0.8,
        maximum_union_mass_fraction=0.5,
    )
    metrics = {
        "relative_nll_improvement_over_a48": 0.01,
        "relative_nll_improvement_over_uncapped": 0.01,
        "median_matched_jaccard": 0.81,
        "stable_entities": 53,
        "entity_count_agreement": 0.9,
        "capacity_saturated": False,
        "mdl_union_mass_fraction": 0.4,
        "unresolved_certificate_written": True,
        "no_queries_labels_or_codebooks": True,
    }
    assert make_gate(metrics, args)["pass"]
    metrics["median_matched_jaccard"] = 0.7
    assert not make_gate(metrics, args)["pass"]
