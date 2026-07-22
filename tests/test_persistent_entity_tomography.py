from types import SimpleNamespace

import numpy as np

from build_persistent_entity_tomography import (
    ViewExclusiveUnionFind,
    make_gate,
    profile_jaccard,
    reciprocal_match_edges,
)


def test_view_exclusive_union_rejects_converging_tracks():
    union_find = ViewExclusiveUnionFind([0, 1, 0])
    assert union_find.union(0, 1)
    assert not union_find.union(1, 2)
    assert union_find.rejected_view_conflicts == 1


def test_reciprocal_edges_preserve_levels():
    first = {
        "coverage": np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "descriptors": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        "levels": np.asarray([0, 1]),
    }
    second = {
        "coverage": np.asarray([[0.9, 0, 0], [0, 0.9, 0]], dtype=np.float32),
        "descriptors": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        "levels": np.asarray([0, 1]),
    }
    edges = reciprocal_match_edges(first, second, 0.3, 0.35, 0.75, 0.85, 0.4)
    assert {(edge[1], edge[2]) for edge in edges} == {(0, 0), (1, 1)}


def test_profile_jaccard_uses_binary_ownership_support():
    assert profile_jaccard([0.9, 0.6, 0.1], [0.8, 0.2, 0.1]) == 0.5


def test_gate_rejects_union_saturation():
    args = SimpleNamespace(
        minimum_nll_improvement=0.1,
        minimum_split_stability=0.8,
        minimum_stable_slots=8,
        minimum_persistence_views=3,
        minimum_slot_count_agreement=0.8,
        minimum_union_mass_fraction=0.01,
        maximum_union_mass_fraction=0.5,
    )
    metrics = {
        "relative_nll_improvement": 0.2,
        "median_matched_jaccard": 0.85,
        "stable_slots": 10,
        "minimum_slot_support_views": 3,
        "slot_count_agreement": 0.9,
        "capacity_saturated": False,
        "mdl_union_mass_fraction": 0.2,
        "unresolved_certificate_written": True,
    }
    assert make_gate(metrics, args)["pass"]
    metrics["mdl_union_mass_fraction"] = 0.8
    assert not make_gate(metrics, args)["pass"]
