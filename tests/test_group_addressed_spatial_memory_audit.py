from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from build_group_addressed_spatial_memory_audit import (
    bounded_group_profiles,
    contrastive_group_scores,
    make_gate,
    signed_group_profiles,
)


def test_bounded_profiles_drop_unanchored_mask_fragments():
    profiles = np.asarray([[0.9, 0.4, 0.0, 0.3]], dtype=np.float32)
    graph = csr_matrix(
        np.asarray(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.float32,
        )
    )
    args = SimpleNamespace(
        minimum_atom_contact=0.05,
        core_coverage_threshold=0.65,
        boundary_coverage_threshold=0.20,
        minimum_core_atoms=1,
    )
    bounded, core, support, _, stats = bounded_group_profiles(profiles, graph, args)
    np.testing.assert_allclose(bounded, [[0.9, 0.4, 0.0, 0.0]])
    assert core.tolist() == [[True, False, False, False]]
    assert support.tolist() == [[True, True, False, False]]
    assert stats["rejected_unanchored_components"] == 1


def test_ring_contrast_penalizes_group_whose_exterior_matches_query():
    model = {
        "descriptors": np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32),
        "ring_descriptors": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "ring_valid": np.asarray([True, True]),
    }
    args = SimpleNamespace(ring_contrast_margin=0.05, ring_contrast_weight=0.5)
    adjusted, core, ring = contrastive_group_scores(np.asarray([1.0, 0.0]), model, args)
    assert core[0] > core[1]
    assert adjusted[1] > adjusted[0]
    assert ring[0] > ring[1]


def test_signed_group_profile_suppresses_exterior_responsibility():
    model = {
        "profiles": np.asarray([[0.8, 0.8]], dtype=np.float32),
        "members": [[{"view_index": 0, "proposal_index": 0}]],
    }
    views = [
        {
            "view_index": 0,
            "quality": np.asarray([1.0]),
            "visibility": np.asarray([1.0, 1.0]),
            "coverage": np.asarray([[0.8, 0.8]], dtype=np.float32),
        }
    ]
    ring_views = {0: np.asarray([[0.0, 0.8]], dtype=np.float32)}
    args = SimpleNamespace(exterior_evidence_weight=1.0, signed_evidence_epsilon=1e-6)
    signed, stats = signed_group_profiles(model, views, ring_views, args)
    assert signed[0, 0] > 0.79
    assert 0.39 < signed[0, 1] < 0.41
    assert stats["groups_without_members"] == 0


def test_gate_requires_spatial_and_semantic_checks():
    args = SimpleNamespace(
        minimum_spill_reduction=0.25,
        minimum_recall_retention=0.85,
        maximum_nll_regression=0.02,
        minimum_ring_contrast_nll_improvement=0.0,
        minimum_split_stability=0.65,
        minimum_stable_groups=100,
        minimum_group_count_agreement=0.8,
    )
    metrics = {
        "unique_group_address_contract": True,
        "bounded_spill_reduction": 0.3,
        "bounded_recall_retention": 0.9,
        "bounded_relative_nll_regression": 0.01,
        "ring_contrast_relative_nll_improvement": 0.01,
        "median_matched_jaccard": 0.7,
        "stable_groups": 100,
        "group_count_agreement": 0.9,
        "capacity_saturated": False,
        "no_evaluation_queries_labels_or_codebooks": True,
        "unresolved_certificate_written": True,
    }
    assert make_gate(metrics, args)["pass"]
    metrics["bounded_recall_retention"] = 0.5
    assert not make_gate(metrics, args)["pass"]
