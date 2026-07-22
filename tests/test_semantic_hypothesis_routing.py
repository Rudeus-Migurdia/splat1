import pytest

torch = pytest.importorskip("torch")

from semantic_hypothesis_routing import (  # noqa: E402
    blend_contrastive_group_hypotheses,
    blend_dual_code_hypotheses,
    blend_group_hypotheses,
    blend_sparse_hypothesis,
    fuse_calibrated_equal_query_tokens,
    fuse_calibrated_hierarchical_memory,
    fuse_equal_query_tokens,
    fuse_hierarchical_semantic_memory,
    fuse_information_gain_equal_query_tokens,
    fuse_multiscale_set_relation_token_scores,
    fuse_quantization_aware_equal_query_tokens,
    fuse_signed_relation_graph_scores,
    route_group_hypotheses,
)


def test_contrastive_group_only_cancels_added_gain():
    base = torch.tensor([[0.4], [0.4], [0.4]])
    candidate = torch.tensor([[0.8], [0.8], [0.3]])
    competitor = torch.tensor([[0.7], [0.2], [0.9]])
    membership = torch.ones_like(candidate)
    reliability = torch.ones_like(candidate)
    valid = torch.ones_like(candidate, dtype=torch.bool)
    competitor_valid = torch.tensor([[True], [True], [True]])
    output, stats = blend_contrastive_group_hypotheses(
        base,
        candidate,
        competitor,
        membership,
        reliability,
        valid,
        competitor_valid,
    )
    assert torch.allclose(output, torch.tensor([[0.5], [0.8], [0.4]]))
    assert torch.all(output >= base)
    assert stats["suppressed_candidates"] == 1


def test_dual_code_requires_semantic_and_identity_agreement():
    base = torch.tensor([[0.4], [0.4], [0.4]])
    semantic = torch.tensor([[0.7], [0.3], [0.8]])
    identity = torch.tensor([[0.8], [0.8], [0.8]])
    competitor = torch.tensor([[0.2], [0.2], [0.7]])
    ones = torch.ones_like(identity)
    valid = torch.ones_like(identity, dtype=torch.bool)
    output, _ = blend_dual_code_hypotheses(
        base, semantic, identity, competitor, ones, ones, valid, valid, True
    )
    assert torch.allclose(output, torch.tensor([[0.7], [0.4], [0.5]]))
    assert torch.all(output >= base)


def test_group_blend_uses_best_reliability_weighted_positive_gain():
    base = torch.tensor([[0.2], [0.4]])
    candidate = torch.tensor([[0.8, 0.7], [0.3, 0.9]])
    membership = torch.ones_like(candidate)
    reliability = torch.tensor([[0.25, 0.75], [1.0, 0.2]])
    valid = torch.ones_like(candidate, dtype=torch.bool)

    output, stats = blend_group_hypotheses(
        base, candidate, membership, reliability, valid
    )

    assert output[:, 0].tolist() == pytest.approx([0.575, 0.5])
    assert stats["routed_points"] == 2


def test_query_gain_routes_only_positive_hypotheses_with_fixed_budget():
    base = torch.tensor([[0.2], [0.6], [0.1], [0.4]])
    candidates = torch.tensor(
        [[0.5, 0.3], [0.5, 0.4], [0.2, 0.8], [0.9, 0.1]]
    )
    memberships = torch.ones_like(candidates)
    valid = torch.tensor(
        [[True, True], [True, True], [True, True], [True, True]]
    )

    output, stats = route_group_hypotheses(
        base, candidates, memberships, valid, 0.25, "query_gain"
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.2, 0.6, 0.8, 0.4])
    assert stats["routed_points"] == 1
    assert stats["routed_fraction_covered"] == pytest.approx(0.25)


def test_membership_gain_can_select_a_more_reliable_candidate():
    base = torch.tensor([[0.1]])
    candidates = torch.tensor([[0.9, 0.7]])
    memberships = torch.tensor([[0.1, 1.0]])
    valid = torch.tensor([[True, True]])

    output, stats = route_group_hypotheses(
        base, candidates, memberships, valid, 1.0, "membership_gain"
    )

    assert output.item() == pytest.approx(0.7)
    assert stats["mean_membership_routed"] == pytest.approx(1.0)


def test_invalid_candidates_and_zero_budget_fall_back_to_base():
    base = torch.tensor([[0.2], [0.4]])
    candidates = torch.tensor([[0.9], [0.8]])
    memberships = torch.ones_like(candidates)
    valid = torch.tensor([[False], [True]])

    output, stats = route_group_hypotheses(
        base, candidates, memberships, valid, 0.0, "query_gain"
    )

    assert torch.equal(output, base)
    assert stats["covered_points"] == 1
    assert stats["routed_points"] == 0


def test_routing_validates_shapes_and_fraction():
    base = torch.zeros((2, 1))
    candidates = torch.zeros((2, 2))
    memberships = torch.zeros((2, 2))
    valid = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="route_fraction"):
        route_group_hypotheses(
            base, candidates, memberships, valid, 1.1, "query_gain"
        )
    with pytest.raises(ValueError, match="must match"):
        route_group_hypotheses(
            base, candidates, memberships[:, :1], valid, 0.5, "query_gain"
        )


def test_query_margin_rejects_a_candidate_owned_by_another_query():
    base = torch.tensor([[0.1], [0.1]])
    candidates = torch.tensor([[0.8], [0.7]])
    memberships = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    specificity = torch.tensor([[0.0], [0.2]])

    output, stats = route_group_hypotheses(
        base,
        candidates,
        memberships,
        valid,
        1.0,
        "membership_margin_gain",
        specificity,
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.1, 0.7])
    assert stats["routed_points"] == 1
    assert stats["mean_query_specificity_routed"] == pytest.approx(0.2)


def test_full_reliability_ranks_stable_candidate_ahead_of_raw_gain():
    base = torch.tensor([[0.1]])
    candidates = torch.tensor([[0.9, 0.7]])
    memberships = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    reliability = torch.tensor([[0.1, 0.9]])
    output, stats = route_group_hypotheses(
        base,
        candidates,
        memberships,
        valid,
        1.0,
        "reliability_gain",
        candidate_reliability=reliability,
    )
    assert output.item() == pytest.approx(0.7)
    assert stats["mean_reliability_routed"] == pytest.approx(0.9)


def test_sparse_blend_preserves_base_and_scales_only_positive_gain():
    base = torch.tensor([[0.2], [0.8], [0.1]])
    candidate = torch.tensor([[0.6], [0.4], [0.9]])
    reliability = torch.tensor([[0.5], [1.0], [0.25]])
    valid = torch.tensor([[True], [True], [False]])
    output, stats = blend_sparse_hypothesis(
        base, candidate, reliability, valid
    )
    assert output.squeeze(1).tolist() == pytest.approx([0.4, 0.8, 0.1])
    assert stats["routed_points"] == 1


def test_hierarchical_memory_reads_a_different_level_for_each_query():
    base = torch.tensor([[0.4], [0.4]])
    candidates = torch.tensor([[0.8, 0.5], [0.2, 0.9]])
    memberships = torch.ones_like(candidates)
    reliability = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    levels = torch.tensor([[0, 3], [0, 3]])

    output, stats = fuse_hierarchical_semantic_memory(
        base, candidates, memberships, reliability, valid, 0.05, levels
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.8, 0.9], abs=5e-3)
    assert stats["dominant_level_counts"] == {"level_0": 1, "level_3": 1}


def test_hierarchical_memory_reliability_interpolates_from_the_base():
    base = torch.tensor([[0.4]])
    candidates = torch.tensor([[0.8, 0.1]])
    memberships = torch.ones_like(candidates)
    reliability = torch.tensor([[0.25, 0.0]])
    valid = torch.ones_like(candidates, dtype=torch.bool)

    output, stats = fuse_hierarchical_semantic_memory(
        base, candidates, memberships, reliability, valid, 0.1
    )

    assert output.item() == pytest.approx(0.5)
    assert stats["mean_dynamic_confidence"] == pytest.approx(0.25)


def test_calibrated_hierarchy_keeps_base_when_levels_are_indistinguishable():
    base = torch.tensor([[0.4]])
    candidates = torch.tensor([[0.8, 0.79]])
    ones = torch.ones_like(candidates)
    levels = torch.tensor([[0, 3]])

    output, stats = fuse_calibrated_hierarchical_memory(
        base, candidates, ones, ones, torch.ones_like(candidates, dtype=torch.bool),
        0.10, levels, 0.25, 0.05,
    )

    assert output.item() == pytest.approx(0.4, abs=3e-3)
    assert stats["mean_margin_gate"] < 0.01


def test_calibrated_hierarchy_can_select_a_distinct_peer_level():
    base = torch.tensor([[0.4], [0.4]])
    candidates = torch.tensor([[0.8, 0.5], [0.2, 0.5]])
    ones = torch.ones_like(candidates)
    levels = torch.tensor([[0, 3], [0, 3]])

    output, stats = fuse_calibrated_hierarchical_memory(
        base, candidates, ones, ones, torch.ones_like(candidates, dtype=torch.bool),
        0.10, levels, 0.25, 0.05,
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.8, 0.5], abs=5e-3)
    assert stats["dominant_level_counts"] == {"level_0": 1, "level_3": 1}


def test_equal_query_tokens_are_invariant_to_slot_permutation():
    base = torch.tensor([[0.1], [0.2]])
    candidates = torch.tensor([[0.8, 0.4, 0.2, 0.1], [0.1, 0.3, 0.9, 0.2]])
    memberships = torch.ones_like(candidates)
    reliability = torch.tensor([[0.8, 0.6, 0.9, 0.7], [0.7, 0.8, 0.6, 0.9]])
    valid = torch.ones_like(candidates, dtype=torch.bool)
    levels = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    output, _ = fuse_equal_query_tokens(
        base, candidates, memberships, reliability, valid, 0.05, levels
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _ = fuse_equal_query_tokens(
        base,
        candidates[:, permutation],
        memberships[:, permutation],
        reliability[:, permutation],
        valid[:, permutation],
        0.05,
        levels[:, permutation],
    )
    assert torch.allclose(output, permuted)
    assert output.squeeze(1).tolist() == pytest.approx([0.8, 0.9], abs=5e-3)


def test_equal_query_tokens_use_base_only_without_a_reliable_slot():
    base = torch.tensor([[0.4], [0.6]])
    candidates = torch.tensor([[0.9, 0.2, 0.1, 0.0], [0.9, 0.8, 0.7, 0.6]])
    memberships = torch.ones_like(candidates)
    reliability = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.ones_like(candidates, dtype=torch.bool)
    output, stats = fuse_equal_query_tokens(
        base, candidates, memberships, reliability, valid, 0.05
    )
    assert output[0].item() == pytest.approx(0.9, abs=5e-3)
    assert output[1].item() == pytest.approx(0.6)
    assert stats["fallback_points"] == 1


def test_equal_query_max_returns_the_best_reliability_adjusted_token():
    base = torch.tensor([[0.2]])
    candidates = torch.tensor([[0.80, 0.79, 0.1, 0.0]])
    memberships = torch.ones_like(candidates)
    reliability = torch.tensor([[0.01, 1.0, 1.0, 1.0]])
    valid = torch.ones_like(candidates, dtype=torch.bool)
    output, stats = fuse_equal_query_tokens(
        base, candidates, memberships, reliability, valid, 0.05, hard=True
    )
    assert output.item() == pytest.approx(0.79)
    assert stats["hard_query_retrieval"] is True


def test_equal_query_margin_top2_blends_only_near_ties():
    base = torch.tensor([[0.2], [0.2]])
    candidates = torch.tensor(
        [[0.80, 0.795, 0.1, 0.0], [0.80, 0.70, 0.1, 0.0]]
    )
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)

    output, stats = fuse_equal_query_tokens(
        base,
        candidates,
        ones,
        ones,
        valid,
        0.05,
        hard=True,
        tie_margin=0.01,
    )

    expected_tie = torch.softmax(torch.tensor([0.80, 0.795]) / 0.05, dim=0)
    expected_score = float((expected_tie * torch.tensor([0.80, 0.795])).sum())
    assert output.squeeze(1).tolist() == pytest.approx([expected_score, 0.80])
    assert stats["margin_aware_top2"] is True
    assert stats["tie_blended_points"] == 1
    assert stats["tie_blended_fraction_covered"] == pytest.approx(0.5)


def test_equal_query_margin_top2_is_invariant_to_slot_permutation():
    base = torch.tensor([[0.1]])
    candidates = torch.tensor([[0.61, 0.60, 0.59, 0.2]])
    reliability = torch.tensor([[0.9, 0.7, 1.0, 0.8]])
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    output, _ = fuse_equal_query_tokens(
        base,
        candidates,
        ones,
        reliability,
        valid,
        0.05,
        hard=True,
        tie_margin=0.02,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _ = fuse_equal_query_tokens(
        base,
        candidates[:, permutation],
        ones[:, permutation],
        reliability[:, permutation],
        valid[:, permutation],
        0.05,
        hard=True,
        tie_margin=0.02,
    )
    assert torch.allclose(output, permuted)


def test_signed_relation_graph_smooths_positive_and_sharpens_negative_edges():
    scores = torch.tensor([[0.9], [0.1]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.int32)
    positive = torch.ones((2, 1))
    smoothed, smooth_stats = fuse_signed_relation_graph_scores(
        scores, neighbors, positive, 0.20, 0.10, 1.0
    )
    assert smoothed.squeeze(1).tolist() == pytest.approx([0.74, 0.26])
    assert smooth_stats["positive_edge_slots"] == 2

    sharpened, sharp_stats = fuse_signed_relation_graph_scores(
        scores, neighbors, -positive, 0.20, 0.10, 1.0
    )
    assert sharpened.squeeze(1).tolist() == pytest.approx([0.98, 0.02])
    assert sharp_stats["negative_edge_slots"] == 2


def test_multiscale_set_relations_update_peer_tokens_before_retrieval():
    scores = torch.tensor([[0.9, 0.2], [0.1, 0.8]])
    levels = torch.tensor([[0, 1], [0, 1]])
    selectable = torch.ones_like(scores, dtype=torch.bool)
    neighbors = torch.tensor([[1], [0]], dtype=torch.int32)
    signatures = torch.tensor(
        [[[1.0, -1.0]], [[1.0, -1.0]]], dtype=torch.float32
    )

    output, stats = fuse_multiscale_set_relation_token_scores(
        scores,
        levels,
        selectable,
        neighbors,
        signatures,
        positive_strength=0.20,
        negative_strength=0.10,
        maximum_delta=1.0,
    )

    assert torch.allclose(
        output, torch.tensor([[0.74, 0.14], [0.26, 0.86]]), atol=1e-6
    )
    assert stats["corrected_token_slots"] == 4


def test_signed_relation_graph_is_bounded_and_uses_relation_confidence():
    scores = torch.tensor([[0.9], [0.1]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.int64)
    weak_positive = torch.full((2, 1), 0.5)
    output, stats = fuse_signed_relation_graph_scores(
        scores, neighbors, weak_positive, 1.0, 0.0, 0.05
    )
    assert output.squeeze(1).tolist() == pytest.approx([0.85, 0.15])
    assert stats["clipped_points"] == 2


def test_quantization_interval_changes_only_an_ambiguous_high_error_winner():
    base = torch.tensor([[0.2], [0.2], [0.3]])
    candidates = torch.tensor(
        [[0.80, 0.79], [0.80, 0.70], [0.90, 0.85]]
    )
    errors = torch.tensor([[0.50, 0.05], [0.10, 0.10], [0.10, 0.10]])
    ones = torch.ones_like(candidates)
    valid = torch.tensor([[True, True], [True, True], [False, False]])

    output, stats = fuse_quantization_aware_equal_query_tokens(
        base,
        candidates,
        ones,
        ones,
        errors,
        valid,
        temperature=0.05,
        uncertainty_scale=0.10,
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.79, 0.80, 0.30])
    assert stats["interval_dominant_points"] == 1
    assert stats["ambiguous_points"] == 1
    assert stats["selection_changed_points"] == 1


def test_quantization_interval_is_peer_permutation_invariant():
    base = torch.tensor([[0.1]])
    candidates = torch.tensor([[0.81, 0.80, 0.79, 0.2]])
    reliability = torch.tensor([[0.9, 0.7, 1.0, 0.8]])
    errors = torch.tensor([[0.4, 0.1, 0.05, 0.2]])
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    output, _ = fuse_quantization_aware_equal_query_tokens(
        base,
        candidates,
        ones,
        reliability,
        errors,
        valid,
        0.05,
        0.05,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _ = fuse_quantization_aware_equal_query_tokens(
        base,
        candidates[:, permutation],
        ones[:, permutation],
        reliability[:, permutation],
        errors[:, permutation],
        valid[:, permutation],
        0.05,
        0.05,
    )
    assert torch.allclose(output, permuted)


def test_equal_query_percentile_calibration_changes_only_the_winning_slot():
    base = torch.tensor([[0.2]])
    candidates = torch.tensor([[0.81, 0.78, 0.1, 0.0]])
    percentiles = torch.tensor([[0.70, 0.99, 0.1, 0.0]])
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)

    output, stats = fuse_calibrated_equal_query_tokens(
        base,
        candidates,
        percentiles,
        ones,
        ones,
        valid,
        "percentile",
        0.05,
        torch.tensor([[0, 1, 2, 3]]),
    )

    assert output.item() == pytest.approx(0.78)
    assert stats["level_score_calibration"] == "percentile"
    assert stats["dominant_level_counts"] == {"level_0": 0, "level_1": 1, "level_2": 0, "level_3": 0}


def test_equal_query_tail_evidence_keeps_reliability_in_the_decision():
    base = torch.tensor([[0.2], [0.3]])
    candidates = torch.tensor([[0.82, 0.79], [0.75, 0.70]])
    evidence = torch.tensor([[4.0, 4.4], [3.0, 2.0]])
    memberships = torch.ones_like(candidates)
    reliability = torch.tensor([[1.0, 0.5], [0.1, 1.0]])
    valid = torch.tensor([[True, True], [False, False]])

    output, stats = fuse_calibrated_equal_query_tokens(
        base,
        candidates,
        evidence,
        memberships,
        reliability,
        valid,
        "tail_evidence",
        0.05,
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.82, 0.3])
    assert stats["covered_points"] == 1
    assert stats["fallback_points"] == 1


def test_information_gain_can_select_a_more_distinctive_lower_raw_score():
    base = torch.tensor([[0.2]])
    candidates = torch.tensor([[0.86, 0.81, 0.4, 0.3]])
    information_gain = torch.tensor([[0.1, 1.2, -0.4, -0.8]])
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)

    output, stats = fuse_information_gain_equal_query_tokens(
        base,
        candidates,
        information_gain,
        ones,
        ones,
        valid,
        torch.tensor([[0, 1, 2, 3]]),
    )

    assert output.item() == pytest.approx(0.81)
    assert stats["local_counterfactual_penalty"] is False
    assert stats["dominant_level_counts"] == {
        "level_0": 0,
        "level_1": 1,
        "level_2": 0,
        "level_3": 0,
    }


def test_counterfactual_gain_only_penalizes_ambiguous_tokens():
    base = torch.tensor([[0.2], [0.3]])
    candidates = torch.tensor([[0.84, 0.80], [0.75, 0.70]])
    information_gain = torch.tensor([[1.0, 0.8], [0.6, 0.4]])
    local_gain = torch.tensor([[-0.5, 0.2], [0.3, -0.2]])
    ones = torch.ones_like(candidates)
    valid = torch.tensor([[True, True], [False, False]])

    output, stats = fuse_information_gain_equal_query_tokens(
        base,
        candidates,
        information_gain,
        ones,
        ones,
        valid,
        local_counterfactual_gain=local_gain,
    )

    assert output.squeeze(1).tolist() == pytest.approx([0.80, 0.3])
    assert stats["counterfactual_nonpositive_points"] == 0
    assert stats["fallback_points"] == 1


def test_information_gain_routing_is_invariant_to_slot_permutation():
    base = torch.tensor([[0.2], [0.4]])
    candidates = torch.tensor([[0.8, 0.7, 0.6], [0.5, 0.9, 0.3]])
    gain = torch.tensor([[0.2, 0.9, 0.1], [0.7, 0.6, 0.2]])
    local = torch.tensor([[0.1, -0.1, -0.2], [-0.2, 0.1, 0.0]])
    reliability = torch.tensor([[1.0, 0.8, 1.0], [1.0, 1.0, 0.5]])
    ones = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    output, _ = fuse_information_gain_equal_query_tokens(
        base,
        candidates,
        gain,
        ones,
        reliability,
        valid,
        local_counterfactual_gain=local,
    )
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = fuse_information_gain_equal_query_tokens(
        base,
        candidates[:, permutation],
        gain[:, permutation],
        ones[:, permutation],
        reliability[:, permutation],
        valid[:, permutation],
        local_counterfactual_gain=local[:, permutation],
    )

    assert torch.allclose(output, permuted)
