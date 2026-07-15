import pytest

torch = pytest.importorskip("torch")

from semantic_hypothesis_routing import blend_sparse_hypothesis, route_group_hypotheses


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
