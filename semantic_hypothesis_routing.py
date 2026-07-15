"""Query-time routing between a canonical score and independent hypotheses."""

import math

import torch


@torch.no_grad()
def route_group_hypotheses(
    base_scores,
    candidate_scores,
    memberships,
    valid,
    route_fraction,
    priority_mode,
    query_specificity=None,
    candidate_reliability=None,
):
    """Switch a fixed budget of covered points to a better group hypothesis."""
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if candidate_scores.shape != memberships.shape or candidate_scores.shape != valid.shape:
        raise ValueError("candidate scores, memberships, and validity must match")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("candidate scores must have shape [N, K]")
    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError("route_fraction must be in [0, 1]")
    if priority_mode not in {
        "query_gain",
        "membership_gain",
        "query_margin_gain",
        "membership_margin_gain",
        "reliability_gain",
        "reliability_margin_gain",
    }:
        raise ValueError(f"Unknown routing priority: {priority_mode}")
    if "margin" in priority_mode:
        if query_specificity is None or query_specificity.shape != candidate_scores.shape:
            raise ValueError("Margin routing requires matching query_specificity")
    if "reliability" in priority_mode:
        if candidate_reliability is None or candidate_reliability.shape != candidate_scores.shape:
            raise ValueError("Reliability routing requires matching candidate_reliability")

    gains = (candidate_scores - base_scores).clamp_min(0.0)
    priority = gains
    if "membership" in priority_mode:
        priority = priority * memberships.clamp(0.0, 1.0)
    if "reliability" in priority_mode:
        priority = priority * candidate_reliability.clamp(0.0, 1.0)
    if "margin" in priority_mode:
        priority = priority * query_specificity.clamp_min(0.0)
    priority = torch.where(valid, priority, torch.zeros_like(priority))

    best_priority, best_slot = priority.max(dim=1)
    best_scores = candidate_scores.gather(1, best_slot.unsqueeze(1))
    best_gains = gains.gather(1, best_slot.unsqueeze(1)).squeeze(1)
    best_memberships = memberships.gather(1, best_slot.unsqueeze(1)).squeeze(1)
    best_specificity = (
        query_specificity.gather(1, best_slot.unsqueeze(1)).squeeze(1)
        if query_specificity is not None
        else torch.zeros_like(best_gains)
    )
    best_reliability = (
        candidate_reliability.gather(1, best_slot.unsqueeze(1)).squeeze(1)
        if candidate_reliability is not None
        else torch.zeros_like(best_gains)
    )
    covered = valid.any(dim=1)
    eligible = covered & (best_priority > 0.0)
    covered_count = int(covered.sum().item())
    budget = min(
        int(eligible.sum().item()),
        int(math.ceil(route_fraction * covered_count)),
    )
    routed = torch.zeros_like(covered)
    if budget > 0:
        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        selected = torch.topk(
            best_priority[eligible_indices],
            k=budget,
            largest=True,
            sorted=False,
        ).indices
        routed[eligible_indices[selected]] = True

    output = torch.where(routed.unsqueeze(1), best_scores, base_scores)
    routed_count = int(routed.sum().item())
    stats = {
        "covered_points": covered_count,
        "eligible_positive_points": int(eligible.sum().item()),
        "route_budget_points": int(math.ceil(route_fraction * covered_count)),
        "routed_points": routed_count,
        "routed_fraction_all": float(routed.float().mean().item()),
        "routed_fraction_covered": float(routed_count / max(covered_count, 1)),
        "mean_gain_routed": float(best_gains[routed].mean().item()) if routed_count else 0.0,
        "mean_membership_routed": (
            float(best_memberships[routed].mean().item()) if routed_count else 0.0
        ),
        "mean_query_specificity_routed": (
            float(best_specificity[routed].mean().item()) if routed_count else 0.0
        ),
        "mean_reliability_routed": (
            float(best_reliability[routed].mean().item()) if routed_count else 0.0
        ),
    }
    return output, stats


@torch.no_grad()
def blend_sparse_hypothesis(
    base_scores,
    candidate_scores,
    reliability,
    valid,
    query_specificity=None,
):
    """Reliability-weight a positive score-space hypothesis without changing the base."""
    if not (
        base_scores.shape
        == candidate_scores.shape
        == reliability.shape
        == valid.shape
    ):
        raise ValueError("Sparse hypothesis tensors must have matching shapes")
    gain = (candidate_scores - base_scores).clamp_min(0.0)
    gate = reliability.clamp(0.0, 1.0) * valid.to(reliability.dtype)
    if query_specificity is not None:
        if query_specificity.shape != base_scores.shape:
            raise ValueError("query_specificity must match sparse hypothesis scores")
        gate = gate * (query_specificity > 0.0).to(gate.dtype)
    routed = (gain > 0.0) & (gate > 0.0)
    output = base_scores + gate * gain
    count = int(routed.sum().item())
    stats = {
        "covered_points": int(valid.sum().item()),
        "routed_points": count,
        "routed_fraction_all": float(routed.float().mean().item()),
        "routed_fraction_covered": float(count / max(int(valid.sum().item()), 1)),
        "mean_gain_routed": float(gain[routed].mean().item()) if count else 0.0,
        "mean_reliability_routed": (
            float(reliability[routed].mean().item()) if count else 0.0
        ),
    }
    return output, stats
