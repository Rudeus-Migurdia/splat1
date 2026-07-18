"""Query-time routing between a canonical score and independent hypotheses."""

import math

import torch
from torch.nn import functional as F


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


@torch.no_grad()
def blend_group_hypotheses(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
):
    """Blend the best positive group mode by its training-derived reliability."""
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Group hypothesis tensors must have matching shapes")
    if candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Group hypotheses must match the base point count")
    gains = (candidate_scores - base_scores).clamp_min(0.0)
    gates = (
        memberships.clamp(0.0, 1.0)
        * candidate_reliability.clamp(0.0, 1.0)
        * valid.to(candidate_scores.dtype)
    )
    priority = gains * gates
    best_priority, best_slot = priority.max(dim=1)
    best_gain = gains.gather(1, best_slot.unsqueeze(1))
    best_gate = gates.gather(1, best_slot.unsqueeze(1))
    routed = best_priority > 0.0
    output = base_scores + best_gate * best_gain
    count = int(routed.sum().item())
    return output, {
        "covered_points": int(valid.any(dim=1).sum().item()),
        "routed_points": count,
        "routed_fraction_all": float(routed.float().mean().item()),
        "mean_gain_routed": float(best_gain[routed].mean().item()) if count else 0.0,
        "mean_reliability_routed": float(best_gate[routed].mean().item()) if count else 0.0,
    }


@torch.no_grad()
def fuse_hierarchical_semantic_memory(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
    temperature,
    candidate_levels=None,
):
    """Fuse resident L0--L3 tokens using query similarity and split reliability.

    This readout differs from the legacy positive-gain maximum: every query gets
    a normalized distribution over the valid hierarchy levels, then interpolates
    from the base score with the selected tokens' reliability.  A level can thus
    suppress an over-confident base response when it is the more relevant memory.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Hierarchical-memory tensors must have matching shapes")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if candidate_levels is not None and candidate_levels.shape != candidate_scores.shape:
        raise ValueError("candidate_levels must match candidate scores")

    gates = (
        memberships.clamp(0.0, 1.0)
        * candidate_reliability.clamp(0.0, 1.0)
        * valid.to(candidate_scores.dtype)
    )
    selectable = valid & (gates > 0.0)
    covered = selectable.any(dim=1)
    logits = candidate_scores / temperature + gates.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    weights = torch.softmax(safe_logits, dim=1)
    weights = torch.where(covered.unsqueeze(1), weights, torch.zeros_like(weights))
    fused_scores = (weights * candidate_scores).sum(dim=1, keepdim=True)
    confidence = (weights * gates).sum(dim=1, keepdim=True)
    output = torch.where(
        covered.unsqueeze(1),
        base_scores + confidence * (fused_scores - base_scores),
        base_scores,
    )

    dominant_slot = weights.argmax(dim=1)
    dominant_weight = weights.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
    stats = {
        "covered_points": int(covered.sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "mean_dynamic_confidence": float(confidence[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_dominant_level_weight": float(dominant_weight[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta": float((output - base_scores)[covered].mean().item())
        if covered.any()
        else 0.0,
    }
    if candidate_levels is not None and covered.any():
        selected_levels = candidate_levels.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
        level_weights = {}
        dominant_counts = {}
        for level in torch.unique(candidate_levels[selectable]).tolist():
            level = int(level)
            if level < 0:
                continue
            level_mask = candidate_levels == level
            level_weights[f"level_{level}"] = float(
                weights[level_mask].mean().item()
            )
            dominant_counts[f"level_{level}"] = int(
                ((selected_levels == level) & covered).sum().item()
            )
        stats["mean_query_weight_by_level"] = level_weights
        stats["dominant_level_counts"] = dominant_counts
    return output, stats


@torch.no_grad()
def fuse_calibrated_hierarchical_memory(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
    temperature,
    candidate_levels,
    margin_threshold,
    margin_temperature,
):
    """Read four peer hierarchy slots with level calibration and a margin gate.

    Codebooks at finer levels can have a sharper raw cosine distribution simply
    because their vocabulary is larger.  We normalize scores per resident level
    before choosing a level, but retain the original cosine scores in the final
    interpolation.  Reliability measures source quality; the margin separately
    measures whether this query actually distinguishes a hierarchy level.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
        == candidate_levels.shape
    ):
        raise ValueError("Calibrated hierarchy tensors must have matching shapes")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if temperature <= 0.0 or margin_temperature <= 0.0:
        raise ValueError("Hierarchy temperatures must be positive")
    if margin_threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")

    gates = (
        memberships.clamp(0.0, 1.0)
        * candidate_reliability.clamp(0.0, 1.0)
        * valid.to(candidate_scores.dtype)
    )
    selectable = valid & (gates > 0.0)
    covered = selectable.any(dim=1)
    calibrated = torch.zeros_like(candidate_scores)
    level_stats = {}
    for level in torch.unique(candidate_levels[selectable]).tolist():
        level = int(level)
        if level < 0:
            continue
        mask = selectable & (candidate_levels == level)
        values = candidate_scores[mask]
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1e-4)
        calibrated = torch.where(mask, (candidate_scores - mean) / std, calibrated)
        level_stats[f"level_{level}"] = {
            "mean": float(mean.item()),
            "std": float(std.item()),
        }

    logits = calibrated / temperature + gates.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    weights = torch.softmax(safe_logits, dim=1)
    weights = torch.where(covered.unsqueeze(1), weights, torch.zeros_like(weights))
    fused_scores = (weights * candidate_scores).sum(dim=1, keepdim=True)
    reliability = (weights * gates).sum(dim=1, keepdim=True)

    calibrated_masked = torch.where(
        selectable, calibrated, torch.full_like(calibrated, -torch.inf)
    )
    top_values = calibrated_masked.topk(k=min(2, calibrated.shape[1]), dim=1).values
    if top_values.shape[1] == 1:
        level_margin = torch.full_like(top_values[:, 0], torch.inf)
    else:
        second = top_values[:, 1]
        level_margin = torch.where(
            torch.isfinite(second),
            top_values[:, 0] - second,
            torch.full_like(second, torch.inf),
        )
    margin_gate = torch.sigmoid(
        (level_margin - margin_threshold) / margin_temperature
    ).unsqueeze(1)
    confidence = reliability * margin_gate
    output = torch.where(
        covered.unsqueeze(1),
        base_scores + confidence * (fused_scores - base_scores),
        base_scores,
    )

    dominant_slot = weights.argmax(dim=1)
    dominant_weight = weights.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
    stats = {
        "covered_points": int(covered.sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "mean_dynamic_confidence": float(confidence[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_reliability": float(reliability[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_level_margin": float(level_margin[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_margin_gate": float(margin_gate[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_dominant_level_weight": float(dominant_weight[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta": float((output - base_scores)[covered].mean().item())
        if covered.any()
        else 0.0,
        "level_score_calibration": level_stats,
    }
    if covered.any():
        selected_levels = candidate_levels.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
        stats["dominant_level_counts"] = {
            f"level_{int(level)}": int(((selected_levels == level) & covered).sum().item())
            for level in torch.unique(candidate_levels[selectable]).tolist()
            if int(level) >= 0
        }
    return output, stats


@torch.no_grad()
def fuse_equal_query_tokens(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
    temperature,
    candidate_levels=None,
    hard=False,
):
    """Compare four peer tokens to the query and fuse only in score space.

    Every slot follows the same rule. There is no hierarchy prior, per-level
    calibration, or interpolation toward a preferred base representation. The
    base is used only when a Gaussian has no reliable resident token.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Equal-token tensors must have matching shapes")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if candidate_levels is not None and candidate_levels.shape != candidate_scores.shape:
        raise ValueError("candidate_levels must match candidate scores")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    selectable = valid & (quality > 0.0)
    covered = selectable.any(dim=1)
    logits = candidate_scores / temperature + quality.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))

    if hard:
        dominant_slot = safe_logits.argmax(dim=1)
        weights = F.one_hot(
            dominant_slot, num_classes=candidate_scores.shape[1]
        ).to(candidate_scores.dtype)
        weights = torch.where(covered.unsqueeze(1), weights, torch.zeros_like(weights))
    else:
        weights = torch.softmax(safe_logits, dim=1)
        weights = torch.where(covered.unsqueeze(1), weights, torch.zeros_like(weights))
        dominant_slot = weights.argmax(dim=1)

    fused_scores = (weights * candidate_scores).sum(dim=1, keepdim=True)
    output = torch.where(covered.unsqueeze(1), fused_scores, base_scores)
    dominant_weight = weights.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
    entropy = -(
        weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
    ).sum(dim=1)
    valid_slots = selectable.sum(dim=1).clamp_min(1)
    entropy_scale = valid_slots.float().log()
    normalized_entropy = torch.where(
        valid_slots > 1,
        entropy / entropy_scale.clamp_min(1e-8),
        torch.zeros_like(entropy),
    )
    stats = {
        "covered_points": int(covered.sum().item()),
        "fallback_points": int((~covered).sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "hard_query_retrieval": bool(hard),
        "mean_valid_slots": float(valid_slots[covered].float().mean().item())
        if covered.any()
        else 0.0,
        "mean_dominant_token_weight": float(dominant_weight[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_normalized_token_entropy": float(normalized_entropy[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta_from_fallback": float((output - base_scores)[covered].mean().item())
        if covered.any()
        else 0.0,
    }
    if candidate_levels is not None and covered.any():
        selected_levels = candidate_levels.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
        level_weights = {}
        dominant_counts = {}
        for level in torch.unique(candidate_levels[selectable]).tolist():
            level = int(level)
            if level < 0:
                continue
            level_mask = selectable & (candidate_levels == level)
            level_weights[f"level_{level}"] = float(
                weights[level_mask].mean().item()
            )
            dominant_counts[f"level_{level}"] = int(
                ((selected_levels == level) & covered).sum().item()
            )
        stats["mean_query_weight_by_level"] = level_weights
        stats["dominant_level_counts"] = dominant_counts
    return output, stats


@torch.no_grad()
def blend_contrastive_group_hypotheses(
    base_scores,
    candidate_scores,
    competitor_scores,
    memberships,
    candidate_reliability,
    valid,
    competitor_valid,
    competitor_weight=1.0,
):
    """Use neighboring semantic atoms only to cancel a candidate's positive gain."""
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == competitor_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
        == competitor_valid.shape
    ):
        raise ValueError("Contrastive group hypothesis tensors must match")
    if candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Contrastive hypotheses must match the base point count")
    if competitor_weight < 0.0:
        raise ValueError("competitor_weight must be non-negative")

    positive_gain = (candidate_scores - base_scores).clamp_min(0.0)
    competing_gain = (competitor_scores - base_scores).clamp_min(0.0)
    competing_gain = torch.where(
        competitor_valid,
        competing_gain,
        torch.zeros_like(competing_gain),
    )
    contrastive_gain = (
        positive_gain - competitor_weight * competing_gain
    ).clamp_min(0.0)
    gates = (
        memberships.clamp(0.0, 1.0)
        * candidate_reliability.clamp(0.0, 1.0)
        * valid.to(candidate_scores.dtype)
    )
    priority = contrastive_gain * gates
    best_priority, best_slot = priority.max(dim=1)
    best_gain = contrastive_gain.gather(1, best_slot.unsqueeze(1))
    best_gate = gates.gather(1, best_slot.unsqueeze(1))
    best_positive = positive_gain.gather(1, best_slot.unsqueeze(1))
    best_competing = competing_gain.gather(1, best_slot.unsqueeze(1))
    routed = best_priority > 0.0
    output = base_scores + best_gate * best_gain
    count = int(routed.sum().item())
    suppressed = (positive_gain > 0.0) & (contrastive_gain < positive_gain)
    return output, {
        "covered_points": int(valid.any(dim=1).sum().item()),
        "routed_points": count,
        "routed_fraction_all": float(routed.float().mean().item()),
        "suppressed_candidates": int(suppressed.sum().item()),
        "suppressed_fraction_valid": float(
            suppressed.sum().item() / max(int(valid.sum().item()), 1)
        ),
        "mean_positive_gain_routed": (
            float(best_positive[routed].mean().item()) if count else 0.0
        ),
        "mean_competing_gain_routed": (
            float(best_competing[routed].mean().item()) if count else 0.0
        ),
        "mean_contrastive_gain_routed": (
            float(best_gain[routed].mean().item()) if count else 0.0
        ),
        "mean_reliability_routed": (
            float(best_gate[routed].mean().item()) if count else 0.0
        ),
    }


@torch.no_grad()
def blend_dual_code_hypotheses(
    base_scores,
    semantic_scores,
    identity_scores,
    competitor_scores,
    memberships,
    candidate_reliability,
    valid,
    competitor_valid,
    use_competitor=False,
):
    """Require a shared semantic atom and an identity code to support the gain."""
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        semantic_scores.shape
        == identity_scores.shape
        == competitor_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
        == competitor_valid.shape
    ):
        raise ValueError("Dual-code hypothesis tensors must match")
    semantic_gain = (semantic_scores - base_scores).clamp_min(0.0)
    identity_gain = (identity_scores - base_scores).clamp_min(0.0)
    competing_gain = (competitor_scores - base_scores).clamp_min(0.0)
    competing_gain = torch.where(
        competitor_valid, competing_gain, torch.zeros_like(competing_gain)
    )
    identity_margin = (
        (identity_gain - competing_gain).clamp_min(0.0)
        if use_competitor
        else identity_gain
    )
    agreed_gain = torch.minimum(semantic_gain, identity_margin)
    gates = (
        memberships.clamp(0.0, 1.0)
        * candidate_reliability.clamp(0.0, 1.0)
        * valid.to(semantic_scores.dtype)
    )
    priority = agreed_gain * gates
    best_priority, best_slot = priority.max(dim=1)
    best_gain = agreed_gain.gather(1, best_slot.unsqueeze(1))
    best_gate = gates.gather(1, best_slot.unsqueeze(1))
    routed = best_priority > 0.0
    output = base_scores + best_gate * best_gain
    count = int(routed.sum().item())
    identity_positive = identity_gain > 0.0
    rejected = identity_positive & (agreed_gain < identity_gain)
    return output, {
        "covered_points": int(valid.any(dim=1).sum().item()),
        "routed_points": count,
        "routed_fraction_all": float(routed.float().mean().item()),
        "rejected_identity_candidates": int(rejected.sum().item()),
        "rejected_fraction_identity_positive": float(
            rejected.sum().item() / max(int(identity_positive.sum().item()), 1)
        ),
        "mean_agreed_gain_routed": (
            float(best_gain[routed].mean().item()) if count else 0.0
        ),
        "mean_reliability_routed": (
            float(best_gate[routed].mean().item()) if count else 0.0
        ),
        "uses_competitor": bool(use_competitor),
    }
