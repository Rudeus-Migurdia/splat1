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
    tie_margin=0.0,
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
    if tie_margin < 0.0:
        raise ValueError("tie_margin must be non-negative")
    if tie_margin > 0.0 and not hard:
        raise ValueError("tie_margin is only supported by hard retrieval")

    quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    selectable = valid & (quality > 0.0)
    covered = selectable.any(dim=1)
    logits = candidate_scores / temperature + quality.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))

    if hard:
        if tie_margin == 0.0:
            dominant_slot = safe_logits.argmax(dim=1)
            top_count = 0
        else:
            top_count = min(2, candidate_scores.shape[1])
            top_logits, top_slots = safe_logits.topk(top_count, dim=1)
            dominant_slot = top_slots[:, 0]
        hard_weights = F.one_hot(
            dominant_slot, num_classes=candidate_scores.shape[1]
        ).to(candidate_scores.dtype)
        hard_weights = torch.where(
            covered.unsqueeze(1), hard_weights, torch.zeros_like(hard_weights)
        )
        weights = hard_weights
        tie_blended = torch.zeros_like(covered)
        top2_margin = torch.zeros_like(dominant_slot, dtype=candidate_scores.dtype)
        if top_count == 2:
            has_two_slots = selectable.sum(dim=1) >= 2
            top2_margin = (top_logits[:, 0] - top_logits[:, 1]) * temperature
            tie_blended = covered & has_two_slots & (top2_margin <= tie_margin)
            pair_weights = torch.softmax(top_logits, dim=1)
            blended_weights = torch.zeros_like(candidate_scores).scatter(
                1, top_slots, pair_weights
            )
            weights = torch.where(
                tie_blended.unsqueeze(1), blended_weights, hard_weights
            )
    else:
        weights = torch.softmax(safe_logits, dim=1)
        weights = torch.where(covered.unsqueeze(1), weights, torch.zeros_like(weights))
        dominant_slot = weights.argmax(dim=1)
        tie_blended = torch.zeros_like(covered)
        top2_margin = torch.zeros_like(dominant_slot, dtype=candidate_scores.dtype)

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
        "hard_query_retrieval": bool(hard and tie_margin == 0.0),
        "margin_aware_top2": bool(hard and tie_margin > 0.0),
        "tie_margin": float(tie_margin),
        "tie_blended_points": int(tie_blended.sum().item()),
        "tie_blended_fraction_covered": float(
            tie_blended.sum().float().div(covered.sum().clamp_min(1)).item()
        ),
        "mean_top2_adjusted_margin": float(
            top2_margin[(selectable.sum(dim=1) >= 2) & covered].mean().item()
        )
        if ((selectable.sum(dim=1) >= 2) & covered).any()
        else 0.0,
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
def fuse_query_conditioned_spatial_posterior(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
    semantic_temperature,
    spatial_core_scores,
    spatial_ring_scores,
    spatial_memberships,
    spatial_reliability,
    spatial_entropy,
    spatial_valid,
    ring_weight,
    contrast_temperature,
    maximum_penalty,
    core_membership,
    entropy_relaxation,
    gaussian_atom_ids=None,
    atom_neighbor_ids=None,
    geodesic_delta=0.05,
    recovery_factor=0.2,
):
    """Retrieve four semantic tokens first, then apply one spatial posterior."""
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Semantic candidate tensors must match")
    if spatial_core_scores.shape != spatial_memberships.shape:
        raise ValueError("Spatial scores and memberships must match")
    if spatial_core_scores.shape != spatial_ring_scores.shape:
        raise ValueError("Core and ring scores must match")
    if spatial_core_scores.shape != spatial_reliability.shape:
        raise ValueError("Spatial reliability must match spatial scores")
    if spatial_valid.shape != spatial_core_scores.shape:
        raise ValueError("Spatial validity must match spatial scores")
    if spatial_core_scores.shape[:2] != candidate_scores.shape:
        raise ValueError("Spatial slots must align with semantic candidates")
    if spatial_entropy.shape != candidate_scores.shape:
        raise ValueError("Spatial entropy must align with semantic candidates")
    if semantic_temperature <= 0.0 or contrast_temperature <= 0.0:
        raise ValueError("Temperatures must be positive")
    if maximum_penalty < 0.0 or core_membership <= 0.0:
        raise ValueError("Spatial penalty configuration is invalid")
    if not 0.0 <= entropy_relaxation <= 1.0:
        raise ValueError("entropy_relaxation must be in [0, 1]")
    if not 0.0 <= recovery_factor <= 1.0:
        raise ValueError("recovery_factor must be in [0, 1]")

    semantic_quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    semantic_selectable = valid & (semantic_quality > 0.0)
    covered = semantic_selectable.any(dim=1)
    semantic_logits = (
        candidate_scores / semantic_temperature
        + semantic_quality.clamp_min(1e-8).log()
    )
    semantic_logits = torch.where(
        semantic_selectable,
        semantic_logits,
        torch.full_like(semantic_logits, -torch.inf),
    )
    safe_logits = torch.where(covered.unsqueeze(1), semantic_logits, torch.zeros_like(semantic_logits))
    dominant_slot = safe_logits.argmax(dim=1)
    semantic_output = candidate_scores.gather(1, dominant_slot.unsqueeze(1))
    semantic_output = torch.where(covered.unsqueeze(1), semantic_output, base_scores)

    slot_index = dominant_slot[:, None, None].expand(-1, 1, spatial_core_scores.shape[2])
    selected_core = spatial_core_scores.gather(1, slot_index).squeeze(1)
    selected_ring = spatial_ring_scores.gather(1, slot_index).squeeze(1)
    selected_membership = spatial_memberships.gather(1, slot_index).squeeze(1)
    selected_reliability = spatial_reliability.gather(1, slot_index).squeeze(1)
    selected_valid = spatial_valid.gather(1, slot_index).squeeze(1)
    selected_entropy = spatial_entropy.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)

    contrast = selected_core - ring_weight * selected_ring
    query_trust = torch.sigmoid(contrast / contrast_temperature)
    support = (selected_membership / core_membership).clamp(0.0, 1.0)
    structural_trust = selected_reliability.clamp(0.0, 1.0) * (
        1.0 - entropy_relaxation * selected_entropy.clamp(0.0, 1.0).unsqueeze(1)
    )
    posterior = support * query_trust * structural_trust
    posterior = torch.where(selected_valid, posterior, torch.zeros_like(posterior))
    best_posterior = posterior.max(dim=1).values
    best_support = support.max(dim=1).values
    best_query_trust = query_trust.max(dim=1).values
    spatial_covered = selected_valid.any(dim=1)
    penalty = maximum_penalty * best_query_trust * structural_trust.max(dim=1).values * (
        1.0 - best_support
    )
    penalty = torch.where(spatial_covered & covered, penalty, torch.zeros_like(penalty))

    recovered = torch.zeros_like(covered)
    if gaussian_atom_ids is not None or atom_neighbor_ids is not None:
        if gaussian_atom_ids is None or atom_neighbor_ids is None:
            raise ValueError("Geodesic recovery requires atom IDs and neighbors together")
        if gaussian_atom_ids.shape != (candidate_scores.shape[0],):
            raise ValueError("Gaussian atom IDs must have shape [N]")
        atom_count = int(atom_neighbor_ids.shape[0])
        atom_scores = torch.full(
            (atom_count,), -torch.inf, dtype=semantic_output.dtype, device=semantic_output.device
        )
        atom_scores.scatter_reduce_(
            0,
            gaussian_atom_ids,
            semantic_output.squeeze(1),
            reduce="amax",
            include_self=True,
        )
        seed_points = covered & spatial_covered & (best_support >= 1.0) & (best_query_trust >= 0.5)
        atom_seed = torch.zeros(atom_count, dtype=torch.float32, device=semantic_output.device)
        atom_seed.scatter_reduce_(
            0,
            gaussian_atom_ids,
            seed_points.to(torch.float32),
            reduce="amax",
            include_self=True,
        )
        safe_neighbors = atom_neighbor_ids.clamp_min(0)
        neighbor_valid = atom_neighbor_ids >= 0
        neighbor_seed = torch.where(
            neighbor_valid,
            atom_seed[safe_neighbors] > 0.0,
            torch.zeros_like(neighbor_valid),
        )
        neighbor_scores = torch.where(
            neighbor_seed,
            atom_scores[safe_neighbors],
            torch.full_like(atom_scores[safe_neighbors], -torch.inf),
        ).max(dim=1).values
        point_neighbor_scores = neighbor_scores[gaussian_atom_ids]
        boundary = (
            covered
            & spatial_covered
            & (best_support < 1.0)
            & ((selected_entropy >= 0.25) | (best_support >= 0.1))
        )
        recovered = (
            boundary
            & torch.isfinite(point_neighbor_scores)
            & (semantic_output.squeeze(1) >= point_neighbor_scores - geodesic_delta)
        )
        penalty = torch.where(recovered, recovery_factor * penalty, penalty)

    output = semantic_output - penalty.unsqueeze(1)
    return output, {
        "semantic_covered_points": int(covered.sum().item()),
        "semantic_fallback_points": int((~covered).sum().item()),
        "semantic_mean_valid_slots": float(semantic_selectable.sum(dim=1)[covered].float().mean().item())
        if covered.any()
        else 0.0,
        "spatial_covered_points": int(spatial_covered.sum().item()),
        "mean_spatial_support": float(best_support[spatial_covered].mean().item())
        if spatial_covered.any()
        else 0.0,
        "mean_query_trust": float(best_query_trust[spatial_covered].mean().item())
        if spatial_covered.any()
        else 0.0,
        "mean_spatial_posterior": float(best_posterior[spatial_covered].mean().item())
        if spatial_covered.any()
        else 0.0,
        "penalized_points": int((penalty > 0.0).sum().item()),
        "mean_penalty": float(penalty[penalty > 0.0].mean().item())
        if (penalty > 0.0).any()
        else 0.0,
        "geodesic_recovered_points": int(recovered.sum().item()),
        "geodesic_recovered_fraction": float(recovered.float().mean().item()),
        "maximum_penalty": float(maximum_penalty),
    }


@torch.no_grad()
def conformal_spatial_gate(
    semantic_scores,
    spatial_support,
    covered,
    anchor_quantile,
    outside_quantile,
    temperature,
    use_null_expert,
    semantic_preservation_quantile=None,
):
    """Calibrate spatial trust from selected-core and rejected-region scores."""
    if semantic_scores.ndim != 1 or spatial_support.shape != semantic_scores.shape:
        raise ValueError("Conformal scores and spatial support must be vectors")
    if covered.shape != semantic_scores.shape or covered.dtype != torch.bool:
        raise ValueError("Conformal coverage must be a boolean vector")
    if not 0.0 <= anchor_quantile <= 1.0 or not 0.0 <= outside_quantile <= 1.0:
        raise ValueError("Conformal quantiles must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("Conformal temperature must be positive")
    if semantic_preservation_quantile is not None and not (
        0.0 <= semantic_preservation_quantile <= 1.0
    ):
        raise ValueError("Semantic preservation quantile must be in [0, 1]")

    anchor_mask = covered & (spatial_support >= 0.95)
    outside_mask = covered & (spatial_support <= 0.05)
    if not anchor_mask.any():
        return torch.zeros_like(semantic_scores), {
            "anchor_points": 0,
            "outside_points": int(outside_mask.sum().item()),
            "anchor_score": 0.0,
            "outside_score": 0.0,
            "null_expert_weight": 1.0,
            "mean_conformal_gate": 0.0,
        }

    raw_anchor_score = torch.quantile(
        semantic_scores[anchor_mask], anchor_quantile
    )
    semantic_preservation_score = raw_anchor_score
    if semantic_preservation_quantile is not None:
        semantic_preservation_score = torch.quantile(
            semantic_scores[covered], semantic_preservation_quantile
        )
    anchor_score = torch.minimum(raw_anchor_score, semantic_preservation_score)
    point_gate = torch.sigmoid((anchor_score - semantic_scores) / temperature)
    query_trust = torch.ones((), dtype=semantic_scores.dtype, device=semantic_scores.device)
    outside_score = anchor_score
    if use_null_expert:
        if outside_mask.any():
            outside_score = torch.quantile(
                semantic_scores[outside_mask], outside_quantile
            )
            query_trust = torch.sigmoid(
                (anchor_score - outside_score) / temperature
            )
        else:
            query_trust = torch.zeros_like(query_trust)
    gate = torch.where(covered, point_gate * query_trust, torch.zeros_like(point_gate))
    return gate, {
        "anchor_points": int(anchor_mask.sum().item()),
        "outside_points": int(outside_mask.sum().item()),
        "anchor_score": float(anchor_score.item()),
        "raw_anchor_score": float(raw_anchor_score.item()),
        "semantic_preservation_score": float(
            semantic_preservation_score.item()
        ),
        "outside_score": float(outside_score.item()),
        "null_expert_weight": float((1.0 - query_trust).item()),
        "mean_conformal_gate": float(gate[covered].mean().item())
        if covered.any()
        else 0.0,
    }


@torch.no_grad()
def fuse_global_sparse_group_retrieval(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    valid,
    semantic_temperature,
    group_confidence,
    group_probability,
    spatial_memberships,
    spatial_entropy,
    spatial_valid,
    maximum_penalty,
    core_membership,
    entropy_relaxation,
    gaussian_atom_ids=None,
    atom_neighbor_ids=None,
    geodesic_delta=0.05,
    recovery_factor=0.2,
    anchor_quantile=None,
    outside_quantile=0.75,
    anchor_temperature=0.02,
    use_null_expert=False,
    semantic_preservation_quantile=None,
):
    """Retrieve semantic tokens, then restrict them to globally selected Groups."""
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Semantic candidate tensors must match")
    if not (
        group_confidence.shape
        == group_probability.shape
        == spatial_memberships.shape
        == spatial_valid.shape
    ):
        raise ValueError("Global Group posterior tensors must match")
    if group_confidence.shape[:2] != candidate_scores.shape:
        raise ValueError("Global Group slots must align with semantic candidates")
    if spatial_entropy.shape != candidate_scores.shape:
        raise ValueError("Spatial entropy must align with semantic candidates")
    if semantic_temperature <= 0.0 or maximum_penalty < 0.0 or core_membership <= 0.0:
        raise ValueError("Global sparse retrieval configuration is invalid")
    if not 0.0 <= entropy_relaxation <= 1.0:
        raise ValueError("entropy_relaxation must be in [0, 1]")
    if anchor_quantile is not None and not 0.0 <= anchor_quantile <= 1.0:
        raise ValueError("anchor_quantile must be in [0, 1]")

    semantic_quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    semantic_selectable = valid & (semantic_quality > 0.0)
    covered = semantic_selectable.any(dim=1)
    logits = candidate_scores / semantic_temperature + semantic_quality.clamp_min(1e-8).log()
    logits = torch.where(semantic_selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    dominant_slot = safe_logits.argmax(dim=1)
    semantic_output = candidate_scores.gather(1, dominant_slot.unsqueeze(1))
    semantic_output = torch.where(covered.unsqueeze(1), semantic_output, base_scores)

    slot_index = dominant_slot[:, None, None].expand(-1, 1, group_confidence.shape[2])
    selected_confidence = group_confidence.gather(1, slot_index).squeeze(1)
    selected_probability = group_probability.gather(1, slot_index).squeeze(1)
    selected_membership = spatial_memberships.gather(1, slot_index).squeeze(1)
    selected_valid = spatial_valid.gather(1, slot_index).squeeze(1)
    selected_entropy = spatial_entropy.gather(1, dominant_slot.unsqueeze(1)).squeeze(1)
    support = (
        (selected_membership / core_membership).clamp(0.0, 1.0)
        * selected_confidence.clamp(0.0, 1.0)
    )
    support = torch.where(selected_valid, support, torch.zeros_like(support))
    best_support = support.max(dim=1).values
    best_probability = selected_probability.max(dim=1).values
    spatial_covered = selected_valid.any(dim=1)
    entropy_gate = 1.0 - entropy_relaxation * selected_entropy.clamp(0.0, 1.0)
    penalty = maximum_penalty * (1.0 - best_support) * entropy_gate
    penalty = torch.where(spatial_covered & covered, penalty, torch.zeros_like(penalty))

    conformal_stats = {}
    if anchor_quantile is not None:
        conformal_gate, conformal_stats = conformal_spatial_gate(
            semantic_output.squeeze(1),
            best_support,
            spatial_covered & covered,
            anchor_quantile,
            outside_quantile,
            anchor_temperature,
            use_null_expert,
            semantic_preservation_quantile,
        )
        penalty = penalty * conformal_gate

    recovered = torch.zeros_like(covered)
    if gaussian_atom_ids is not None or atom_neighbor_ids is not None:
        if gaussian_atom_ids is None or atom_neighbor_ids is None:
            raise ValueError("Geodesic recovery requires atom IDs and neighbors together")
        atom_count = int(atom_neighbor_ids.shape[0])
        atom_scores = torch.full(
            (atom_count,), -torch.inf, dtype=semantic_output.dtype, device=semantic_output.device
        )
        atom_scores.scatter_reduce_(
            0, gaussian_atom_ids, semantic_output.squeeze(1), reduce="amax", include_self=True
        )
        seed_points = covered & spatial_covered & (best_support >= 0.95)
        atom_seed = torch.zeros(atom_count, dtype=torch.float32, device=semantic_output.device)
        atom_seed.scatter_reduce_(
            0, gaussian_atom_ids, seed_points.float(), reduce="amax", include_self=True
        )
        safe_neighbors = atom_neighbor_ids.clamp_min(0)
        neighbor_valid = atom_neighbor_ids >= 0
        neighbor_seed = neighbor_valid & (atom_seed[safe_neighbors] > 0.0)
        neighbor_scores = torch.where(
            neighbor_seed,
            atom_scores[safe_neighbors],
            torch.full_like(atom_scores[safe_neighbors], -torch.inf),
        ).max(dim=1).values
        point_neighbor_scores = neighbor_scores[gaussian_atom_ids]
        boundary = covered & spatial_covered & (best_support < 0.95)
        recovered = (
            boundary
            & torch.isfinite(point_neighbor_scores)
            & (semantic_output.squeeze(1) >= point_neighbor_scores - geodesic_delta)
        )
        penalty = torch.where(recovered, recovery_factor * penalty, penalty)

    output = semantic_output - penalty.unsqueeze(1)
    diagnostics = {
        "semantic_covered_points": int(covered.sum().item()),
        "semantic_fallback_points": int((~covered).sum().item()),
        "semantic_mean_valid_slots": float(semantic_selectable.sum(dim=1)[covered].float().mean().item())
        if covered.any()
        else 0.0,
        "spatial_covered_points": int(spatial_covered.sum().item()),
        "mean_selected_group_support": float(best_support[spatial_covered].mean().item())
        if spatial_covered.any()
        else 0.0,
        "mean_selected_group_probability": float(best_probability[spatial_covered].mean().item())
        if spatial_covered.any()
        else 0.0,
        "penalized_points": int((penalty > 0.0).sum().item()),
        "mean_penalty": float(penalty[penalty > 0.0].mean().item())
        if (penalty > 0.0).any()
        else 0.0,
        "geodesic_recovered_points": int(recovered.sum().item()),
        "geodesic_recovered_fraction": float(recovered.float().mean().item()),
        "maximum_penalty": float(maximum_penalty),
    }
    diagnostics.update(conformal_stats)
    return output, diagnostics


@torch.no_grad()
def complete_scores_from_seeded_groups(
    scores,
    candidate_scores,
    semantic_memberships,
    candidate_reliability,
    semantic_valid,
    semantic_temperature,
    group_ids,
    group_confidence,
    spatial_memberships,
    spatial_valid,
    gaussian_atom_ids,
    atom_neighbor_ids,
    atom_neighbor_weights,
    core_membership,
    boundary_membership,
    seed_support,
    seed_quantile,
    seed_score_floor,
    target_quantile,
    semantic_delta,
    agreement_temperature,
    completion_strength,
    maximum_expansion_ratio,
    minimum_seed_points,
    minimum_contact,
    maximum_hops,
    atom_centroids=None,
    group_principal_axes=None,
    group_axis_ratios=None,
    anisotropic_axis_floor=0.15,
    anisotropic_budget_floor=0.25,
    anisotropic_semantic_floor=0.50,
    anisotropic_direction_power=1.0,
    seed_conditioned_anisotropy=False,
    token_profile_gate=False,
    token_profile_quantile=0.90,
    token_profile_margin=0.03,
    token_profile_temperature=0.01,
    token_profile_minimum_slots=2,
):
    """Complete a query score only inside a certified, seed-connected Group.

    The resident tokens remain unchanged. A Group may raise weak member scores
    only when it contains strong semantic seeds, is selected by the global
    Group posterior, and forms one connected component in the 3D atom graph.
    """
    if scores.ndim != 2 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == semantic_memberships.shape
        == candidate_reliability.shape
        == semantic_valid.shape
    ):
        raise ValueError("Semantic candidate tensors must match")
    expected_group_shape = (*candidate_scores.shape, 2)
    if not (
        group_ids.shape
        == group_confidence.shape
        == spatial_memberships.shape
        == spatial_valid.shape
        == expected_group_shape
    ):
        raise ValueError("Group candidate tensors must have shape [N, S, 2]")
    if gaussian_atom_ids.shape != (scores.shape[0],):
        raise ValueError("Gaussian atom IDs must have shape [N]")
    if atom_neighbor_ids.shape != atom_neighbor_weights.shape:
        raise ValueError("Atom neighbor IDs and weights must match")
    if atom_neighbor_ids.ndim != 2:
        raise ValueError("Atom neighbors must have shape [A, K]")
    if semantic_temperature <= 0.0 or agreement_temperature <= 0.0:
        raise ValueError("Completion temperatures must be positive")
    if not 0.0 < boundary_membership <= core_membership:
        raise ValueError("Completion memberships are invalid")
    if not 0.0 <= seed_support <= 1.0:
        raise ValueError("seed_support must be in [0, 1]")
    if not 0.0 <= seed_quantile <= 1.0:
        raise ValueError("seed_quantile must be in [0, 1]")
    if not torch.isfinite(scores.new_tensor(seed_score_floor)):
        raise ValueError("seed_score_floor must be finite")
    if not 0.0 <= target_quantile <= 1.0:
        raise ValueError("target_quantile must be in [0, 1]")
    if semantic_delta < 0.0 or not 0.0 <= completion_strength <= 1.0:
        raise ValueError("Completion score controls are invalid")
    if maximum_expansion_ratio < 0.0 or minimum_seed_points <= 0:
        raise ValueError("Completion budget is invalid")
    if minimum_contact < 0.0 or maximum_hops <= 0:
        raise ValueError("Completion connectivity controls are invalid")
    anisotropic_inputs = (
        atom_centroids,
        group_principal_axes,
        group_axis_ratios,
    )
    anisotropic = all(value is not None for value in anisotropic_inputs)
    if any(value is not None for value in anisotropic_inputs) and not anisotropic:
        raise ValueError("Anisotropic completion geometry must be provided together")
    if anisotropic:
        atom_count = atom_neighbor_ids.shape[0]
        if atom_centroids.shape != (atom_count, 3):
            raise ValueError("Anisotropic atom centroids must have shape [A, 3]")
        if group_principal_axes.ndim != 3 or group_principal_axes.shape[1:] != (3, 3):
            raise ValueError("Group principal axes must have shape [G, 3, 3]")
        if group_axis_ratios.shape != group_principal_axes.shape[:1] + (3,):
            raise ValueError("Group axis ratios must have shape [G, 3]")
        if not 0.0 < anisotropic_axis_floor <= 1.0:
            raise ValueError("anisotropic_axis_floor must be in (0, 1]")
        if not 0.0 <= anisotropic_budget_floor <= 1.0:
            raise ValueError("anisotropic_budget_floor must be in [0, 1]")
        if not 0.0 <= anisotropic_semantic_floor <= 1.0:
            raise ValueError("anisotropic_semantic_floor must be in [0, 1]")
        if anisotropic_direction_power <= 0.0:
            raise ValueError("anisotropic_direction_power must be positive")
    if not 0.0 <= token_profile_quantile <= 1.0:
        raise ValueError("token_profile_quantile must be in [0, 1]")
    if token_profile_margin < 0.0 or token_profile_temperature <= 0.0:
        raise ValueError("Token-profile gate controls are invalid")
    if token_profile_gate and (
        token_profile_minimum_slots <= 0
        or token_profile_minimum_slots > candidate_scores.shape[1]
    ):
        raise ValueError("token_profile_minimum_slots is invalid")

    semantic_quality = semantic_memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    selectable = semantic_valid & (semantic_quality > 0.0)
    covered = selectable.any(dim=1)
    logits = candidate_scores / semantic_temperature + semantic_quality.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    dominant_slot = safe_logits.argmax(dim=1)

    gather_index = dominant_slot[:, None, None].expand(-1, 1, 2)
    selected_ids = group_ids.gather(1, gather_index).squeeze(1)
    selected_confidence = group_confidence.gather(1, gather_index).squeeze(1)
    selected_membership = spatial_memberships.gather(1, gather_index).squeeze(1)
    selected_valid = spatial_valid.gather(1, gather_index).squeeze(1)
    support = (
        (selected_membership / core_membership).clamp(0.0, 1.0)
        * selected_confidence.clamp(0.0, 1.0)
    )
    selected_valid = selected_valid & (selected_ids >= 0)
    support = torch.where(selected_valid, support, torch.zeros_like(support))
    best_group_slot = support.argmax(dim=1)
    assigned_group = selected_ids.gather(1, best_group_slot.unsqueeze(1)).squeeze(1)
    assigned_membership = selected_membership.gather(
        1, best_group_slot.unsqueeze(1)
    ).squeeze(1)
    assigned_support = support.gather(1, best_group_slot.unsqueeze(1)).squeeze(1)
    assigned_valid = (
        covered
        & selected_valid.gather(1, best_group_slot.unsqueeze(1)).squeeze(1)
        & (assigned_support > 0.0)
    )

    output = scores.clone()
    flat_scores = scores[:, 0]
    potential_seed = (
        assigned_valid
        & (assigned_membership >= core_membership)
        & (assigned_support >= seed_support)
    )
    candidate_groups = torch.unique(assigned_group[potential_seed])
    safe_neighbors = atom_neighbor_ids.clamp_min(0).long()
    valid_edges = (atom_neighbor_ids >= 0) & (atom_neighbor_weights >= minimum_contact)
    atom_count = atom_neighbor_ids.shape[0]
    completed = torch.zeros_like(covered)
    certified_groups = 0
    rejected_small_seed = 0
    reached_hop_limit = 0
    shape_budget_scales = []
    secondary_axis_ratios = []
    edge_retention = []
    seed_conditioned_groups = 0
    seed_shape_fallback_groups = 0
    profile_certified_groups = 0
    profile_rejected_groups = 0
    profile_radii = []
    profile_candidate_gates = []

    for group_id in candidate_groups.tolist():
        group_core = potential_seed & (assigned_group == group_id)
        if int(group_core.sum().item()) < minimum_seed_points:
            rejected_small_seed += 1
            continue
        local_threshold = torch.quantile(flat_scores[group_core], seed_quantile)
        seed_threshold = torch.maximum(
            local_threshold, flat_scores.new_tensor(seed_score_floor)
        )
        seeds = group_core & (flat_scores >= seed_threshold)
        seed_count = int(seeds.sum().item())
        if seed_count < minimum_seed_points:
            rejected_small_seed += 1
            continue
        target_score = torch.quantile(flat_scores[seeds], target_quantile)
        members = (
            assigned_valid
            & (assigned_group == group_id)
            & (assigned_membership >= boundary_membership)
        )
        profile_gate = torch.ones_like(flat_scores)
        if token_profile_gate:
            masked_profiles = torch.where(
                semantic_valid,
                candidate_scores,
                torch.full_like(candidate_scores, -torch.inf),
            )
            profile_peak = masked_profiles.max(dim=1, keepdim=True).values
            relative_profiles = torch.where(
                semantic_valid,
                candidate_scores - profile_peak,
                torch.zeros_like(candidate_scores),
            )
            seed_valid = semantic_valid[seeds]
            seed_profiles = relative_profiles[seeds]
            profile_count = seed_valid.sum(dim=0)
            prototype_valid = profile_count > 0
            if int(prototype_valid.sum().item()) < token_profile_minimum_slots:
                profile_rejected_groups += 1
                continue
            prototype = (
                (seed_profiles * seed_valid.to(seed_profiles.dtype)).sum(dim=0)
                / profile_count.clamp_min(1).to(seed_profiles.dtype)
            )
            common = semantic_valid & prototype_valid.unsqueeze(0)
            common_count = common.sum(dim=1)
            profile_distance = (
                (relative_profiles - prototype.unsqueeze(0)).abs()
                * common.to(candidate_scores.dtype)
            ).sum(dim=1) / common_count.clamp_min(1).to(candidate_scores.dtype)
            valid_seed_distance = seeds & (common_count >= token_profile_minimum_slots)
            if int(valid_seed_distance.sum().item()) < minimum_seed_points:
                profile_rejected_groups += 1
                continue
            radius = torch.quantile(
                profile_distance[valid_seed_distance], token_profile_quantile
            ) + token_profile_margin
            profile_gate = torch.sigmoid(
                (radius - profile_distance) / token_profile_temperature
            )
            profile_gate = torch.where(
                common_count >= token_profile_minimum_slots,
                profile_gate,
                torch.zeros_like(profile_gate),
            )
            profile_certified_groups += 1
            profile_radii.append(float(radius.item()))
            member_gate = profile_gate[members]
            if member_gate.numel():
                profile_candidate_gates.append(float(member_gate.mean().item()))
        seed_atoms = torch.zeros(atom_count, dtype=torch.float32, device=scores.device)
        seed_atoms.scatter_reduce_(
            0, gaussian_atom_ids, seeds.float(), reduce="amax", include_self=True
        )
        budget_scale = 1.0
        local_semantic_delta = semantic_delta
        group_valid_edges = valid_edges
        if anisotropic:
            local_axes = group_principal_axes[group_id]
            local_ratios = group_axis_ratios[group_id]
            if seed_conditioned_anisotropy:
                seed_atom_rows = torch.nonzero(seed_atoms > 0.0, as_tuple=False).squeeze(1)
                if seed_atom_rows.numel() >= 3:
                    seed_points = atom_centroids[seed_atom_rows]
                    centered = seed_points - seed_points.mean(dim=0, keepdim=True)
                    covariance = centered.T @ centered / float(seed_atom_rows.numel())
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                    order = torch.arange(2, -1, -1, device=scores.device)
                    eigenvalues = eigenvalues[order].clamp_min(0.0)
                    if float(eigenvalues[0].item()) > 1e-12:
                        local_axes = eigenvectors[:, order].T
                        local_ratios = torch.sqrt(
                            eigenvalues / eigenvalues[0].clamp_min(1e-12)
                        )
                        seed_conditioned_groups += 1
                    else:
                        seed_shape_fallback_groups += 1
                else:
                    seed_shape_fallback_groups += 1
            ratios = local_ratios.clamp(min=anisotropic_axis_floor, max=1.0)
            secondary_ratio = float(ratios[1].item())
            budget_scale = anisotropic_budget_floor + (
                1.0 - anisotropic_budget_floor
            ) * secondary_ratio
            semantic_scale = anisotropic_semantic_floor + (
                1.0 - anisotropic_semantic_floor
            ) * secondary_ratio
            local_semantic_delta *= semantic_scale
            offsets = atom_centroids[safe_neighbors] - atom_centroids[:, None, :]
            directions = offsets / offsets.norm(dim=2, keepdim=True).clamp_min(1e-8)
            projections = torch.einsum(
                "akd,cd->akc", directions, local_axes
            )
            directional_conductance = torch.sqrt(
                (projections.square() * ratios.square()).sum(dim=2).clamp_min(0.0)
            )
            directional_conductance = directional_conductance.pow(
                anisotropic_direction_power
            )
            group_valid_edges = (
                (atom_neighbor_ids >= 0)
                & (atom_neighbor_weights * directional_conductance >= minimum_contact)
            )
            shape_budget_scales.append(budget_scale)
            secondary_axis_ratios.append(secondary_ratio)
            base_edge_count = int(valid_edges.sum().item())
            edge_retention.append(
                float(group_valid_edges.sum().item() / max(base_edge_count, 1))
            )
        semantic_gate = torch.sigmoid(
            (flat_scores - (target_score - local_semantic_delta))
            / agreement_temperature
        )
        candidates = members & (semantic_gate > 1e-4) & (profile_gate > 1e-4)

        allowed_atoms = torch.zeros_like(seed_atoms)
        allowed_atoms.scatter_reduce_(
            0, gaussian_atom_ids, candidates.float(), reduce="amax", include_self=True
        )
        reachable = seed_atoms > 0.0
        allowed = allowed_atoms > 0.0
        for hop in range(maximum_hops):
            adjacent = (group_valid_edges & reachable[safe_neighbors]).any(dim=1)
            updated = reachable | (allowed & adjacent)
            if torch.equal(updated, reachable):
                break
            reachable = updated
        else:
            reached_hop_limit += 1

        reachable_points = candidates & reachable[gaussian_atom_ids]
        expandable = reachable_points & (~seeds) & (flat_scores < target_score)
        expandable_rows = torch.nonzero(expandable, as_tuple=False).squeeze(1)
        maximum_new = int(round(seed_count * maximum_expansion_ratio * budget_scale))
        if maximum_new <= 0 or expandable_rows.numel() == 0:
            certified_groups += 1
            continue
        agreement = (
            assigned_support.clamp(0.0, 1.0)
            * (assigned_membership / core_membership).clamp(0.0, 1.0)
            * semantic_gate
            * profile_gate
        )
        gain = (target_score - flat_scores).clamp_min(0.0) * agreement
        if expandable_rows.numel() > maximum_new:
            ranking = gain[expandable_rows]
            expandable_rows = expandable_rows[ranking.topk(maximum_new).indices]
        delta = completion_strength * gain[expandable_rows]
        output[expandable_rows, 0] += delta
        completed[expandable_rows] = delta > 0.0
        certified_groups += 1

    changed = completed & (output[:, 0] != scores[:, 0])
    return output, {
        "seeded_group_completion": True,
        "candidate_seed_groups": int(candidate_groups.numel()),
        "certified_groups": int(certified_groups),
        "rejected_small_seed_groups": int(rejected_small_seed),
        "completed_points": int(changed.sum().item()),
        "completed_fraction": float(changed.float().mean().item()),
        "mean_completion_delta": float(
            (output[changed, 0] - scores[changed, 0]).mean().item()
        ) if changed.any() else 0.0,
        "maximum_completion_delta": float(
            (output[changed, 0] - scores[changed, 0]).max().item()
        ) if changed.any() else 0.0,
        "minimum_seed_score": float(seed_score_floor),
        "target_quantile": float(target_quantile),
        "groups_reaching_hop_limit": int(reached_hop_limit),
        "maximum_expansion_ratio": float(maximum_expansion_ratio),
        "anisotropic_completion": bool(anisotropic),
        "mean_shape_budget_scale": float(sum(shape_budget_scales) / len(shape_budget_scales))
        if shape_budget_scales else 1.0,
        "mean_secondary_axis_ratio": float(
            sum(secondary_axis_ratios) / len(secondary_axis_ratios)
        ) if secondary_axis_ratios else 1.0,
        "mean_anisotropic_edge_retention": float(sum(edge_retention) / len(edge_retention))
        if edge_retention else 1.0,
        "seed_conditioned_anisotropy": bool(seed_conditioned_anisotropy),
        "seed_conditioned_groups": int(seed_conditioned_groups),
        "seed_shape_fallback_groups": int(seed_shape_fallback_groups),
        "anisotropic_direction_power": float(anisotropic_direction_power),
        "token_profile_gate": bool(token_profile_gate),
        "profile_certified_groups": int(profile_certified_groups),
        "profile_rejected_groups": int(profile_rejected_groups),
        "mean_profile_radius": float(sum(profile_radii) / len(profile_radii))
        if profile_radii else 0.0,
        "mean_profile_candidate_gate": float(
            sum(profile_candidate_gates) / len(profile_candidate_gates)
        ) if profile_candidate_gates else 0.0,
    }


@torch.no_grad()
def fuse_multiscale_set_relation_token_scores(
    candidate_scores,
    candidate_levels,
    selectable,
    neighbor_ids,
    relation_signatures,
    positive_strength,
    negative_strength,
    maximum_delta,
):
    """Apply sparse signed relations to each peer token before query retrieval."""
    if not (
        candidate_scores.shape == candidate_levels.shape == selectable.shape
        and candidate_scores.ndim == 2
    ):
        raise ValueError("Candidate scores, levels, and masks must have shape [N, S]")
    if neighbor_ids.ndim != 2 or neighbor_ids.shape[0] != candidate_scores.shape[0]:
        raise ValueError("Neighbor IDs must have shape [N, K]")
    if relation_signatures.shape != (
        neighbor_ids.shape[0],
        neighbor_ids.shape[1],
        candidate_scores.shape[1],
    ):
        raise ValueError("Relation signatures must have shape [N, K, S]")
    if relation_signatures.numel() and (
        relation_signatures.min() < -1.0 or relation_signatures.max() > 1.0
    ):
        raise ValueError("Relation signatures must lie in [-1, 1]")

    corrected = candidate_scores.clone()
    level_diagnostics = {}
    corrected_slots = 0
    for level in range(candidate_scores.shape[1]):
        level_slots = selectable & (candidate_levels == level)
        if (level_slots.sum(dim=1) > 1).any():
            raise ValueError("Each Gaussian may expose at most one token per level")
        level_valid = level_slots.any(dim=1)
        slot_indices = level_slots.to(torch.int64).argmax(dim=1)
        level_scores = candidate_scores.gather(1, slot_indices.unsqueeze(1))
        adjacent_valid = level_valid[neighbor_ids.long()]
        active_edges = level_valid.unsqueeze(1) & adjacent_valid
        signed_weights = relation_signatures[:, :, level] * active_edges.to(
            relation_signatures.dtype
        )
        updated, diagnostics = fuse_signed_relation_graph_scores(
            level_scores,
            neighbor_ids,
            signed_weights,
            positive_strength,
            negative_strength,
            maximum_delta,
        )
        delta = (updated - level_scores).squeeze(1)
        active_points = level_valid & (signed_weights != 0.0).any(dim=1)
        rows = torch.nonzero(active_points, as_tuple=False).squeeze(1)
        if rows.numel():
            corrected[rows, slot_indices[rows]] += delta[rows]
        corrected_slots += int(rows.numel())
        level_diagnostics[f"level_{level}"] = diagnostics

    return corrected, {
        "multiscale_set_relation_correction": True,
        "corrected_token_slots": corrected_slots,
        "corrected_slot_fraction": float(
            corrected_slots / max(int(selectable.sum().item()), 1)
        ),
        "levels": level_diagnostics,
    }


@torch.no_grad()
def fuse_signed_relation_graph_scores(
    scores,
    neighbor_ids,
    signed_relation_weights,
    positive_strength,
    negative_strength,
    maximum_delta,
):
    """Apply one query-aware signed-graph step after peer-token retrieval.

    Positive edges smooth score disagreement. Negative edges sharpen existing
    score differences instead of imposing a semantic label. The update is a
    single Jacobi step over the unmodified query scores, so it cannot propagate
    recursively through a large or accidentally connected component.
    """
    if scores.ndim != 2 or scores.shape[1] != 1:
        raise ValueError("scores must have shape [N, 1]")
    if neighbor_ids.shape != signed_relation_weights.shape:
        raise ValueError("Relation neighbor IDs and weights must match")
    if neighbor_ids.ndim != 2 or neighbor_ids.shape[0] != scores.shape[0]:
        raise ValueError("Relation graph must have shape [N, K]")
    if neighbor_ids.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ValueError("Relation neighbor IDs must be integer tensors")
    if positive_strength < 0.0 or negative_strength < 0.0:
        raise ValueError("Relation strengths must be non-negative")
    if maximum_delta <= 0.0:
        raise ValueError("maximum_delta must be positive")
    if neighbor_ids.numel() and (
        int(neighbor_ids.min()) < 0 or int(neighbor_ids.max()) >= scores.shape[0]
    ):
        raise ValueError("Relation neighbor IDs exceed the score table")
    if not torch.isfinite(signed_relation_weights).all():
        raise ValueError("Relation weights must be finite")
    if (signed_relation_weights.abs() > 1.0 + 1e-6).any():
        raise ValueError("Relation weights must lie in [-1, 1]")

    flat_scores = scores[:, 0]
    neighbor_scores = flat_scores[neighbor_ids.long()]
    score_difference = neighbor_scores - flat_scores.unsqueeze(1)
    positive = signed_relation_weights.clamp_min(0.0)
    negative = (-signed_relation_weights).clamp_min(0.0)

    positive_sum = positive.sum(dim=1)
    negative_sum = negative.sum(dim=1)
    positive_count = (positive > 0.0).sum(dim=1)
    negative_count = (negative > 0.0).sum(dim=1)
    positive_confidence = positive_sum / positive_count.clamp_min(1).to(scores.dtype)
    negative_confidence = negative_sum / negative_count.clamp_min(1).to(scores.dtype)
    positive_delta = (
        (positive * score_difference).sum(dim=1)
        / positive_sum.clamp_min(1e-8)
        * positive_confidence
    )
    negative_delta = (
        (negative * -score_difference).sum(dim=1)
        / negative_sum.clamp_min(1e-8)
        * negative_confidence
    )
    positive_delta = torch.where(
        positive_sum > 0.0, positive_delta, torch.zeros_like(positive_delta)
    )
    negative_delta = torch.where(
        negative_sum > 0.0, negative_delta, torch.zeros_like(negative_delta)
    )
    raw_delta = positive_strength * positive_delta + negative_strength * negative_delta
    applied_delta = raw_delta.clamp(-maximum_delta, maximum_delta)
    output = scores + applied_delta.unsqueeze(1)
    active = (positive_sum > 0.0) | (negative_sum > 0.0)
    clipped = raw_delta.abs() > maximum_delta
    stats = {
        "relation_graph_correction": True,
        "positive_strength": float(positive_strength),
        "negative_strength": float(negative_strength),
        "maximum_delta": float(maximum_delta),
        "active_points": int(active.sum().item()),
        "active_fraction": float(active.float().mean().item()),
        "positive_edge_slots": int((positive > 0.0).sum().item()),
        "negative_edge_slots": int((negative > 0.0).sum().item()),
        "mean_positive_edges_active": float(
            positive_count[active].float().mean().item()
        )
        if active.any()
        else 0.0,
        "mean_negative_edges_active": float(
            negative_count[active].float().mean().item()
        )
        if active.any()
        else 0.0,
        "mean_absolute_delta_active": float(applied_delta[active].abs().mean().item())
        if active.any()
        else 0.0,
        "mean_signed_delta_active": float(applied_delta[active].mean().item())
        if active.any()
        else 0.0,
        "clipped_points": int(clipped.sum().item()),
        "clipped_fraction_active": float(
            clipped.sum().float().div(active.sum().clamp_min(1)).item()
        ),
    }
    return output, stats


@torch.no_grad()
def fuse_quantization_aware_equal_query_tokens(
    base_scores,
    candidate_scores,
    memberships,
    candidate_reliability,
    quantization_error,
    valid,
    temperature,
    uncertainty_scale,
    candidate_levels=None,
    uncertainty_measure_name="chord_error",
):
    """Select among peer tokens using quantization-adaptive score intervals.

    The four hierarchy slots remain symmetric.  Each slot's codebook
    reconstruction error defines a score-space uncertainty radius.  A raw
    winner is retained when its lower bound dominates every competing upper
    bound; otherwise the token with the strongest lower confidence bound wins.
    The selected token's unmodified relevancy score is returned so downstream
    thresholds stay comparable with hard equal-token retrieval.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == quantization_error.shape
        == valid.shape
    ):
        raise ValueError("Quantization-aware equal-token tensors must match")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if candidate_levels is not None and candidate_levels.shape != candidate_scores.shape:
        raise ValueError("candidate_levels must match candidate scores")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if uncertainty_scale < 0.0:
        raise ValueError("uncertainty_scale must be non-negative")
    if (quantization_error < 0.0).any():
        raise ValueError("quantization_error must be non-negative")

    quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    selectable = (
        valid
        & (quality > 0.0)
        & torch.isfinite(candidate_scores)
        & torch.isfinite(quantization_error)
    )
    covered = selectable.any(dim=1)
    adjusted_scores = candidate_scores + temperature * quality.clamp_min(1e-8).log()
    radii = uncertainty_scale * quantization_error
    lower = torch.where(
        selectable, adjusted_scores - radii, torch.full_like(adjusted_scores, -torch.inf)
    )
    upper = torch.where(
        selectable, adjusted_scores + radii, torch.full_like(adjusted_scores, -torch.inf)
    )
    centers = torch.where(
        selectable, adjusted_scores, torch.full_like(adjusted_scores, -torch.inf)
    )

    raw_slot = centers.argmax(dim=1)
    raw_lower = lower.gather(1, raw_slot.unsqueeze(1)).squeeze(1)
    competing_upper = upper.scatter(
        1, raw_slot.unsqueeze(1), torch.full_like(raw_slot.unsqueeze(1), -torch.inf, dtype=upper.dtype)
    ).max(dim=1).values
    single_slot = selectable.sum(dim=1) == 1
    interval_dominant = covered & (single_slot | (raw_lower > competing_upper))
    robust_slot = lower.argmax(dim=1)
    selected_slot = torch.where(interval_dominant, raw_slot, robust_slot)
    selected_scores = candidate_scores.gather(1, selected_slot.unsqueeze(1))
    output = torch.where(covered.unsqueeze(1), selected_scores, base_scores)

    ambiguous = covered & ~interval_dominant
    changed = ambiguous & (selected_slot != raw_slot)
    selected_error = quantization_error.gather(1, selected_slot.unsqueeze(1)).squeeze(1)
    selected_radius = radii.gather(1, selected_slot.unsqueeze(1)).squeeze(1)
    stats = {
        "covered_points": int(covered.sum().item()),
        "fallback_points": int((~covered).sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "hard_query_retrieval": True,
        "quantization_aware_interval": True,
        "quantization_uncertainty_measure": str(uncertainty_measure_name),
        "quantization_uncertainty_scale": float(uncertainty_scale),
        "interval_dominant_points": int(interval_dominant.sum().item()),
        "ambiguous_points": int(ambiguous.sum().item()),
        "ambiguous_fraction_covered": float(
            ambiguous.sum().float().div(covered.sum().clamp_min(1)).item()
        ),
        "selection_changed_points": int(changed.sum().item()),
        "selection_changed_fraction_covered": float(
            changed.sum().float().div(covered.sum().clamp_min(1)).item()
        ),
        "mean_selected_quantization_error": float(selected_error[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_selected_uncertainty_measure": float(
            selected_error[covered].mean().item()
        )
        if covered.any()
        else 0.0,
        "mean_selected_uncertainty_radius": float(selected_radius[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta_from_fallback": float((output - base_scores)[covered].mean().item())
        if covered.any()
        else 0.0,
    }
    if candidate_levels is not None and covered.any():
        selected_levels = candidate_levels.gather(
            1, selected_slot.unsqueeze(1)
        ).squeeze(1)
        stats["dominant_level_counts"] = {
            f"level_{int(level)}": int(((selected_levels == level) & covered).sum().item())
            for level in torch.unique(candidate_levels[selectable]).tolist()
            if int(level) >= 0
        }
    return output, stats


@torch.no_grad()
def fuse_calibrated_equal_query_tokens(
    base_scores,
    candidate_scores,
    calibration_scores,
    memberships,
    candidate_reliability,
    valid,
    calibration_mode,
    temperature,
    candidate_levels=None,
):
    """Select a peer token with level-calibrated evidence, then return raw score.

    Calibration only changes which resident slot wins. The selected token keeps
    its original relevancy score so fixed downstream selection thresholds remain
    comparable to uncalibrated equal-token retrieval.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == calibration_scores.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Calibrated equal-token tensors must have matching shapes")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if candidate_levels is not None and candidate_levels.shape != candidate_scores.shape:
        raise ValueError("candidate_levels must match candidate scores")
    if calibration_mode not in {"percentile", "tail_evidence"}:
        raise ValueError(f"Unknown equal-token calibration: {calibration_mode}")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    selectable = valid & (quality > 0.0) & torch.isfinite(calibration_scores)
    covered = selectable.any(dim=1)
    scale = temperature if calibration_mode == "percentile" else 1.0
    logits = calibration_scores / scale + quality.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    dominant_slot = safe_logits.argmax(dim=1)
    selected_scores = candidate_scores.gather(1, dominant_slot.unsqueeze(1))
    output = torch.where(covered.unsqueeze(1), selected_scores, base_scores)
    selected_calibration = calibration_scores.gather(
        1, dominant_slot.unsqueeze(1)
    ).squeeze(1)
    valid_slots = selectable.sum(dim=1)

    stats = {
        "covered_points": int(covered.sum().item()),
        "fallback_points": int((~covered).sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "hard_query_retrieval": True,
        "level_score_calibration": calibration_mode,
        "calibration_temperature": float(scale),
        "mean_valid_slots": float(valid_slots[covered].float().mean().item())
        if covered.any()
        else 0.0,
        "mean_selected_calibration": float(selected_calibration[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta_from_fallback": float(
            (output - base_scores)[covered].mean().item()
        )
        if covered.any()
        else 0.0,
    }
    if candidate_levels is not None and covered.any():
        selected_levels = candidate_levels.gather(
            1, dominant_slot.unsqueeze(1)
        ).squeeze(1)
        stats["dominant_level_counts"] = {
            f"level_{int(level)}": int(
                ((selected_levels == level) & covered).sum().item()
            )
            for level in torch.unique(candidate_levels[selectable]).tolist()
            if int(level) >= 0
        }
    return output, stats


@torch.no_grad()
def fuse_information_gain_equal_query_tokens(
    base_scores,
    candidate_scores,
    information_gain,
    memberships,
    candidate_reliability,
    valid,
    candidate_levels=None,
    local_counterfactual_gain=None,
):
    """Route peer tokens by information gain while preserving raw score output.

    The global term measures how informative a token is relative to its complete
    level codebook.  The optional local term is a Bayes factor against the
    token's nearest same-level semantic alternatives.  Its log-sigmoid is the
    probability that the candidate wins a pairwise counterfactual comparison:
    ties pay a log(2) uncertainty cost, while decisive wins approach zero cost.
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 1:
        raise ValueError("base_scores must have shape [N, 1]")
    if not (
        candidate_scores.shape
        == information_gain.shape
        == memberships.shape
        == candidate_reliability.shape
        == valid.shape
    ):
        raise ValueError("Information-gain token tensors must have matching shapes")
    if candidate_scores.ndim != 2 or candidate_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("Candidate scores must have shape [N, K]")
    if candidate_levels is not None and candidate_levels.shape != candidate_scores.shape:
        raise ValueError("candidate_levels must match candidate scores")
    if (
        local_counterfactual_gain is not None
        and local_counterfactual_gain.shape != candidate_scores.shape
    ):
        raise ValueError("local_counterfactual_gain must match candidate scores")

    quality = memberships.clamp(0.0, 1.0) * candidate_reliability.clamp(0.0, 1.0)
    evidence = information_gain
    if local_counterfactual_gain is not None:
        evidence = evidence + F.logsigmoid(local_counterfactual_gain)
    selectable = valid & (quality > 0.0) & torch.isfinite(evidence)
    covered = selectable.any(dim=1)
    logits = evidence + quality.clamp_min(1e-8).log()
    logits = torch.where(selectable, logits, torch.full_like(logits, -torch.inf))
    safe_logits = torch.where(covered.unsqueeze(1), logits, torch.zeros_like(logits))
    dominant_slot = safe_logits.argmax(dim=1)
    selected_scores = candidate_scores.gather(1, dominant_slot.unsqueeze(1))
    output = torch.where(covered.unsqueeze(1), selected_scores, base_scores)
    selected_gain = information_gain.gather(
        1, dominant_slot.unsqueeze(1)
    ).squeeze(1)
    valid_slots = selectable.sum(dim=1)

    stats = {
        "covered_points": int(covered.sum().item()),
        "fallback_points": int((~covered).sum().item()),
        "routed_points": int(covered.sum().item()),
        "routed_fraction_all": float(covered.float().mean().item()),
        "hard_query_retrieval": True,
        "level_score_calibration": "occupancy_prior_information_gain",
        "local_counterfactual_penalty": local_counterfactual_gain is not None,
        "mean_valid_slots": float(valid_slots[covered].float().mean().item())
        if covered.any()
        else 0.0,
        "mean_selected_information_gain": float(selected_gain[covered].mean().item())
        if covered.any()
        else 0.0,
        "mean_score_delta_from_fallback": float(
            (output - base_scores)[covered].mean().item()
        )
        if covered.any()
        else 0.0,
    }
    if local_counterfactual_gain is not None:
        selected_local = local_counterfactual_gain.gather(
            1, dominant_slot.unsqueeze(1)
        ).squeeze(1)
        nonpositive = covered & (selected_local <= 0.0)
        selected_penalty = -F.logsigmoid(selected_local)
        stats.update(
            {
                "counterfactual_nonpositive_points": int(nonpositive.sum().item()),
                "counterfactual_nonpositive_fraction_covered": float(
                    nonpositive.sum().float().div(covered.sum().clamp_min(1)).item()
                ),
                "mean_selected_local_counterfactual_gain": float(
                    selected_local[covered].mean().item()
                )
                if covered.any()
                else 0.0,
                "mean_selected_counterfactual_penalty": float(
                    selected_penalty[covered].mean().item()
                )
                if covered.any()
                else 0.0,
            }
        )
    if candidate_levels is not None and covered.any():
        selected_levels = candidate_levels.gather(
            1, dominant_slot.unsqueeze(1)
        ).squeeze(1)
        stats["dominant_level_counts"] = {
            f"level_{int(level)}": int(
                ((selected_levels == level) & covered).sum().item()
            )
            for level in torch.unique(candidate_levels[selectable]).tolist()
            if int(level) >= 0
        }
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
