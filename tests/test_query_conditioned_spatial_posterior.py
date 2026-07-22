import numpy as np
import pytest

from build_query_conditioned_spatial_posterior import top2_group_support


def test_top2_group_support_preserves_overlap_without_cross_level_mixing():
    profiles = np.array(
        [
            [0.8, 0.2], [0.6, 0.7], [0.1, 0.4],
            [0.9, 0.1], [0.2, 0.8],
            [0.7, 0.3], [0.4, 0.9],
            [0.5, 0.6], [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    levels = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    ids, memberships, entropy = top2_group_support(profiles, levels, 0.15)
    assert ids.shape == memberships.shape == (2, 4, 2)
    assert ids[0, 0].tolist() == [0, 1]
    assert ids[1, 0].tolist() == [1, 2]
    assert ids[0, 1].tolist() == [3, 4]
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))


torch = pytest.importorskip("torch")

from semantic_hypothesis_routing import (
    complete_scores_from_seeded_groups,
    conformal_spatial_gate,
    fuse_query_conditioned_spatial_posterior,
)


def routing_inputs():
    base = torch.tensor([[0.1], [0.2]])
    candidates = torch.tensor([[0.4, 0.7, 0.5, 0.3], [0.8, 0.6, 0.4, 0.2]])
    semantic_membership = torch.ones_like(candidates)
    semantic_reliability = torch.ones_like(candidates)
    valid = torch.ones_like(candidates, dtype=torch.bool)
    spatial_shape = (2, 4, 2)
    core = torch.full(spatial_shape, 0.8)
    ring = torch.full(spatial_shape, 0.2)
    spatial_membership = torch.full(spatial_shape, 0.3)
    spatial_reliability = torch.ones(spatial_shape)
    entropy = torch.zeros((2, 4))
    spatial_valid = torch.ones(spatial_shape, dtype=torch.bool)
    return (
        base, candidates, semantic_membership, semantic_reliability, valid,
        core, ring, spatial_membership, spatial_reliability, entropy, spatial_valid,
    )


def run_routing(inputs, maximum_penalty):
    return fuse_query_conditioned_spatial_posterior(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0.05,
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10],
        ring_weight=1.0,
        contrast_temperature=0.05,
        maximum_penalty=maximum_penalty,
        core_membership=0.3,
        entropy_relaxation=0.75,
    )


def test_zero_penalty_is_exact_equal_token_hard_retrieval():
    output, stats = run_routing(routing_inputs(), 0.0)
    assert torch.allclose(output, torch.tensor([[0.7], [0.8]]))
    assert stats["semantic_mean_valid_slots"] == 4.0
    assert stats["mean_penalty"] == 0.0


def test_spatial_penalty_is_applied_once_after_semantic_selection():
    inputs = list(routing_inputs())
    inputs[7][:] = 0.15
    output, stats = run_routing(inputs, 0.06)
    assert torch.all(output < torch.tensor([[0.7], [0.8]]))
    assert stats["penalized_points"] == 2
    assert stats["semantic_mean_valid_slots"] == 4.0


def test_conformal_gate_protects_scores_stronger_than_the_spatial_anchor():
    scores = torch.tensor([0.8, 0.7, 0.6, 0.4, 0.3])
    support = torch.tensor([1.0, 0.98, 0.0, 0.0, 0.0])
    covered = torch.ones(5, dtype=torch.bool)
    gate, stats = conformal_spatial_gate(
        scores,
        support,
        covered,
        anchor_quantile=0.20,
        outside_quantile=0.75,
        temperature=0.02,
        use_null_expert=False,
    )
    assert gate[0] < gate[3]
    assert gate[1] < gate[4]
    assert stats["anchor_points"] == 2


def test_null_expert_disables_untrustworthy_spatial_anchor():
    scores = torch.tensor([0.5, 0.45, 0.8, 0.7])
    support = torch.tensor([1.0, 0.98, 0.0, 0.0])
    covered = torch.ones(4, dtype=torch.bool)
    gate, stats = conformal_spatial_gate(
        scores,
        support,
        covered,
        anchor_quantile=0.20,
        outside_quantile=0.75,
        temperature=0.02,
        use_null_expert=True,
    )
    assert gate.max() < 1e-3
    assert stats["null_expert_weight"] > 0.999


def test_semantic_mass_constraint_caps_an_overconfident_spatial_anchor():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.2])
    support = torch.tensor([1.0, 0.98, 0.0, 0.0, 0.0])
    covered = torch.ones(5, dtype=torch.bool)
    gate, stats = conformal_spatial_gate(
        scores,
        support,
        covered,
        anchor_quantile=0.20,
        outside_quantile=0.75,
        temperature=0.02,
        use_null_expert=False,
        semantic_preservation_quantile=0.75,
    )
    assert stats["anchor_score"] == pytest.approx(0.8)
    assert stats["raw_anchor_score"] > stats["anchor_score"]
    assert gate[1] == pytest.approx(0.5)


def completion_inputs():
    scores = torch.tensor([[0.90], [0.80], [0.45], [0.44], [0.43], [0.42]])
    candidates = scores.clone()
    semantic_membership = torch.ones_like(candidates)
    reliability = torch.ones_like(candidates)
    semantic_valid = torch.ones_like(candidates, dtype=torch.bool)
    group_ids = torch.full((6, 1, 2), -1, dtype=torch.long)
    group_ids[:, 0, 0] = 0
    group_confidence = torch.zeros((6, 1, 2))
    group_confidence[:, 0, 0] = 1.0
    spatial_membership = torch.zeros((6, 1, 2))
    spatial_membership[:, 0, 0] = 0.3
    spatial_valid = group_ids >= 0
    gaussian_atom_ids = torch.tensor([0, 0, 1, 1, 2, 3])
    atom_neighbor_ids = torch.tensor([[1, -1], [0, 2], [1, -1], [-1, -1]])
    atom_neighbor_weights = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    return (
        scores, candidates, semantic_membership, reliability, semantic_valid,
        group_ids, group_confidence, spatial_membership, spatial_valid,
        gaussian_atom_ids, atom_neighbor_ids, atom_neighbor_weights,
    )


def run_completion(inputs, minimum_seed_points=2, maximum_expansion_ratio=2.0):
    return complete_scores_from_seeded_groups(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0.05,
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
        core_membership=0.3,
        boundary_membership=0.05,
        seed_support=0.95,
        seed_quantile=0.75,
        seed_score_floor=0.55,
        target_quantile=0.20,
        semantic_delta=0.4,
        agreement_temperature=0.02,
        completion_strength=0.75,
        maximum_expansion_ratio=maximum_expansion_ratio,
        minimum_seed_points=minimum_seed_points,
        minimum_contact=0.05,
        maximum_hops=8,
    )


def test_seeded_group_completion_reaches_only_the_seed_connected_component():
    inputs = completion_inputs()
    output, stats = run_completion(inputs)
    assert output[2, 0] > inputs[0][2, 0]
    assert output[4, 0] > inputs[0][4, 0]
    assert output[5, 0] == inputs[0][5, 0]
    assert stats["certified_groups"] == 1
    assert stats["completed_points"] == 3


def test_seeded_group_completion_obeys_the_expansion_budget():
    inputs = completion_inputs()
    _, stats = run_completion(inputs, maximum_expansion_ratio=0.5)
    assert stats["completed_points"] == 1


def test_seeded_group_completion_exactly_falls_back_without_enough_seeds():
    inputs = completion_inputs()
    output, stats = run_completion(inputs, minimum_seed_points=3)
    assert torch.equal(output, inputs[0])
    assert stats["certified_groups"] == 0


def test_anisotropic_completion_blocks_cross_axis_diffusion():
    inputs = completion_inputs()
    atom_centroids = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 3.0, 0.0]]
    )
    axes = torch.eye(3).unsqueeze(0)
    ratios = torch.tensor([[1.0, 0.01, 0.01]])
    output, stats = complete_scores_from_seeded_groups(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0.05,
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
        core_membership=0.3,
        boundary_membership=0.05,
        seed_support=0.95,
        seed_quantile=0.75,
        seed_score_floor=0.55,
        target_quantile=0.20,
        semantic_delta=0.4,
        agreement_temperature=0.02,
        completion_strength=0.75,
        maximum_expansion_ratio=4.0,
        minimum_seed_points=2,
        minimum_contact=0.20,
        maximum_hops=8,
        atom_centroids=atom_centroids,
        group_principal_axes=axes,
        group_axis_ratios=ratios,
        anisotropic_axis_floor=0.05,
        anisotropic_budget_floor=0.25,
        anisotropic_semantic_floor=0.50,
    )
    assert output[2, 0] > inputs[0][2, 0]
    assert output[4, 0] == inputs[0][4, 0]
    assert stats["anisotropic_completion"] is True
    assert stats["mean_shape_budget_scale"] < 0.5


def test_token_profile_gate_preserves_relative_multiscale_shape():
    inputs = list(completion_inputs())
    inputs[0] = torch.tensor([[0.90], [0.88], [0.52], [0.54], [0.40], [0.30]])
    inputs[1] = torch.tensor(
        [
            [0.90, 0.80, 0.70, 0.60],
            [0.88, 0.78, 0.68, 0.58],
            [0.52, 0.42, 0.32, 0.22],
            [0.54, 0.10, 0.10, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.30, 0.20, 0.10, 0.00],
        ]
    )
    inputs[2] = torch.ones_like(inputs[1])
    inputs[3] = torch.ones_like(inputs[1])
    inputs[4] = torch.ones_like(inputs[1], dtype=torch.bool)
    inputs[5] = torch.zeros((6, 4, 2), dtype=torch.long)
    inputs[6] = torch.ones((6, 4, 2))
    inputs[7] = torch.full((6, 4, 2), 0.3)
    inputs[8] = torch.ones((6, 4, 2), dtype=torch.bool)
    output, stats = complete_scores_from_seeded_groups(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0.05,
        inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
        core_membership=0.3,
        boundary_membership=0.05,
        seed_support=0.95,
        seed_quantile=0.75,
        seed_score_floor=0.55,
        target_quantile=0.20,
        semantic_delta=0.4,
        agreement_temperature=0.02,
        completion_strength=0.75,
        maximum_expansion_ratio=4.0,
        minimum_seed_points=2,
        minimum_contact=0.05,
        maximum_hops=8,
        token_profile_gate=True,
        token_profile_quantile=0.90,
        token_profile_margin=0.03,
        token_profile_temperature=0.01,
        token_profile_minimum_slots=2,
    )
    assert output[2, 0] > inputs[0][2, 0]
    assert output[3, 0] == inputs[0][3, 0]
    assert stats["profile_certified_groups"] == 1
