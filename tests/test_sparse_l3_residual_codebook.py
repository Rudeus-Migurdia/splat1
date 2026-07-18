import json

import numpy as np
import torch

from build_sparse_l3_residual_codebook import (
    deterministic_top_fraction,
    residual_alphas,
    write_quantized_hypothesis,
)
from train_frozen_old_l2_gate import FrozenOldL2Gate, mix_experts, reliability_prior


def test_frozen_gate_stays_near_reliability_prior_and_masks_missing_expert():
    features = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=-1)
    reliability = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.4]])
    valid = torch.tensor([[True, True], [True, False], [False, True]])
    model = FrozenOldL2Gate(hidden_dim=4, max_logit_delta=0.5)
    weights, prior = model(features, reliability, valid)
    assert torch.allclose(weights, prior)
    assert weights[1].tolist() == [1.0, 0.0]
    assert weights[2].tolist() == [0.0, 1.0]
    mixed, effective = mix_experts(features, weights, valid)
    assert mixed.shape == (3, 8)
    assert torch.allclose(effective.sum(dim=1), torch.ones(3))


def test_reliability_prior_prefers_more_consistent_available_expert():
    reliability = torch.tensor([[0.9, 0.3], [0.1, 0.8]])
    valid = torch.ones_like(reliability, dtype=torch.bool)
    prior = reliability_prior(reliability, valid)
    assert prior[0, 0] > prior[0, 1]
    assert prior[1, 1] > prior[1, 0]
    assert torch.allclose(prior.sum(dim=1), torch.ones(2))


def test_sparse_selection_is_stable_capped_and_alpha_is_bounded():
    eligible = np.array([True, True, False, True, True])
    scores = np.array([0.4, 0.9, 1.0, 0.9, 0.1], dtype=np.float32)
    selected = deterministic_top_fraction(eligible, scores, maximum_count=2)
    assert selected.tolist() == [1, 3]
    alpha = residual_alphas(scores[selected], alpha_max=0.2)
    assert np.all(alpha >= 0.05)
    assert np.all(alpha <= 0.2)


def test_quantized_sparse_artifact_keeps_only_code_ids(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    codebook = np.eye(4, dtype=np.float32)
    np.save(root / "l3_codebook.npy", codebook.astype(np.float16))
    manifest = write_quantized_hypothesis(
        root / "sparse",
        root,
        point_ids=np.array([1, 4]),
        code_ids=np.array([2, 3]),
        alpha=np.array([0.1, 0.2], dtype=np.float32),
        codebook=codebook,
        num_gaussians=6,
        method="test",
        source="fixture",
    )
    payload = json.loads((root / "sparse" / "manifest.json").read_text())
    assert payload["representation"] == "sparse_quantized_semantic_hypothesis"
    assert payload["num_hypotheses"] == 2
    assert "features" not in payload
    assert np.load(root / "sparse" / "code_ids.npy").dtype == np.uint16
    assert manifest["maximum_alpha"] <= 0.2 + 1e-6
