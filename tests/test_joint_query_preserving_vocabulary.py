import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from train_joint_query_preserving_vocabulary import (  # noqa: E402
    FixedSharedAssignment,
    query_rank_losses,
    reconstruct_fixed_assignment,
)


def test_sparse_assignment_loading_and_reconstruction(tmp_path):
    manifest = {
        "representation": "gaussian_adaptive_shared_codebook",
        "storage_layout": "base_plus_sparse_overflow",
        "num_gaussians": 3,
        "feature_dim": 2,
        "num_codes": 3,
        "id_slots": 2,
        "invalid_id": 65535,
        "point_code_ids": "point_code_ids.npy",
        "valid_mask": "valid_mask.npy",
        "overflow_point_ids": "overflow_point_ids.npy",
        "overflow_code_ids": "overflow_code_ids.npy",
        "overflow_slots": "overflow_slots.npy",
        "overflow_weights": "overflow_weights.npy",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "point_code_ids.npy", np.array([0, 1, 65535], np.uint16))
    np.save(tmp_path / "valid_mask.npy", np.array([True, True, False]))
    np.save(tmp_path / "overflow_point_ids.npy", np.array([0], np.uint32))
    np.save(tmp_path / "overflow_code_ids.npy", np.array([2], np.uint16))
    np.save(tmp_path / "overflow_slots.npy", np.array([1], np.uint8))
    np.save(tmp_path / "overflow_weights.npy", np.array([128], np.uint8))

    assignment = FixedSharedAssignment(tmp_path)
    ids, weights = assignment.batch(np.array([0, 1]), "cpu")
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    reconstruction = reconstruct_fixed_assignment(codebook, ids, weights)
    expected = torch.nn.functional.normalize(
        torch.tensor([[1.0, 128.0 / 255.0], [0.0, 1.0]]), dim=-1
    )
    assert torch.allclose(reconstruction, expected)


def test_query_rank_losses_are_zero_for_identical_features():
    query_bank = torch.eye(3)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    kl, margin = query_rank_losses(target, target, query_bank, 0.1)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert margin.item() == pytest.approx(0.0, abs=1e-6)


def test_query_margin_penalizes_reversed_ranking():
    query_bank = torch.eye(3)
    target = torch.tensor([[0.9, 0.4, 0.0]])
    prediction = torch.tensor([[0.4, 0.9, 0.0]])
    _, margin = query_rank_losses(prediction, target, query_bank, 0.1)
    assert margin.item() > 0.0
