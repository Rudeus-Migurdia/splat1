import json
import importlib.util
from argparse import Namespace

import numpy as np
import pytest

from build_compact_group_hierarchy import (
    main as build_group_hierarchy,
    reestimate_group_features,
)
from build_gaussian_multilevel_codebook import NumpyFeatureSource, build_codebook
from build_gaussian_adaptive_codebook import assign_sparse_codes
from build_gaussian_multilevel_codebook import SearchIndex
from merge_multilevel_codebook_to_shared import merge_multilevel_artifact
from propagate_gaussian_codebook_coverage import load_source_mask, select_nearest_fill
from build_multiview_sam_source_mask import build_source_mask
from build_consensus_propagated_codebook import inverse_distance_weights, normalized_cosine
from build_multiview_mask_track_hierarchy import (
    aggregate_point_tracks,
    compute_node_importance,
)


def test_multilevel_codebook_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(8, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assignments = rng.integers(0, centers.shape[0], size=192)
    features = centers[assignments] + 0.08 * rng.normal(size=(192, 16))
    features = features.astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    valid = np.ones(features.shape[0], dtype=bool)
    valid[-5:] = False
    features[-5:] = 0.0

    feature_path = tmp_path / "features.npy"
    mask_path = tmp_path / "valid.npy"
    output_dir = tmp_path / "artifact"
    np.save(feature_path, features)
    np.save(mask_path, valid)
    args = Namespace(
        codes_per_level=[8, 8],
        train_samples=128,
        iterations=12,
        assignment_chunk=31,
        faiss_gpu=False,
        seed=3,
        output_dir=str(output_dir),
    )
    build_codebook(NumpyFeatureSource(feature_path, mask_path), args)

    with open(output_dir / "manifest.json") as source:
        manifest = json.load(source)
    point_ids = np.load(output_dir / "point_code_ids.npy")
    assert manifest["levels"] == 2
    assert manifest["code_counts"] == [8, 8]
    assert manifest["num_valid_gaussians"] == 187
    assert manifest["mean_reconstruction_cosine"] > 0.9
    assert point_ids.dtype == np.uint16
    assert np.all(point_ids[-5:] == manifest["invalid_id"])
    assert manifest["storage"]["total_semantic_bytes"] < manifest["storage"][
        "full_per_gaussian_fp16_bytes"
    ]

    if importlib.util.find_spec("torch") is None:
        return
    import torch
    from train_gaussian_multilevel_codebook import MultilevelGaussianCodebook

    model = MultilevelGaussianCodebook(output_dir, device="cpu")
    gaussian_ids = np.array([[0, 1, -1], [2, 3, 4]], dtype=np.int64)

    decoded = model(torch.from_numpy(gaussian_ids))
    assert decoded.shape == (2, 3, 16)
    assert torch.allclose(decoded[0, 2], torch.zeros(16))
    assert torch.allclose(decoded.norm(dim=-1)[gaussian_ids >= 0], torch.ones(5), atol=1e-5)


def test_query_kl_is_zero_for_matching_features():
    torch = pytest.importorskip("torch")
    from train_gaussian_multilevel_codebook import query_distribution_kl

    features = torch.nn.functional.normalize(torch.randn(12, 16), dim=-1)
    queries = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
    value = query_distribution_kl(features, features, queries, temperature=0.1)
    assert float(value) < 1e-6


def test_split_reliability_rewards_cross_view_agreement_and_balanced_support():
    torch = pytest.importorskip("torch")
    from build_split_consistency_fusion import split_reliability

    split_features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    split_weights = torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 1.0]])
    reliability, valid = split_reliability(split_features, split_weights)

    assert valid.tolist() == [True, True, True]
    assert reliability[0] == pytest.approx(1.0)
    assert reliability[1] == pytest.approx(0.0)
    assert 0.0 < float(reliability[2]) < 1.0


def test_confidence_gated_query_kl_downweights_ambiguous_targets():
    torch = pytest.importorskip("torch")
    from train_gaussian_multilevel_codebook import query_distribution_kl

    queries = torch.eye(3)
    prediction = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.51, 0.49, 0.0]])
    legacy = query_distribution_kl(prediction, target, queries, temperature=0.1)
    gated = query_distribution_kl(
        prediction,
        target,
        queries,
        temperature=0.1,
        confidence_power=2.0,
    )
    assert float(gated) > float(legacy)


def test_segment_contrastive_loss_rewards_the_matching_sam_segment():
    torch = pytest.importorskip("torch")
    from train_gaussian_multilevel_codebook import segment_contrastive_loss

    features = torch.eye(3)
    ids = torch.tensor([0, 2])
    matched = features[ids]
    swapped = features[torch.tensor([2, 0])]
    assert float(segment_contrastive_loss(matched, ids, features, 0.1)) < float(
        segment_contrastive_loss(swapped, ids, features, 0.1)
    )




def test_adaptive_shared_codebook_allocates_only_useful_extra_ids():
    codebook = np.eye(4, dtype=np.float32)
    targets = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    ids, weights, cosine = assign_sparse_codes(
        targets,
        codebook,
        SearchIndex(codebook, spherical=True),
        max_ids=3,
        min_gain=0.01,
        target_cosine=0.999,
    )

    assert (ids[0] >= 0).sum() == 2
    assert (ids[1] >= 0).sum() == 1
    assert weights[0, 1] > 0
    assert weights[1, 1] == 0
    assert cosine[0] > 0.999
    assert cosine[1] > 0.999


def test_adaptive_shared_codebook_respects_per_item_capacity_and_minimum_ids():
    codebook = np.eye(4, dtype=np.float32)
    targets = np.array(
        [[1.0, 0.6, 0.2, 0.0], [0.1, 1.0, 0.5, 0.0]], dtype=np.float32
    )
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    ids, _, _ = assign_sparse_codes(
        targets,
        codebook,
        SearchIndex(codebook, spherical=True),
        max_ids=3,
        min_gain=0.0,
        target_cosine=1.0,
        min_ids=2,
        max_ids_per_item=np.array([2, 3]),
        required_ids_per_item=np.array([2, 3]),
    )

    assert (ids[0] >= 0).sum() == 2
    assert (ids[1] >= 0).sum() == 3


def test_sparse_fine_selection_uses_fixed_fraction_of_unlabeled_candidates():
    torch = pytest.importorskip("torch")
    from build_uncertainty_capacity_fusion import select_sparse_fine_points

    score = torch.tensor([0.1, 0.8, 0.4, 0.9, 0.7])
    eligible = torch.tensor([True, True, False, True, True])
    selected, threshold = select_sparse_fine_points(score, eligible, 0.5)

    assert selected.tolist() == [False, True, False, True, False]
    assert threshold == pytest.approx(0.8)


def test_multilevel_artifact_merges_into_exact_unit_sum_shared_codebook(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "shared"
    source.mkdir()
    level0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    level1 = np.array([[0.1, 0.2], [-0.2, 0.1]], dtype=np.float16)
    point_ids = np.array([[0, 1], [1, 0], [65535, 65535]], dtype=np.uint16)
    valid = np.array([True, True, False])
    np.save(source / "level0.npy", level0)
    np.save(source / "level1.npy", level1)
    np.save(source / "ids.npy", point_ids)
    np.save(source / "valid.npy", valid)
    with open(source / "manifest.json", "w") as handle:
        json.dump(
            {
                "representation": "gaussian_multilevel_residual_codebook",
                "feature_dim": 2,
                "num_gaussians": 3,
                "levels": 2,
                "code_counts": [2, 2],
                "codebook_files": ["level0.npy", "level1.npy"],
                "point_code_ids": "ids.npy",
                "valid_mask": "valid.npy",
                "invalid_id": 65535,
            },
            handle,
        )

    manifest = merge_multilevel_artifact(source, output)
    merged = np.load(output / "codebook_shared.npy")
    merged_ids = np.load(output / "point_code_ids.npy")
    assert manifest["representation"] == "gaussian_adaptive_shared_codebook"
    assert manifest["weight_dtype"] == "implicit_unit"
    np.testing.assert_array_equal(merged_ids[:2], [[0, 3], [1, 2]])
    np.testing.assert_array_equal(merged_ids[2], [65535, 65535])
    np.testing.assert_allclose(merged[merged_ids[0]].sum(axis=0), level0[0] + level1[1])
    assert manifest["conversion"]["max_abs_reconstruction_error"] == 0.0


def test_coverage_propagation_selects_only_the_nearest_missing_points():
    distances = np.array([0.7, 0.1, 0.4, 0.2], dtype=np.float32)
    selected, radius = select_nearest_fill(
        distances,
        valid_count=4,
        total_count=10,
        target_coverage=0.6,
    )
    assert selected.tolist() == [False, True, False, True]
    assert radius == pytest.approx(0.2)


def test_coverage_source_gate_requires_matching_boolean_shape(tmp_path):
    path = tmp_path / "sources.npy"
    np.save(path, np.array([True, False, True]))
    assert load_source_mask(path, 3).tolist() == [True, False, True]
    with pytest.raises(ValueError, match="Source mask must have shape"):
        load_source_mask(path, 4)


def test_multiview_sam_source_mask_is_a_valid_codebook_subset(tmp_path):
    codebook = tmp_path / "codebook"
    hierarchy = tmp_path / "hierarchy"
    codebook.mkdir()
    hierarchy.mkdir()
    np.save(codebook / "valid.npy", np.array([True, True, False, True]))
    np.save(hierarchy / "ids.npy", np.array([[0], [9], [1], [2]], dtype=np.uint16))
    np.save(hierarchy / "weights.npy", np.array([[255], [0], [255], [4]], dtype=np.uint8))
    (codebook / "manifest.json").write_text(json.dumps({"valid_mask": "valid.npy"}))
    (hierarchy / "manifest.json").write_text(
        json.dumps(
            {
                "point_group_ids": "ids.npy",
                "point_group_weights": "weights.npy",
                "invalid_id": 9,
            }
        )
    )
    output = tmp_path / "sources.npy"
    result = build_source_mask(codebook, hierarchy, output, min_weight=1)
    assert np.load(output).tolist() == [True, False, False, True]
    assert result["source_count"] == 2


def test_local_consensus_helpers_prefer_nearer_source_and_measure_agreement():
    weights = inverse_distance_weights(
        np.array([0.1, 0.4], dtype=np.float32),
        np.array([0.3, 0.6], dtype=np.float32),
    )
    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 0] > weights[1, 1]
    cosine = normalized_cosine(
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(cosine, [1.0, 0.0])


def test_group_view_importance_respects_kl_and_point_contributions():
    features = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    supports = np.array([8.0, 2.0, 4.0], dtype=np.float32)
    tracks = np.array([0, 0, 1], dtype=np.int64)
    weights, metrics = compute_node_importance(
        features,
        supports,
        tracks,
        mode="information_kl",
        temperature=0.5,
        max_kl=0.05,
        ratio_clip=3.0,
        agreement_power=1.0,
        information_weight=1.0,
    )
    assert weights[:2].sum() == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)
    assert metrics["weighted_kl_target_to_behavior"] <= 0.05 + 1e-6
    assert metrics["max_importance_ratio"] <= 3.0 + 1e-6

    point_ids, point_scores = aggregate_point_tracks(
        num_gaussians=3,
        num_tracks=2,
        points=np.array([0, 0, 0, 1], dtype=np.int64),
        tracks=np.array([0, 0, 1, 1], dtype=np.int64),
        scores=np.array([0.2, 0.3, 0.4, 1.0], dtype=np.float32),
        top_m=2,
    )
    assert point_ids[0].tolist() == [0, 1]
    np.testing.assert_allclose(point_scores[0], [5.0 / 9.0, 4.0 / 9.0])
    assert point_ids[1, 0] == 1
    assert point_ids[2].tolist() == [-1, -1]


def test_compact_group_hierarchy_uses_small_ids_and_weights(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    group_features = rng.normal(size=(7, 16)).astype(np.float32)
    group_ids = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    group_scores = np.array(
        [
            [0.5, 0.3, 0.2, 0.1],
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    feature_path = tmp_path / "groups.npy"
    assignment_path = tmp_path / "assignments.npz"
    output_dir = tmp_path / "compact"
    np.save(feature_path, group_features)
    np.savez(
        assignment_path,
        top_group_ids=group_ids,
        top_group_scores=group_scores,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_compact_group_hierarchy.py",
            "--group_features",
            str(feature_path),
            "--assignments",
            str(assignment_path),
            "--top_m",
            "3",
            "--output_dir",
            str(output_dir),
        ],
    )
    build_group_hierarchy()

    with open(output_dir / "manifest.json") as source:
        manifest = json.load(source)
    packed_ids = np.load(output_dir / "point_group_ids.npy")
    packed_weights = np.load(output_dir / "point_group_weights.npy")
    assert packed_ids.dtype == np.uint16
    assert packed_weights.dtype == np.uint8
    assert packed_ids.shape == (3, 3)
    assert np.all(packed_ids[2] == manifest["invalid_id"])
    assert abs(int(packed_weights[0].sum()) - 255) <= 1
    assert manifest["covered_fraction"] == pytest.approx(2.0 / 3.0)


def test_group_tokens_can_be_reestimated_from_discrete_gaussians(tmp_path):
    codebook_dir = tmp_path / "codebook"
    codebook_dir.mkdir()
    level0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    level1 = np.zeros((1, 2), dtype=np.float16)
    point_ids = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.uint16)
    valid = np.ones(3, dtype=bool)
    np.save(codebook_dir / "level0.npy", level0)
    np.save(codebook_dir / "level1.npy", level1)
    np.save(codebook_dir / "ids.npy", point_ids)
    np.save(codebook_dir / "valid.npy", valid)
    with open(codebook_dir / "manifest.json", "w") as output:
        json.dump(
            {
                "codebook_files": ["level0.npy", "level1.npy"],
                "point_code_ids": "ids.npy",
                "valid_mask": "valid.npy",
            },
            output,
        )

    group_ids = np.array([[0], [0], [1]], dtype=np.int64)
    group_scores = np.ones((3, 1), dtype=np.float32)
    features, supported, _ = reestimate_group_features(
        str(codebook_dir), group_ids, group_scores, chunk_size=2
    )

    np.testing.assert_allclose(features[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(features[1], [0.0, 1.0], atol=1e-6)
    assert supported.tolist() == [True, True]
