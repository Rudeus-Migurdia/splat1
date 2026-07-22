import numpy as np
import pytest

torch = pytest.importorskip("torch")

from build_post_aggregation_filtered_consensus import (
    allocate_stratified_cluster_budget,
    match_projected_masks,
    project_cluster_labels,
    remap_hdbscan_labels,
    scaffold_size_diagnostics,
)


def test_allocate_stratified_cluster_budget_is_bounded_and_deterministic():
    allocation = allocate_stratified_cluster_budget([2, 4, 8], total_clusters=8)

    assert allocation.tolist() == [2, 2, 4]
    assert allocation.sum() == 8
    assert np.all(allocation <= np.array([2, 4, 8]))


def test_project_cluster_labels_sums_mass_instead_of_selecting_top_point():
    point_ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.int64)
    point_weights = torch.tensor([[0.40, 0.31, 0.30, 0.0]], dtype=torch.float32)
    point_clusters = torch.tensor([0, 1, 1], dtype=torch.int64)

    projected = project_cluster_labels(
        point_ids,
        point_weights,
        point_clusters,
        num_clusters=2,
        topk=4,
        chunk_size=1,
    )

    assert projected.tolist() == [1]


def test_project_cluster_labels_ignores_excluded_hdbscan_noise():
    point_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    point_weights = torch.tensor([[0.60, 0.21, 0.20]], dtype=torch.float32)
    point_clusters = torch.tensor([-1, 1, 1], dtype=torch.int64)

    projected = project_cluster_labels(
        point_ids,
        point_weights,
        point_clusters,
        num_clusters=2,
        topk=3,
        chunk_size=1,
    )

    assert projected.tolist() == [1]


def test_match_projected_masks_uses_dominant_cluster_iou_and_strict_gate():
    segmentation = np.array(
        [
            [5, 5, -1, 9],
            [5, 5, -1, 9],
            [-1, -1, -1, 9],
            [-1, -1, -1, 9],
        ],
        dtype=np.int32,
    )
    projected = np.array(
        [
            [2, 2, -1, 3],
            [2, 2, -1, 3],
            [-1, -1, -1, 3],
            [-1, -1, 3, 3],
        ],
        dtype=np.int32,
    )

    filtered, records = match_projected_masks(
        segmentation, projected, iou_threshold=0.8
    )
    by_segment = {record["segment_id"]: record for record in records}

    assert by_segment[5]["iou"] == 1.0
    assert by_segment[5]["kept"]
    assert by_segment[9]["iou"] == 0.8
    assert not by_segment[9]["kept"]
    assert np.all(filtered[segmentation == 5] == 5)
    assert np.all(filtered[segmentation == 9] == -1)


def test_match_projected_masks_rejects_unrendered_segment():
    segmentation = np.array([[7, 7], [-1, -1]], dtype=np.int32)
    projected = np.full_like(segmentation, -1)

    filtered, records = match_projected_masks(segmentation, projected, 0.8)

    assert records == [
        {
            "segment_id": 7,
            "cluster_id": -1,
            "intersection": 0,
            "segment_area": 2,
            "cluster_area": 0,
            "iou": 0.0,
            "kept": False,
        }
    ]
    assert np.all(filtered == -1)


def test_remap_hdbscan_labels_reserves_zero_for_noise():
    mapped, mapping = remap_hdbscan_labels(np.array([-1, 7, 3, 7, -1]))

    assert mapped.tolist() == [0, 2, 1, 2, 0]
    assert mapping == {3: 1, 7: 2}


def test_remap_hdbscan_labels_can_exclude_noise():
    mapped, mapping = remap_hdbscan_labels(
        np.array([-1, 7, 3, 7, -1]), noise_policy="exclude"
    )

    assert mapped.tolist() == [-1, 1, 0, 1, -1]
    assert mapping == {3: 0, 7: 1}


def test_scaffold_size_diagnostics_counts_codebook_and_gaussians():
    diagnostics = scaffold_size_diagnostics(
        np.array([0, 1, 1], dtype=np.int32),
        np.array([0, 1, 1, 2, 2, 2], dtype=np.int64),
        num_clusters=2,
    )

    assert diagnostics["codebook_cluster_size_quantiles"]["0.0"] == 1.0
    assert diagnostics["codebook_cluster_size_quantiles"]["1.0"] == 2.0
    assert diagnostics["gaussian_cluster_size_quantiles"]["0.0"] == 1.0
    assert diagnostics["gaussian_cluster_size_quantiles"]["1.0"] == 5.0
    assert diagnostics["empty_projected_clusters"] == 0
