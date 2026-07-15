import numpy as np
from scipy import sparse

from build_full_contribution_group_membership import (
    candidate_offsets,
    candidate_positions_for_points,
    binary_entropy,
    foreground_pairs,
    foreground_pairs_from_signatures,
    pack_top_memberships,
    reduce_sparse_pairs,
    selected_candidate_values,
    mutual_soft_overlap_matches,
    mutual_memory_matches,
    rofa_inlier_mask,
    segment_gaussian_signatures,
    segment_spatial_statistics,
)
from semantic_gaussian_association import (
    normalize_sparse_rows,
    prune_sparse_rows,
    semantic_geometry_union,
)
from combine_semantic_association_caches import combine_signatures


class TensorStub:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    @property
    def shape(self):
        return self.value.shape


def test_foreground_pairs_uses_all_ray_contributions():
    cache = {
        "point_ids": TensorStub([[0, 1], [0, 2]]),
        "point_weights": TensorStub([[0.6, 0.4], [0.3, 0.7]]),
        "segment_ids": TensorStub([0, 1]),
    }
    pairs, values = foreground_pairs(cache, np.array([0, 1]), num_tracks=2)
    assert pairs.tolist() == [0, 1, 2, 5]
    assert np.allclose(values, [0.6, 0.3, 0.4, 0.7])


def test_foreground_pairs_can_use_semantic_selected_signatures():
    signatures = sparse.csr_matrix(
        np.array([[0.6, 0.0, 0.4], [0.0, 0.7, 0.3]], dtype=np.float32)
    )
    pairs, values = foreground_pairs_from_signatures(
        signatures, np.array([1, 0]), num_tracks=2
    )
    assert pairs.tolist() == [1, 2, 4, 5]
    order = np.argsort(pairs)
    assert np.allclose(values[order], [0.6, 0.7, 0.3, 0.4])


def test_sparse_reduce_and_candidate_expansion():
    pairs, values = reduce_sparse_pairs([3, 1, 3], [0.2, 0.5, 0.4])
    assert pairs.tolist() == [1, 3]
    assert np.allclose(values, [0.5, 0.6])
    offsets = candidate_offsets(np.array([0, 0, 2]), num_gaussians=4)
    positions, rows = candidate_positions_for_points(np.array([0, 2, 3]), offsets)
    assert positions.tolist() == [0, 1, 2]
    assert rows.tolist() == [0, 0, 1]


def test_pack_top_memberships_preserves_absolute_probability():
    ids, scores = pack_top_memberships(
        num_gaussians=2,
        candidate_points=np.array([0, 0, 0, 1]),
        candidate_tracks=np.array([0, 1, 2, 1]),
        memberships=np.array([0.6, 0.9, 0.7, 0.4]),
        top_m=2,
        membership_threshold=0.5,
        min_foreground=0.0,
        foreground=np.ones(4),
    )
    assert ids[0].tolist() == [1, 2]
    assert np.allclose(scores[0], [0.9, 0.7])
    assert ids[1].tolist() == [-1, -1]


def test_binary_entropy_requires_two_views_and_peaks_at_disagreement():
    entropy = binary_entropy(
        np.array([0, 1, 2, 1]),
        np.array([1, 2, 2, 4]),
    )
    assert np.allclose(entropy[:3], [0.0, 1.0, 0.0], atol=1e-5)
    assert 0.0 < entropy[3] < 1.0


def test_selected_candidate_values_follow_packed_group_ids():
    values = selected_candidate_values(
        np.array([[4, 2], [-1, -1]]),
        np.array([0, 0, 1]),
        np.array([2, 4, 3]),
        np.array([0.2, 0.4, 0.8]),
        num_tracks=5,
    )
    assert np.allclose(values, [[0.4, 0.2], [0.0, 0.0]])


def test_segment_signatures_use_all_topk_contributions():
    cache = {
        "point_ids": TensorStub([[0, 1], [1, 2]]),
        "point_weights": TensorStub([[0.6, 0.4], [0.2, 0.8]]),
        "segment_ids": TensorStub([0, 1]),
        "feature_latents": TensorStub(np.zeros((2, 3))),
    }
    signatures = segment_gaussian_signatures(cache, num_gaussians=3).toarray()
    assert signatures[0, 1] > 0.0
    assert signatures[1, 2] > signatures[1, 1]
    assert np.allclose(np.linalg.norm(signatures, axis=1), 1.0)


def test_segment_spatial_statistics_use_weighted_3d_support():
    cache = {
        "point_ids": TensorStub([[0, 1], [1, 2]]),
        "point_weights": TensorStub([[0.75, 0.25], [0.5, 0.5]]),
        "segment_ids": TensorStub([0, 1]),
        "feature_latents": TensorStub(np.zeros((2, 3))),
    }
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    centers, radii = segment_spatial_statistics(cache, xyz)
    assert np.allclose(centers[:, 0], [0.5, 3.0])
    assert np.allclose(radii, [np.sqrt(0.75), 1.0])


def test_mutual_soft_overlap_accepts_only_reciprocal_matches():
    current = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    previous = np.array([[0.99, 0.01], [0.01, 0.99]], dtype=np.float32)
    current_features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    previous_features = current_features.copy()
    rows, nodes = mutual_soft_overlap_matches(
        sparse.csr_matrix(current),
        current_features,
        [
            {
                "signatures": sparse.csr_matrix(previous),
                "features": previous_features,
                "nodes": np.array([10, 11]),
            }
        ],
        min_overlap=0.05,
        min_semantic_similarity=0.8,
    )
    assert rows.tolist() == [0, 1]
    assert nodes.tolist() == [10, 11]


def test_global_memory_matching_recovers_a_track_after_a_view_gap():
    current = sparse.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    memory = sparse.csr_matrix(
        np.array([[0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    )
    rows, tracks = mutual_memory_matches(
        current,
        np.array([[1.0, 0.0]], dtype=np.float32),
        memory,
        np.eye(2, dtype=np.float32),
        min_overlap=0.05,
        min_semantic_similarity=0.8,
    )
    assert rows.tolist() == [0]
    assert tracks.tolist() == [0]


def test_spatial_mutual_matching_rejects_semantic_duplicate_far_away():
    current = sparse.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    previous = sparse.csr_matrix(
        np.array([[0.9, 0.1], [1.0, 0.0]], dtype=np.float32)
    )
    features = np.array([[1.0, 0.0]], dtype=np.float32)
    rows, nodes = mutual_soft_overlap_matches(
        current,
        features,
        [
            {
                "signatures": previous,
                "features": np.repeat(features, 2, axis=0),
                "nodes": np.array([10, 11]),
                "centers": np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                "radii": np.array([0.2, 0.2]),
            }
        ],
        min_overlap=0.05,
        min_semantic_similarity=0.8,
        current_centers=np.array([[0.0, 0.0, 0.0]]),
        current_radii=np.array([0.2]),
        max_spatial_distance_ratio=1.5,
    )
    assert rows.tolist() == [0]
    assert nodes.tolist() == [10]


def test_rofa_filters_a_track_level_semantic_outlier():
    features = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    keep, summary = rofa_inlier_mask(
        features,
        np.array([0, 0, 0, 0, 1]),
        tau=1.0,
    )
    assert keep.tolist() == [True, True, True, False, True]
    assert summary["num_removed"] == 1
    assert summary["num_affected_tracks"] == 1


def test_rofa_disabled_preserves_all_observations():
    keep, summary = rofa_inlier_mask(
        np.eye(3, dtype=np.float32),
        np.array([0, 0, 0]),
        tau=0.0,
    )
    assert keep.all()
    assert not summary["enabled"]


class SemanticScoreStub:
    def score(self, point_ids, segment_features, segment_ids=None):
        del segment_features
        del segment_ids
        scores = np.zeros(point_ids.shape, dtype=np.float32)
        scores[point_ids == 3] = 1.0
        return scores


def test_saga_union_keeps_geometry_and_semantic_candidates():
    signatures = sparse.csr_matrix(
        np.array([[0.9, 0.8, 0.7, 0.1]], dtype=np.float32)
    )
    selected, summary = semantic_geometry_union(
        signatures,
        np.array([[1.0, 0.0]], dtype=np.float32),
        SemanticScoreStub(),
        keep_fraction=0.25,
        max_candidates=4,
    )
    assert selected.indices.tolist() == [0, 3]
    assert summary["semantic_rescued_pairs"] == 1


def test_sparse_row_pruning_and_normalization_are_bounded():
    matrix = sparse.csr_matrix(
        np.array([[0.1, 0.9, 0.8], [0.0, 2.0, 0.0]], dtype=np.float32)
    )
    pruned = prune_sparse_rows(matrix, 2)
    assert pruned.getrow(0).nnz == 2
    normalized = normalize_sparse_rows(pruned).toarray()
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_class_and_instance_association_union_preserves_both_cues():
    first = sparse.csr_matrix(np.array([[0.8, 0.0, 0.3]], dtype=np.float32))
    second = sparse.csr_matrix(np.array([[0.0, 0.7, 0.3]], dtype=np.float32))
    combined = combine_signatures(first, second).toarray()
    assert np.allclose(combined, [[0.8, 0.7, 0.3]])
