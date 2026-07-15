#!/usr/bin/env python
"""Semantic-aware Gaussian association for multi-view mask tracks."""

import json
import os

import numpy as np
from scipy import sparse


def normalize_sparse_rows(matrix):
    matrix = matrix.tocsr().astype(np.float32, copy=False)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    scales = np.zeros_like(norms)
    nonzero = norms > 0.0
    scales[nonzero] = 1.0 / norms[nonzero]
    return sparse.diags(scales).dot(matrix).tocsr()


def aggregate_segment_signatures(cache, num_gaussians):
    """Accumulate every cached pixel contribution by (segment, Gaussian)."""
    point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
    point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
    segment_ids = cache["segment_ids"].numpy().astype(np.int64, copy=False)
    num_segments = int(cache["feature_latents"].shape[0])
    rows = np.broadcast_to(segment_ids[:, None], point_ids.shape)
    valid = (
        (rows >= 0)
        & (rows < num_segments)
        & (point_ids >= 0)
        & (point_ids < num_gaussians)
        & (point_weights > 0.0)
    )
    return sparse.coo_matrix(
        (point_weights[valid], (rows[valid], point_ids[valid])),
        shape=(num_segments, num_gaussians),
        dtype=np.float32,
    ).tocsr()


def prune_sparse_rows(matrix, max_nonzero):
    """Retain the largest values per CSR row without densifying the matrix."""
    matrix = matrix.tocsr()
    if max_nonzero <= 0:
        return matrix
    rows = []
    columns = []
    values = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        count = end - start
        if count <= 0:
            continue
        local = np.arange(start, end)
        if count > max_nonzero:
            local = local[
                np.argpartition(matrix.data[local], count - max_nonzero)[-max_nonzero:]
            ]
        rows.append(np.full(local.size, row, dtype=np.int64))
        columns.append(matrix.indices[local].astype(np.int64, copy=False))
        values.append(matrix.data[local].astype(np.float32, copy=False))
    if not rows:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    return sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=matrix.shape,
        dtype=np.float32,
    ).tocsr()


def semantic_geometry_union(
    signatures,
    segment_features,
    semantic_scorer,
    keep_fraction=0.2,
    max_candidates=2048,
):
    """Apply Proto-SaGa's union of geometry- and semantic-ranked candidates."""
    signatures = signatures.tocsr()
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    pool_positions = []
    pool_rows = []
    row_pool_offsets = np.zeros(signatures.shape[0] + 1, dtype=np.int64)
    for row in range(signatures.shape[0]):
        start, end = signatures.indptr[row : row + 2]
        positions = np.arange(start, end)
        if positions.size > max_candidates:
            positions = positions[
                np.argpartition(
                    signatures.data[positions], positions.size - max_candidates
                )[-max_candidates:]
            ]
        pool_positions.append(positions)
        pool_rows.append(np.full(positions.size, row, dtype=np.int64))
        row_pool_offsets[row + 1] = row_pool_offsets[row] + positions.size

    if row_pool_offsets[-1] == 0:
        return signatures.copy(), {
            "candidate_pairs": 0,
            "selected_pairs": 0,
            "geometry_pairs": 0,
            "semantic_pairs": 0,
            "semantic_rescued_pairs": 0,
        }

    pool_positions = np.concatenate(pool_positions)
    pool_rows = np.concatenate(pool_rows)
    pool_points = signatures.indices[pool_positions]
    semantic_scores = semantic_scorer.score(
        pool_points,
        np.asarray(segment_features, dtype=np.float32)[pool_rows],
        pool_rows,
    )

    output_rows = []
    output_columns = []
    output_values = []
    geometry_total = 0
    semantic_total = 0
    rescued_total = 0
    for row in range(signatures.shape[0]):
        pool_start, pool_end = row_pool_offsets[row : row + 2]
        count = pool_end - pool_start
        if count <= 0:
            continue
        source_positions = pool_positions[pool_start:pool_end]
        keep = max(1, int(np.ceil(keep_fraction * count)))
        geometry_local = np.argpartition(
            signatures.data[source_positions], count - keep
        )[-keep:]
        semantic_row = semantic_scores[pool_start:pool_end]
        finite = np.flatnonzero(np.isfinite(semantic_row))
        if finite.size > keep:
            semantic_local = finite[
                np.argpartition(semantic_row[finite], finite.size - keep)[-keep:]
            ]
        else:
            semantic_local = finite
        selected_local = np.union1d(geometry_local, semantic_local)
        semantic_only = np.setdiff1d(semantic_local, geometry_local, assume_unique=False)
        selected_positions = source_positions[selected_local]
        output_rows.append(np.full(selected_positions.size, row, dtype=np.int64))
        output_columns.append(signatures.indices[selected_positions].astype(np.int64))
        output_values.append(signatures.data[selected_positions].astype(np.float32))
        geometry_total += geometry_local.size
        semantic_total += semantic_local.size
        rescued_total += semantic_only.size

    selected = sparse.coo_matrix(
        (np.concatenate(output_values), (np.concatenate(output_rows), np.concatenate(output_columns))),
        shape=signatures.shape,
        dtype=np.float32,
    ).tocsr()
    return selected, {
        "candidate_pairs": int(pool_positions.size),
        "selected_pairs": int(selected.nnz),
        "geometry_pairs": int(geometry_total),
        "semantic_pairs": int(semantic_total),
        "semantic_rescued_pairs": int(rescued_total),
    }


class ResidualCodebookSemanticScorer:
    """Score Gaussian-mask compatibility from an existing discrete A6 codebook."""

    def __init__(self, artifact_dir, device="cuda", chunk_size=65536):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        artifact_dir = os.path.abspath(artifact_dir)
        with open(os.path.join(artifact_dir, "manifest.json")) as source:
            manifest = json.load(source)
        if manifest.get("representation") != "gaussian_multilevel_residual_codebook":
            raise ValueError("Semantic association requires a residual Gaussian codebook")
        self.invalid_id = int(manifest["invalid_id"])
        self.point_ids = np.load(
            os.path.join(artifact_dir, manifest["point_code_ids"]), mmap_mode="r"
        )
        self.valid_mask = np.load(
            os.path.join(artifact_dir, manifest["valid_mask"]), mmap_mode="r"
        )
        self.codebooks = [
            torch.from_numpy(np.load(os.path.join(artifact_dir, name)).astype(np.float32)).to(
                self.device
            )
            for name in manifest["codebook_files"]
        ]
        if self.point_ids.shape[1] != len(self.codebooks):
            raise ValueError("Code IDs and residual levels do not match")

    def score(self, point_ids, segment_features, segment_ids=None):
        torch = self.torch
        del segment_ids
        point_ids = np.asarray(point_ids, dtype=np.int64)
        segment_features = np.asarray(segment_features, dtype=np.float32)
        if segment_features.shape[0] != point_ids.size:
            raise ValueError("Each Gaussian candidate requires one segment feature")
        output = np.full(point_ids.shape, -np.inf, dtype=np.float32)
        for start in range(0, point_ids.size, self.chunk_size):
            end = min(start + self.chunk_size, point_ids.size)
            points = point_ids[start:end]
            valid = np.asarray(self.valid_mask[points], dtype=bool)
            ids = np.asarray(self.point_ids[points], dtype=np.int64)
            valid &= np.all(ids != self.invalid_id, axis=1)
            if not valid.any():
                continue
            selected = torch.from_numpy(ids[valid]).long().to(self.device)
            reconstruction = torch.zeros(
                (selected.shape[0], self.codebooks[0].shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            for level, codebook in enumerate(self.codebooks):
                reconstruction += codebook[selected[:, level]]
            reconstruction = torch.nn.functional.normalize(reconstruction, dim=-1)
            features = torch.from_numpy(segment_features[start:end][valid]).to(self.device)
            features = torch.nn.functional.normalize(features, dim=-1)
            scores = (reconstruction * features).sum(dim=-1).cpu().numpy()
            chunk = output[start:end]
            chunk[valid] = scores.astype(np.float32, copy=False)
        return output


class ViewClassifierSemanticScorer:
    """Proto-SaGa probability from a shared Gaussian latent and view classifiers."""

    def __init__(self, artifact_dir, device="cuda", chunk_size=65536):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        artifact_dir = os.path.abspath(artifact_dir)
        with open(os.path.join(artifact_dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != "view_specific_gaussian_classifier":
            raise ValueError("Unsupported view-classifier artifact")
        self.temperature = float(self.manifest["temperature"])
        self.view_offsets = np.asarray(self.manifest["view_offsets"], dtype=np.int64)
        self.gaussian_features = np.load(
            os.path.join(artifact_dir, self.manifest["gaussian_features"]),
            mmap_mode="r",
        )
        self.classifiers = torch.from_numpy(
            np.load(
                os.path.join(artifact_dir, self.manifest["view_classifiers"])
            ).astype(np.float32)
        ).to(self.device)
        self.current_view = None
        self.current_classifiers = None

    def set_view(self, view_index):
        view_index = int(view_index)
        start, end = self.view_offsets[view_index : view_index + 2]
        self.current_view = view_index
        self.current_classifiers = self.torch.nn.functional.normalize(
            self.classifiers[start:end], dim=-1
        )

    def score(self, point_ids, segment_features, segment_ids=None):
        del segment_features
        if self.current_classifiers is None or segment_ids is None:
            raise ValueError("View classifier scoring requires view and segment IDs")
        torch = self.torch
        point_ids = np.asarray(point_ids, dtype=np.int64)
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
        output = np.full(point_ids.shape, -np.inf, dtype=np.float32)
        for start in range(0, point_ids.size, self.chunk_size):
            end = min(start + self.chunk_size, point_ids.size)
            chunk_points = point_ids[start:end]
            unique_points, inverse = np.unique(chunk_points, return_inverse=True)
            features = torch.from_numpy(
                np.asarray(self.gaussian_features[unique_points], dtype=np.float32)
            ).to(self.device)
            features = torch.nn.functional.normalize(features, dim=-1)
            logits = self.temperature * features @ self.current_classifiers.T
            probabilities = torch.softmax(logits, dim=-1)
            inverse_tensor = torch.from_numpy(inverse).long().to(self.device)
            targets = torch.from_numpy(segment_ids[start:end]).long().to(self.device)
            valid = (targets >= 0) & (targets < probabilities.shape[1])
            if valid.any():
                values = probabilities[inverse_tensor[valid], targets[valid]]
                local = output[start:end]
                local[valid.cpu().numpy()] = values.cpu().numpy().astype(np.float32)
        return output
