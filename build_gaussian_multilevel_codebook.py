#!/usr/bin/env python
"""Build a large residual codebook whose integer IDs live on Gaussians."""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def l2_normalize(value, eps=1e-8):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


class NumpyFeatureSource:
    def __init__(self, feature_path, valid_mask_path=None):
        self.path = os.path.abspath(feature_path)
        self.features = np.load(self.path, mmap_mode="r")
        if self.features.ndim != 2:
            raise ValueError(f"Expected a 2D feature table, got {self.features.shape}")
        self.num_items, self.feature_dim = self.features.shape
        if valid_mask_path:
            valid_mask = np.load(valid_mask_path, mmap_mode="r")
            if valid_mask.shape != (self.num_items,):
                raise ValueError("Valid mask does not match the feature table")
            self.valid_mask = np.asarray(valid_mask, dtype=bool)
        else:
            self.valid_mask = np.zeros(self.num_items, dtype=bool)
            for start in range(0, self.num_items, 65536):
                end = min(start + 65536, self.num_items)
                values = np.asarray(self.features[start:end], dtype=np.float32)
                self.valid_mask[start:end] = np.linalg.norm(values, axis=1) > 1e-8

    def read(self, indices):
        return l2_normalize(np.asarray(self.features[indices], dtype=np.float32))

    def metadata(self):
        return {"type": "npy", "features": self.path}


class DrSplatPqFeatureSource:
    def __init__(self, checkpoint_path, pq_index_path):
        try:
            import faiss
            import torch
        except ImportError as exc:
            raise RuntimeError("Dr.Splat PQ input requires torch and faiss") from exc

        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.pq_index_path = os.path.abspath(pq_index_path)
        model_params, self.checkpoint_iteration = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )
        if len(model_params) != 13:
            raise ValueError(
                "Expected a 13-item Dr.Splat checkpoint with language PQ codes"
            )
        encoded = model_params[7].detach().to(torch.int16).cpu().numpy()
        invalid = np.all(encoded == -1, axis=-1) | np.all(encoded == 255, axis=-1)
        self.encoded = encoded.astype(np.uint8, copy=False)
        self.valid_mask = ~invalid
        self.num_items = int(encoded.shape[0])
        self.pq_index = faiss.read_index(self.pq_index_path)
        self.feature_dim = int(self.pq_index.d)

    def read(self, indices):
        encoded = np.ascontiguousarray(self.encoded[indices], dtype=np.uint8)
        decoded = self.pq_index.sa_decode(encoded).astype(np.float32, copy=False)
        return l2_normalize(decoded)

    def metadata(self):
        return {
            "type": "drsplat_pq",
            "checkpoint": self.checkpoint_path,
            "checkpoint_iteration": int(self.checkpoint_iteration),
            "pq_index": self.pq_index_path,
            "note": "Capacity diagnostic source only; the deployed artifact contains no PQ codes.",
        }


class ConsensusFeatureSource:
    def __init__(self, consensus_path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Consensus input requires torch") from exc
        self.path = os.path.abspath(consensus_path)
        payload = torch.load(self.path, map_location="cpu")
        self.features = payload["initial_features"].detach().cpu()
        weights = payload["total_weights"].detach().cpu()
        if self.features.ndim != 2 or weights.shape != (self.features.shape[0],):
            raise ValueError("Invalid semantic consensus artifact")
        self.num_items, self.feature_dim = self.features.shape
        self.valid_mask = weights.numpy() > 0
        capacity = payload.get("semantic_capacity")
        if capacity is None:
            self.id_capacity = None
        else:
            capacity = capacity.detach().cpu().numpy()
            if capacity.shape != (self.num_items,):
                raise ValueError("Semantic capacity must match the Gaussian count")
            self.id_capacity = np.asarray(capacity, dtype=np.int16)

    def read(self, indices):
        return l2_normalize(self.features[indices].float().numpy())

    def metadata(self):
        return {
            "type": "self_trained_2d_consensus",
            "consensus": self.path,
            "note": "Initialized directly from frozen 2D semantic observations, without a 3D teacher.",
        }


class SearchIndex:
    def __init__(self, centroids, spherical, faiss_index=None):
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.spherical = bool(spherical)
        self.faiss_index = faiss_index

    def search(self, features):
        features = np.ascontiguousarray(features, dtype=np.float32)
        if self.faiss_index is not None:
            _, ids = self.faiss_index.search(features, 1)
            return ids[:, 0].astype(np.int32, copy=False)
        if self.spherical:
            scores = features @ self.centroids.T
            return scores.argmax(axis=1).astype(np.int32)
        distances = (
            np.sum(features * features, axis=1, keepdims=True)
            + np.sum(self.centroids * self.centroids, axis=1)[None, :]
            - 2.0 * features @ self.centroids.T
        )
        return distances.argmin(axis=1).astype(np.int32)


class TorchSearchIndex:
    def __init__(self, centroids, spherical, device="cuda"):
        import torch

        self.torch = torch
        self.centroids = torch.from_numpy(
            np.asarray(centroids, dtype=np.float32)
        ).to(device)
        self.spherical = bool(spherical)
        self.device = device
        self.center_norms = (self.centroids * self.centroids).sum(dim=1)

    def search(self, features):
        torch = self.torch
        values = torch.from_numpy(
            np.ascontiguousarray(features, dtype=np.float32)
        ).to(self.device)
        if self.spherical:
            scores = values @ self.centroids.T
        else:
            scores = -(
                (values * values).sum(dim=1, keepdim=True)
                + self.center_norms.unsqueeze(0)
                - 2.0 * values @ self.centroids.T
            )
        return scores.argmax(dim=1).cpu().numpy().astype(np.int32, copy=False)


def torch_kmeans(features, num_codes, iterations, seed, spherical, batch_size=4096):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch GPU K-means requires torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Torch GPU K-means requested but CUDA is unavailable")

    features = np.asarray(features, dtype=np.float32)
    num_codes = min(int(num_codes), int(features.shape[0]))
    generator = np.random.default_rng(seed)
    initial_ids = generator.choice(features.shape[0], num_codes, replace=False)
    centers = torch.from_numpy(features[initial_ids].copy()).cuda()
    if spherical:
        centers = torch.nn.functional.normalize(centers, dim=-1)

    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(num_codes, dtype=torch.float32, device="cuda")
        center_norms = (centers * centers).sum(dim=1)
        for start in range(0, features.shape[0], batch_size):
            values = torch.from_numpy(
                np.ascontiguousarray(features[start : start + batch_size])
            ).cuda()
            if spherical:
                scores = values @ centers.T
            else:
                scores = -(
                    (values * values).sum(dim=1, keepdim=True)
                    + center_norms.unsqueeze(0)
                    - 2.0 * values @ centers.T
                )
            assignments = scores.argmax(dim=1)
            sums.index_add_(0, assignments, values)
            counts.index_add_(
                0,
                assignments,
                torch.ones(assignments.shape[0], dtype=torch.float32, device="cuda"),
            )
        nonempty = counts > 0
        next_centers = centers.clone()
        next_centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        if spherical:
            next_centers = torch.nn.functional.normalize(next_centers, dim=-1)
        shift = (next_centers - centers).square().mean()
        centers = next_centers
        if float(shift) < 1e-10:
            break
    centroids = centers.detach().cpu().numpy().astype(np.float32, copy=False)
    return centroids, TorchSearchIndex(centroids, spherical=spherical)


def numpy_kmeans(features, num_codes, iterations, seed, spherical):
    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=np.float32)
    if spherical:
        features = l2_normalize(features)
    num_codes = min(int(num_codes), int(features.shape[0]))
    centers = features[rng.choice(features.shape[0], num_codes, replace=False)].copy()
    if spherical:
        centers = l2_normalize(centers)

    assignments = np.full(features.shape[0], -1, dtype=np.int32)
    for _ in range(iterations):
        index = SearchIndex(centers, spherical=spherical)
        next_assignments = index.search(features)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        next_centers = np.zeros_like(centers)
        counts = np.bincount(assignments, minlength=num_codes)
        np.add.at(next_centers, assignments, features)
        for code_id in range(num_codes):
            if counts[code_id] > 0:
                next_centers[code_id] /= float(counts[code_id])
            else:
                next_centers[code_id] = features[
                    rng.integers(0, features.shape[0])
                ]
        centers = l2_normalize(next_centers) if spherical else next_centers
    return centers.astype(np.float32), SearchIndex(centers, spherical=spherical)


def faiss_kmeans(features, num_codes, iterations, seed, spherical, use_gpu):
    try:
        import faiss
    except ImportError:
        if use_gpu:
            return torch_kmeans(
                features,
                num_codes,
                iterations,
                seed,
                spherical,
            )
        return numpy_kmeans(features, num_codes, iterations, seed, spherical)

    features = np.ascontiguousarray(features, dtype=np.float32)
    num_codes = min(int(num_codes), int(features.shape[0]))
    if num_codes == features.shape[0]:
        centroids = features.copy()
        if spherical:
            centroids = l2_normalize(centroids)
        index = (
            faiss.IndexFlatIP(features.shape[1])
            if spherical
            else faiss.IndexFlatL2(features.shape[1])
        )
        index.add(centroids)
        return centroids, SearchIndex(centroids, spherical, index)

    effective_gpu = bool(use_gpu and hasattr(faiss, "StandardGpuResources"))
    if use_gpu and not effective_gpu:
        return torch_kmeans(
            features,
            num_codes,
            iterations,
            seed,
            spherical,
        )
    kmeans = faiss.Kmeans(
        features.shape[1],
        num_codes,
        niter=int(iterations),
        verbose=False,
        seed=int(seed),
        spherical=bool(spherical),
        gpu=effective_gpu,
        min_points_per_centroid=1,
        max_points_per_centroid=max(
            1,
            int(np.ceil(features.shape[0] / max(1, num_codes))),
        ),
    )
    kmeans.train(features)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    return centroids, SearchIndex(centroids, spherical, kmeans.index)


def reconstruct(features, code_ids, codebooks):
    result = np.zeros_like(features, dtype=np.float32)
    for level, codebook in enumerate(codebooks):
        result += codebook[code_ids[:, level]]
    return l2_normalize(result)


def compact_code_ids(code_ids, code_counts):
    max_codes = max(code_counts)
    if max_codes <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.uint32
    sentinel = int(np.iinfo(dtype).max)
    packed = np.full(code_ids.shape, sentinel, dtype=dtype)
    valid = code_ids >= 0
    packed[valid] = code_ids[valid].astype(dtype)
    return packed, sentinel


def build_codebook(source, args):
    valid_indices = np.flatnonzero(source.valid_mask)
    if valid_indices.size == 0:
        raise ValueError("Feature source contains no valid Gaussians")
    code_counts = [min(int(value), int(valid_indices.size)) for value in args.codes_per_level]
    if any(value <= 0 for value in code_counts):
        raise ValueError("All codebook levels must contain at least one code")

    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.train_samples), int(valid_indices.size))
    sample_indices = rng.choice(valid_indices, sample_count, replace=False)
    sample_targets = source.read(sample_indices)
    sample_reconstruction = np.zeros_like(sample_targets)
    point_code_ids = np.full(
        (source.num_items, len(code_counts)),
        -1,
        dtype=np.int32,
    )
    codebooks = []

    for level, code_count in enumerate(code_counts):
        sample_residual = sample_targets - sample_reconstruction
        spherical = level == 0
        if spherical:
            sample_residual = l2_normalize(sample_residual)
        centroids, index = faiss_kmeans(
            sample_residual,
            code_count,
            args.iterations,
            args.seed + level * 9973,
            spherical,
            args.faiss_gpu,
        )
        codebooks.append(centroids)

        for start in range(0, valid_indices.size, args.assignment_chunk):
            indices = valid_indices[start : start + args.assignment_chunk]
            targets = source.read(indices)
            previous = np.zeros_like(targets)
            for previous_level, previous_codebook in enumerate(codebooks[:-1]):
                previous += previous_codebook[point_code_ids[indices, previous_level]]
            residual = targets - previous
            if spherical:
                residual = l2_normalize(residual)
            point_code_ids[indices, level] = index.search(residual)

        sample_reconstruction += centroids[point_code_ids[sample_indices, level]]

    cosine_sum = 0.0
    cosine_square_sum = 0.0
    cosine_min = 1.0
    evaluated = 0
    for start in range(0, valid_indices.size, args.assignment_chunk):
        indices = valid_indices[start : start + args.assignment_chunk]
        targets = source.read(indices)
        decoded = reconstruct(targets, point_code_ids[indices], codebooks)
        cosine = np.sum(targets * decoded, axis=1)
        cosine_sum += float(cosine.sum())
        cosine_square_sum += float(np.square(cosine).sum())
        cosine_min = min(cosine_min, float(cosine.min()))
        evaluated += int(cosine.size)

    packed_ids, sentinel = compact_code_ids(point_code_ids, code_counts)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "point_code_ids.npy", packed_ids)
    np.save(output_dir / "valid_mask.npy", source.valid_mask.astype(np.bool_))
    codebook_files = []
    for level, codebook in enumerate(codebooks):
        name = f"codebook_level_{level}.npy"
        np.save(output_dir / name, codebook.astype(np.float16))
        codebook_files.append(name)

    codebook_bytes = sum(int(codebook.size * np.dtype(np.float16).itemsize) for codebook in codebooks)
    id_bytes = int(packed_ids.nbytes)
    mask_bytes = int(source.valid_mask.nbytes)
    full_fp16_bytes = int(source.num_items * source.feature_dim * 2)
    compact_bytes = codebook_bytes + id_bytes + mask_bytes
    cosine_mean = cosine_sum / max(1, evaluated)
    cosine_variance = cosine_square_sum / max(1, evaluated) - cosine_mean * cosine_mean
    manifest = {
        "format_version": 1,
        "representation": "gaussian_multilevel_residual_codebook",
        "feature_dim": int(source.feature_dim),
        "num_gaussians": int(source.num_items),
        "num_valid_gaussians": int(valid_indices.size),
        "valid_fraction": float(valid_indices.size / source.num_items),
        "levels": len(codebooks),
        "code_counts": [int(value) for value in code_counts],
        "codebook_files": codebook_files,
        "point_code_ids": "point_code_ids.npy",
        "valid_mask": "valid_mask.npy",
        "id_dtype": str(packed_ids.dtype),
        "invalid_id": sentinel,
        "mean_reconstruction_cosine": cosine_mean,
        "std_reconstruction_cosine": float(max(0.0, cosine_variance) ** 0.5),
        "min_reconstruction_cosine": cosine_min,
        "storage": {
            "codebook_bytes_fp16": codebook_bytes,
            "point_id_bytes": id_bytes,
            "valid_mask_bytes": mask_bytes,
            "total_semantic_bytes": compact_bytes,
            "full_per_gaussian_fp16_bytes": full_fp16_bytes,
            "compression_ratio_vs_512d_fp16": float(full_fp16_bytes / max(1, compact_bytes)),
            "bytes_per_gaussian_amortized": float(compact_bytes / source.num_items),
        },
        "source": source.metadata(),
        "args": vars(args),
    }
    with open(output_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


def main():
    parser = ArgumentParser(
        description="Build a large multi-level codebook with compact IDs stored per Gaussian."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="N x D .npy feature table")
    source.add_argument("--consensus", help="Direct 2D semantic consensus.pt cache")
    source.add_argument("--drsplat_checkpoint", help="Dr.Splat checkpoint used for a capacity diagnostic")
    parser.add_argument("--valid_mask", default=None)
    parser.add_argument("--pq_index", default=None)
    parser.add_argument("--codes_per_level", nargs="+", type=int, default=[8192, 8192])
    parser.add_argument("--train_samples", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--assignment_chunk", type=int, default=16384)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.train_samples <= 0 or args.iterations <= 0 or args.assignment_chunk <= 0:
        raise ValueError("Training samples, iterations, and assignment chunk must be positive")
    if args.drsplat_checkpoint:
        if not args.pq_index:
            raise ValueError("--drsplat_checkpoint requires --pq_index")
        feature_source = DrSplatPqFeatureSource(args.drsplat_checkpoint, args.pq_index)
    elif args.consensus:
        feature_source = ConsensusFeatureSource(args.consensus)
    else:
        feature_source = NumpyFeatureSource(args.features, args.valid_mask)
    build_codebook(feature_source, args)


if __name__ == "__main__":
    main()
