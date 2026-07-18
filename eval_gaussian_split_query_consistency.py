#!/usr/bin/env python
"""Compare Gaussian query consistency against two independent training-view splits."""

import json
import os
import sys
from argparse import ArgumentParser

import faiss
import numpy as np
import torch
from torch.nn import functional as F

from eval_lerf_ovs_miou import load_lerf_labels
from eval_lerf_ovs_gaussian_codebook_miou import GaussianCodebookArtifact
from evaluation.openclip_encoder import OpenCLIPNetwork
from train_view_invariant_semantic_atoms import decode_group_features


def reconstruct_sampled_codebook(artifact_dir, sample_ids):
    artifact = GaussianCodebookArtifact(artifact_dir)
    ids = np.asarray(artifact.point_code_ids[sample_ids], dtype=np.int64)
    valid = ids != artifact.invalid_id
    safe = np.where(valid, ids, 0)
    ids_t = torch.from_numpy(safe).long().to("cuda")
    valid_t = torch.from_numpy(valid).to("cuda")
    if artifact.shared_codebook:
        if artifact.point_code_weights is None:
            weights = valid.astype(np.float32)
        else:
            weights = np.asarray(
                artifact.point_code_weights[sample_ids], dtype=np.float32
            ) / 255.0
        weights_t = torch.from_numpy(weights).to("cuda")
        features = (
            artifact.codebooks[0][ids_t]
            * weights_t.unsqueeze(-1)
            * valid_t.unsqueeze(-1)
        ).sum(dim=1)
    else:
        features = torch.zeros(
            (sample_ids.size, artifact.feature_dim),
            dtype=torch.float32,
            device="cuda",
        )
        for level, codebook in enumerate(artifact.codebooks):
            features += codebook[ids_t[:, level]]
    features = F.normalize(features, dim=-1).cpu().numpy().astype(np.float32)
    return features, valid.any(axis=1), artifact.manifest


def query_scores(clip_model, features, num_categories, device, chunk_size):
    features = np.asarray(features, dtype=np.float32)
    result = np.zeros((features.shape[0], num_categories), dtype=np.float32)
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        values = torch.from_numpy(features[start:end]).to(device)
        values = F.normalize(values, dim=-1)
        result[start:end] = torch.cat(
            [
                clip_model.get_activation(values, category).float()
                for category in range(num_categories)
            ],
            dim=1,
        ).cpu().numpy()
    return result


def load_group_sample(artifact_dir, sample_ids):
    with open(os.path.join(artifact_dir, "manifest.json")) as source:
        manifest = json.load(source)
    semantic_ids = np.load(
        os.path.join(artifact_dir, manifest["group_semantic_code_ids"])
    ).astype(np.int64)
    if manifest.get("representation") == "hierarchical_independent_group_codebooks":
        levels = np.load(
            os.path.join(artifact_dir, manifest["group_level"])
        ).astype(np.int64)
        if semantic_ids.ndim != 2 or semantic_ids.shape[1] != 1:
            raise ValueError("Hierarchical semantic IDs must have shape [G, 1]")
        features = np.zeros(
            (semantic_ids.shape[0], int(manifest["feature_dim"])), dtype=np.float32
        )
        invalid_semantic = int(manifest["semantic_invalid_id"])
        for spec in manifest["level_codebooks"]:
            level = int(spec["level"])
            mask = levels == level
            codebook = np.load(
                os.path.join(artifact_dir, spec["codebook"])
            ).astype(np.float32)
            local_ids = semantic_ids[mask, 0]
            if (local_ids == invalid_semantic).any():
                raise ValueError("Hierarchical token has no local semantic ID")
            features[mask] = codebook[local_ids]
        features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
        atom_features = np.zeros_like(features)
    else:
        vocabulary = np.load(
            os.path.join(artifact_dir, manifest["group_codebook"])
        ).astype(np.float32)
        features = decode_group_features(
            vocabulary, semantic_ids, int(manifest["semantic_invalid_id"])
        )
        atom_name = manifest.get("group_semantic_atom_code_ids")
        atom_features = (
            decode_group_features(
                vocabulary,
                np.load(os.path.join(artifact_dir, atom_name)).astype(np.int64),
                int(manifest.get("semantic_atom_invalid_id", manifest["semantic_invalid_id"])),
            )
            if atom_name
            else np.zeros_like(features)
        )
    ids = np.load(
        os.path.join(artifact_dir, manifest["point_group_ids"]), mmap_mode="r"
    )[sample_ids].astype(np.int64)
    invalid = int(manifest["invalid_id"])
    ids[ids == invalid] = -1
    weights = np.load(
        os.path.join(artifact_dir, manifest["point_group_weights"]), mmap_mode="r"
    )[sample_ids].astype(np.float32) / 255.0
    reliability_table = np.load(
        os.path.join(artifact_dir, manifest["group_reliability"])
    ).astype(np.float32)
    safe = np.where(ids >= 0, ids, 0)
    reliability = reliability_table[safe]
    point_reliability_name = manifest.get("point_group_reliability")
    if point_reliability_name:
        point_reliability = np.load(
            os.path.join(artifact_dir, point_reliability_name), mmap_mode="r"
        )[sample_ids].astype(np.float32)
        if point_reliability.shape != ids.shape:
            raise ValueError("Point reliability must match group IDs")
        reliability *= point_reliability
    reliability[ids < 0] = 0.0
    level_name = manifest.get("group_level")
    levels = (
        np.load(os.path.join(artifact_dir, level_name), mmap_mode="r").astype(np.int64)[safe]
        if level_name
        else np.zeros_like(ids, dtype=np.int64)
    )
    levels[ids < 0] = -1
    competitor_name = manifest.get("point_competitor_ids")
    if competitor_name:
        competitor = np.load(
            os.path.join(artifact_dir, competitor_name), mmap_mode="r"
        )[sample_ids].astype(np.int64)
        competitor[competitor == int(manifest.get("competitor_invalid_id", invalid))] = -1
    else:
        competitor = np.full_like(ids, -1)
    return {
        "manifest": manifest,
        "features": features,
        "atom_features": atom_features,
        "ids": ids,
        "weights": weights,
        "reliability": reliability,
        "levels": levels,
        "competitor": competitor,
    }


def group_readout(
    base_scores,
    group_scores,
    group,
    contrastive=False,
    atom_scores=None,
):
    safe = np.where(group["ids"] >= 0, group["ids"], 0)
    positive = group_scores[safe]
    valid = group["ids"] >= 0
    positive_gain = np.maximum(positive - base_scores[:, None, :], 0.0)
    if contrastive:
        competitor_safe = np.where(group["competitor"] >= 0, group["competitor"], 0)
        competitor = group_scores[competitor_safe]
        competitor_valid = group["competitor"] >= 0
        competing_gain = np.maximum(competitor - base_scores[:, None, :], 0.0)
        competing_gain *= competitor_valid[..., None]
        positive_gain = np.maximum(positive_gain - competing_gain, 0.0)
    if atom_scores is not None:
        atom_positive = atom_scores[safe]
        atom_gain = np.maximum(atom_positive - base_scores[:, None, :], 0.0)
        positive_gain = np.minimum(positive_gain, atom_gain)
    gate = group["weights"] * group["reliability"] * valid
    return base_scores + np.max(positive_gain * gate[..., None], axis=1)


def hierarchical_memory_readout(base_scores, group_scores, group, temperature=0.10):
    """Numpy counterpart of the evaluator's query-aware L0--L3 fusion."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    safe = np.where(group["ids"] >= 0, group["ids"], 0)
    candidates = group_scores[safe]
    valid = group["ids"] >= 0
    gates = group["weights"] * group["reliability"] * valid
    selectable = valid & (gates > 0.0)
    logits = candidates / temperature + np.log(np.maximum(gates[..., None], 1e-8))
    logits = np.where(selectable[..., None], logits, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    covered = selectable.any(axis=1)
    maximum[~covered] = 0.0
    weights = np.exp(logits - maximum)
    weights[~covered] = 0.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    fused = (weights * candidates).sum(axis=1)
    confidence = (weights * gates[..., None]).sum(axis=1)
    return np.where(
        covered[:, None],
        base_scores + confidence * (fused - base_scores),
        base_scores,
    )


def calibrated_hierarchical_memory_readout(
    base_scores,
    group_scores,
    group,
    temperature=0.10,
    margin_threshold=0.25,
    margin_temperature=0.10,
):
    """Numpy equivalent of the A27 calibrated peer-token reader."""
    if temperature <= 0.0 or margin_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if margin_threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")
    safe = np.where(group["ids"] >= 0, group["ids"], 0)
    candidates = group_scores[safe]
    valid = group["ids"] >= 0
    gates = group["weights"] * group["reliability"] * valid
    selectable = valid & (gates > 0.0)
    covered = selectable.any(axis=1)
    calibrated = np.zeros_like(candidates)
    for level in np.unique(group["levels"][selectable]):
        if level < 0:
            continue
        mask = selectable & (group["levels"] == level)
        values = candidates[mask]
        mean = values.mean(axis=0, keepdims=True)
        std = np.maximum(values.std(axis=0, keepdims=True), 1e-4)
        calibrated[mask] = (values - mean) / std
    logits = calibrated / temperature + np.log(np.maximum(gates[..., None], 1e-8))
    logits = np.where(selectable[..., None], logits, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    maximum[~covered] = 0.0
    weights = np.exp(logits - maximum)
    weights[~covered] = 0.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    fused = (weights * candidates).sum(axis=1)
    reliability = (weights * gates[..., None]).sum(axis=1)
    calibrated = np.where(selectable[..., None], calibrated, -np.inf)
    if calibrated.shape[1] == 1:
        margin = np.full(candidates.shape[:1] + candidates.shape[2:], np.inf, dtype=np.float32)
    else:
        top = np.sort(calibrated, axis=1)[:, -2:]
        margin = top[:, 1] - top[:, 0]
    valid_count = selectable.sum(axis=1)
    margin[valid_count < 2] = np.inf
    margin_gate = 1.0 / (1.0 + np.exp(
        -(margin - margin_threshold) / margin_temperature
    ))
    confidence = reliability * margin_gate
    return np.where(
        covered[:, None],
        base_scores + confidence * (fused - base_scores),
        base_scores,
    )


def decode_pq_sample(checkpoint, pq_index_path, sample_ids):
    payload = torch.load(checkpoint, map_location="cpu")[0]
    encoded = payload[7][sample_ids].numpy().astype(np.int16, copy=False)
    valid = ~(
        np.all(encoded == -1, axis=-1) | np.all(encoded == 255, axis=-1)
    )
    output = np.zeros((sample_ids.size, 512), dtype=np.float32)
    if valid.any():
        index = faiss.read_index(pq_index_path)
        output[valid] = index.sa_decode(encoded[valid].astype(np.uint8, copy=False))
        output[valid] /= np.maximum(
            np.linalg.norm(output[valid], axis=-1, keepdims=True), 1e-8
        )
    return output, valid


def distribution(scores):
    scores = np.maximum(scores, 1e-8)
    return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-8)


def symmetric_kl(first, second):
    first = np.maximum(first, 1e-8)
    second = np.maximum(second, 1e-8)
    return 0.5 * np.sum(
        first * (np.log(first) - np.log(second))
        + second * (np.log(second) - np.log(first)),
        axis=1,
    )


def compare(canonical, first, second, valid):
    valid = np.asarray(valid, dtype=bool)
    canonical = distribution(canonical[valid])
    first = distribution(first[valid])
    second = distribution(second[valid])
    kl = 0.5 * (symmetric_kl(canonical, first) + symmetric_kl(canonical, second))
    flips = 0.5 * (
        (canonical.argmax(1) != first.argmax(1)).astype(np.float32)
        + (canonical.argmax(1) != second.argmax(1)).astype(np.float32)
    )
    return {
        "num_samples": int(valid.sum()),
        "canonical_split_symmetric_kl": float(kl.mean()),
        "canonical_split_top1_flip_rate": float(flips.mean()),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--pq_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--a14_base_dir", required=True)
    parser.add_argument("--a14_candidate_dir", required=True)
    parser.add_argument("--a20_group_dir", required=True)
    parser.add_argument("--a21_group_dir", required=True)
    parser.add_argument("--a22_group_dir", default=None)
    parser.add_argument("--a23_group_dir", default=None)
    parser.add_argument("--a24_group_dir", default=None)
    parser.add_argument("--a26_group_dir", default=None)
    parser.add_argument("--a27_group_dir", default=None)
    parser.add_argument("--a27_group_query_temperature", type=float, default=0.10)
    parser.add_argument("--a27_level_margin_threshold", type=float, default=0.25)
    parser.add_argument("--a27_level_margin_temperature", type=float, default=0.10)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:])
    if args.samples <= 0 or args.chunk_size <= 0:
        raise ValueError("Sampling parameters must be positive")
    if args.a27_group_query_temperature <= 0.0 or args.a27_level_margin_temperature <= 0.0:
        raise ValueError("A27 temperatures must be positive")
    if args.a27_level_margin_threshold < 0.0:
        raise ValueError("A27 level margin threshold must be non-negative")

    fine = torch.load(args.fine_consensus, map_location="cpu")
    split = F.normalize(fine["split_initial_features"].float(), dim=-1)
    split_weights = fine["split_weights"]
    supported = (split_weights > 0.0).all(dim=0) & (split.norm(dim=-1) > 0.0).all(dim=0)
    supported_ids = torch.nonzero(supported, as_tuple=False).squeeze(1).numpy()
    rng = np.random.default_rng(args.seed)
    if supported_ids.size > args.samples:
        sample_ids = np.sort(rng.choice(supported_ids, args.samples, replace=False))
    else:
        sample_ids = supported_ids
    split_np = split[:, sample_ids].numpy().astype(np.float32)
    del fine, split, split_weights, supported

    _, categories = load_lerf_labels(args.label_dir)
    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(categories)
    first_scores = query_scores(
        clip_model, split_np[0], len(categories), device, args.chunk_size
    )
    second_scores = query_scores(
        clip_model, split_np[1], len(categories), device, args.chunk_size
    )

    pq_features, pq_valid = decode_pq_sample(
        args.pq_checkpoint, args.pq_index, sample_ids
    )
    pq_scores = query_scores(
        clip_model, pq_features, len(categories), device, args.chunk_size
    )
    base_features, base_valid, base_manifest = reconstruct_sampled_codebook(
        args.a14_base_dir, sample_ids
    )
    candidate_features, candidate_valid, candidate_manifest = reconstruct_sampled_codebook(
        args.a14_candidate_dir, sample_ids
    )
    base_scores = query_scores(
        clip_model, base_features, len(categories), device, args.chunk_size
    )
    candidate_scores = query_scores(
        clip_model, candidate_features, len(categories), device, args.chunk_size
    )
    a14_scores = np.maximum(base_scores, candidate_scores)

    a20 = load_group_sample(args.a20_group_dir, sample_ids)
    a20_group_scores = query_scores(
        clip_model, a20["features"], len(categories), device, args.chunk_size
    )
    a20_scores = group_readout(a14_scores, a20_group_scores, a20)
    a21 = load_group_sample(args.a21_group_dir, sample_ids)
    a21_group_scores = query_scores(
        clip_model, a21["features"], len(categories), device, args.chunk_size
    )
    a21_atom_scores = group_readout(a14_scores, a21_group_scores, a21)
    a21_contrastive_scores = group_readout(
        a14_scores, a21_group_scores, a21, contrastive=True
    )
    a22_dual_scores = None
    a22_dual_contrastive_scores = None
    if args.a22_group_dir:
        a22 = load_group_sample(args.a22_group_dir, sample_ids)
        a22_group_scores = query_scores(
            clip_model, a22["features"], len(categories), device, args.chunk_size
        )
        a22_atom_scores = query_scores(
            clip_model, a22["atom_features"], len(categories), device, args.chunk_size
        )
        a22_dual_scores = group_readout(
            a14_scores, a22_group_scores, a22, atom_scores=a22_atom_scores
        )
        a22_dual_contrastive_scores = group_readout(
            a14_scores,
            a22_group_scores,
            a22,
            contrastive=True,
            atom_scores=a22_atom_scores,
        )
    common_valid = base_valid & candidate_valid & pq_valid
    results = {
        "protocol": "gaussian_split_query_consistency",
        "categories": categories,
        "num_requested_samples": args.samples,
        "num_two_split_supported": int(supported_ids.size),
        "num_sampled": int(sample_ids.size),
        "split_split": {
            "symmetric_kl": float(
                symmetric_kl(distribution(first_scores), distribution(second_scores)).mean()
            ),
            "top1_flip_rate": float(
                (first_scores.argmax(1) != second_scores.argmax(1)).mean()
            ),
        },
        "representations": {
            "drsplat_pq_baseline": compare(
                pq_scores, first_scores, second_scores, common_valid
            ),
            "a14": compare(a14_scores, first_scores, second_scores, common_valid),
            "a20": compare(a20_scores, first_scores, second_scores, common_valid),
            "a21_atom_only": compare(
                a21_atom_scores, first_scores, second_scores, common_valid
            ),
            "a21_contrastive": compare(
                a21_contrastive_scores, first_scores, second_scores, common_valid
            ),
        },
        "source": {
            "fine_consensus": os.path.abspath(args.fine_consensus),
            "pq_checkpoint": os.path.abspath(args.pq_checkpoint),
            "a14_base_manifest": base_manifest,
            "a14_candidate_manifest": candidate_manifest,
            "a20_group_dir": os.path.abspath(args.a20_group_dir),
            "a21_group_dir": os.path.abspath(args.a21_group_dir),
            "leakage_control": "training-view split features only; labels provide evaluation query names",
        },
    }
    if a22_dual_scores is not None:
        results["representations"]["a22_dual_agreement"] = compare(
            a22_dual_scores, first_scores, second_scores, common_valid
        )
        results["representations"]["a22_dual_contrastive"] = compare(
            a22_dual_contrastive_scores,
            first_scores,
            second_scores,
            common_valid,
        )
        results["source"]["a22_group_dir"] = os.path.abspath(args.a22_group_dir)
    if args.a23_group_dir:
        a23 = load_group_sample(args.a23_group_dir, sample_ids)
        a23_group_scores = query_scores(
            clip_model, a23["features"], len(categories), device, args.chunk_size
        )
        a23_scores = group_readout(a14_scores, a23_group_scores, a23)
        results["representations"]["a23_signed_membership"] = compare(
            a23_scores, first_scores, second_scores, common_valid
        )
        results["source"]["a23_group_dir"] = os.path.abspath(args.a23_group_dir)
    if args.a24_group_dir:
        a24 = load_group_sample(args.a24_group_dir, sample_ids)
        a24_group_scores = query_scores(
            clip_model, a24["features"], len(categories), device, args.chunk_size
        )
        a24_scores = group_readout(a14_scores, a24_group_scores, a24)
        results["representations"]["a24_multiscale_micro"] = compare(
            a24_scores, first_scores, second_scores, common_valid
        )
        results["source"]["a24_group_dir"] = os.path.abspath(args.a24_group_dir)
    if args.a26_group_dir:
        a26 = load_group_sample(args.a26_group_dir, sample_ids)
        a26_group_scores = query_scores(
            clip_model, a26["features"], len(categories), device, args.chunk_size
        )
        a26_scores = hierarchical_memory_readout(a14_scores, a26_group_scores, a26)
        results["representations"]["a26_hierarchical_memory"] = compare(
            a26_scores, first_scores, second_scores, common_valid
        )
        results["source"]["a26_group_dir"] = os.path.abspath(args.a26_group_dir)
    if args.a27_group_dir:
        a27 = load_group_sample(args.a27_group_dir, sample_ids)
        a27_group_scores = query_scores(
            clip_model, a27["features"], len(categories), device, args.chunk_size
        )
        a27_scores = calibrated_hierarchical_memory_readout(
            a14_scores,
            a27_group_scores,
            a27,
            args.a27_group_query_temperature,
            args.a27_level_margin_threshold,
            args.a27_level_margin_temperature,
        )
        results["representations"]["a27_seeded_four_slot"] = compare(
            a27_scores, first_scores, second_scores, common_valid
        )
        results["source"]["a27_group_dir"] = os.path.abspath(args.a27_group_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
