#!/usr/bin/env python
import json
import os
import shutil
from argparse import ArgumentParser

import numpy as np


def l2_normalize(values, eps=1e-9):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def group_usage_weights(assignments_path, num_groups):
    assignments = np.load(assignments_path)
    top_group_ids = assignments["top_group_ids"].astype(np.int64)
    top_group_scores = assignments["top_group_scores"].astype(np.float32)
    usage = np.zeros((num_groups,), dtype=np.float64)
    valid = top_group_ids >= 0
    np.add.at(usage, top_group_ids[valid], top_group_scores[valid])
    return usage.astype(np.float32)


def make_offsets(group_to_code, num_codes):
    order = np.argsort(group_to_code, kind="stable")
    sorted_codes = group_to_code[order]
    counts = np.bincount(sorted_codes, minlength=num_codes).astype(np.int64)
    offsets = np.zeros((num_codes + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return order.astype(np.int32), offsets


def main():
    parser = ArgumentParser(description="Build a reverse-mounted codebook index: codeword -> groups.")
    parser.add_argument("--codebook_dir", required=True, help="Directory containing codebook.npy and group_to_code.npy.")
    parser.add_argument(
        "--group_features",
        required=True,
        help="Original continuous group tokens used to compute group residuals from their mounted codeword.",
    )
    parser.add_argument("--assignments", required=True, help="point_group_assignments.npz used for usage statistics.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    codebook_path = os.path.join(args.codebook_dir, "codebook.npy")
    group_to_code_path = os.path.join(args.codebook_dir, "group_to_code.npy")
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(codebook_path)
    if not os.path.isfile(group_to_code_path):
        raise FileNotFoundError(group_to_code_path)

    codebook = l2_normalize(np.load(codebook_path).astype(np.float32))
    group_to_code = np.load(group_to_code_path).astype(np.int64)
    group_features = l2_normalize(np.load(args.group_features).astype(np.float32))
    if group_features.shape[0] != group_to_code.shape[0]:
        raise ValueError(
            f"group_features has {group_features.shape[0]} groups, "
            f"but group_to_code has {group_to_code.shape[0]} entries."
        )
    if group_to_code.size and (group_to_code.min() < 0 or group_to_code.max() >= codebook.shape[0]):
        raise ValueError("group_to_code references ids outside codebook.")

    os.makedirs(args.output_dir, exist_ok=True)
    usage = group_usage_weights(args.assignments, group_features.shape[0])
    mounted_code = codebook[group_to_code]
    residuals = group_features - mounted_code
    residual_norm = np.linalg.norm(residuals, axis=1).astype(np.float32)
    cosine_to_code = np.sum(group_features * mounted_code, axis=1).astype(np.float32)
    code_group_indices, code_group_offsets = make_offsets(group_to_code, codebook.shape[0])
    code_usage = np.bincount(group_to_code, weights=usage, minlength=codebook.shape[0]).astype(np.float32)
    code_group_count = np.bincount(group_to_code, minlength=codebook.shape[0]).astype(np.int32)
    active_codes = int((code_group_count > 0).sum())

    shutil.copy2(codebook_path, os.path.join(args.output_dir, "codebook.npy"))
    shutil.copy2(group_to_code_path, os.path.join(args.output_dir, "group_to_code.npy"))
    np.save(os.path.join(args.output_dir, "group_residuals.npy"), residuals.astype(np.float32))
    np.save(os.path.join(args.output_dir, "group_residual_norm.npy"), residual_norm)
    np.save(os.path.join(args.output_dir, "group_cosine_to_code.npy"), cosine_to_code)
    np.save(os.path.join(args.output_dir, "group_usage.npy"), usage)
    np.save(os.path.join(args.output_dir, "code_usage.npy"), code_usage)
    np.save(os.path.join(args.output_dir, "code_group_count.npy"), code_group_count)
    np.savez_compressed(
        os.path.join(args.output_dir, "code_to_groups.npz"),
        indices=code_group_indices,
        offsets=code_group_offsets,
    )

    summary = {
        "codebook_dir": os.path.abspath(args.codebook_dir),
        "group_features": os.path.abspath(args.group_features),
        "assignments": os.path.abspath(args.assignments),
        "num_groups": int(group_features.shape[0]),
        "feature_dim": int(group_features.shape[1]),
        "num_codes": int(codebook.shape[0]),
        "active_codes": active_codes,
        "dead_codes": int(codebook.shape[0] - active_codes),
        "dead_code_ratio": float(1.0 - active_codes / max(codebook.shape[0], 1)),
        "mean_groups_per_active_code": float(code_group_count[code_group_count > 0].mean())
        if active_codes
        else 0.0,
        "mean_cosine_to_code": float(cosine_to_code.mean()) if cosine_to_code.size else 0.0,
        "mean_residual_norm": float(residual_norm.mean()) if residual_norm.size else 0.0,
        "max_code_usage": float(code_usage.max()) if code_usage.size else 0.0,
        "min_active_code_usage": float(code_usage[code_usage > 0].min()) if np.any(code_usage > 0) else 0.0,
    }
    with open(os.path.join(args.output_dir, "reverse_codebook_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
