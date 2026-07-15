#!/usr/bin/env python
"""Apply LaGa-style global directional consistency to a sparse hypothesis."""

import copy
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F


def directional_reweight(base_features, hypothesis_features, reliability):
    if base_features.shape != hypothesis_features.shape:
        raise ValueError("Base and hypothesis features must match")
    if reliability.shape != base_features.shape[:1]:
        raise ValueError("Reliability must have one value per hypothesis")
    agreement = F.cosine_similarity(
        base_features.float(), hypothesis_features.float(), dim=-1
    ).clamp(0.0, 1.0)
    return reliability.float().clamp(0.0, 1.0) * agreement, agreement


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_dir", required=True)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(sys.argv[1:])

    source_dir = os.path.abspath(args.hypothesis_dir)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        source_manifest = json.load(source)
    if source_manifest.get("representation") != "sparse_continuous_semantic_hypothesis":
        raise ValueError("Unsupported sparse hypothesis representation")

    point_ids = np.load(os.path.join(source_dir, source_manifest["point_ids"]))
    features = np.load(os.path.join(source_dir, source_manifest["features"]))
    packed_reliability = np.load(
        os.path.join(source_dir, source_manifest["reliability"])
    )
    if features.shape[0] != point_ids.size or packed_reliability.shape != point_ids.shape:
        raise ValueError("Sparse hypothesis arrays do not match")

    base_payload = load_torch(os.path.abspath(args.base_consensus))
    base_table = base_payload["initial_features"]
    if point_ids.size and int(point_ids.max()) >= base_table.shape[0]:
        raise ValueError("Hypothesis IDs exceed the base consensus")
    selected_base = base_table[torch.from_numpy(point_ids.astype(np.int64))]
    weighted, agreement = directional_reweight(
        selected_base,
        torch.from_numpy(features),
        torch.from_numpy(packed_reliability.astype(np.float32) / 255.0),
    )
    output_reliability = np.rint(weighted.numpy() * 255.0).astype(np.uint8)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "point_ids.npy"), point_ids)
    np.save(os.path.join(output_dir, "features.npy"), features)
    np.save(os.path.join(output_dir, "reliability.npy"), output_reliability)

    manifest = copy.deepcopy(source_manifest)
    manifest["method"] = source_manifest.get("method", "sparse_hypothesis") + "_directional"
    manifest["point_ids"] = "point_ids.npy"
    manifest["features"] = "features.npy"
    manifest["reliability"] = "reliability.npy"
    manifest["mean_reliability"] = float(weighted.mean()) if weighted.numel() else 0.0
    manifest["directional_consistency"] = {
        "formula": "reliability * clamp(cos(hypothesis, A6_global), 0, 1)",
        "mean_agreement": float(agreement.mean()) if agreement.numel() else 0.0,
        "base_consensus": os.path.abspath(args.base_consensus),
        "source_hypothesis": source_dir,
        "paper_inspiration": "LaGa directional consistency",
    }
    manifest["storage"]["reliability_bytes"] = int(output_reliability.nbytes)
    manifest["storage"]["total_semantic_bytes"] = int(
        point_ids.nbytes + features.nbytes + output_reliability.nbytes
    )
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
