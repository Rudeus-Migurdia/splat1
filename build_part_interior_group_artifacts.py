#!/usr/bin/env python
"""Gate exact part-code attachments by local 3D part-interior support."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from build_gaussian_superpoint_support import build_knn, load_geometry


def link(source, destination):
    if os.path.lexists(destination):
        os.unlink(destination)
    os.symlink(os.path.abspath(source), destination)


def write_artifact(source_dir, output_dir, point_ids, point_weights, manifest, note):
    os.makedirs(output_dir, exist_ok=True)
    for name in (
        "shared_vocabulary.npy",
        "group_semantic_code_ids.npy",
        "group_reliability.npy",
    ):
        link(os.path.join(source_dir, name), os.path.join(output_dir, name))
    np.save(os.path.join(output_dir, "point_group_ids.npy"), point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    output_manifest = dict(manifest)
    output_manifest["source"] = {
        **manifest.get("source", {}),
        "interior_parent": source_dir,
        "interior_rule": note,
    }
    output_manifest["covered_fraction"] = float((point_weights[:, 0] > 0).mean())
    output_manifest["storage"] = {
        **manifest["storage"],
        "interior_point_table_bytes": int(point_ids.nbytes + point_weights.nbytes),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(output_manifest, output, indent=2)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--hierarchy_dir", required=True)
    parser.add_argument("--exact_artifact_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--minimum_same_neighbors", type=int, default=4)
    parser.add_argument("--interior_fraction", type=float, default=0.75)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.neighbors <= 1 or not 0.0 <= args.interior_fraction <= 1.0:
        raise ValueError("Interior neighborhood parameters are invalid")
    if not 1 <= args.minimum_same_neighbors <= args.neighbors:
        raise ValueError("Minimum same-part neighbors is invalid")

    hierarchy_dir = os.path.abspath(args.hierarchy_dir)
    exact_dir = os.path.abspath(args.exact_artifact_dir)
    with open(os.path.join(exact_dir, "manifest.json")) as source:
        manifest = json.load(source)
    part_ids = np.load(os.path.join(hierarchy_dir, "part_group_ids.npy")).astype(np.int64)
    point_ids = np.load(os.path.join(exact_dir, "point_group_ids.npy"))
    point_weights = np.load(os.path.join(exact_dir, "point_group_weights.npy"))
    if part_ids.shape[0] != point_ids.shape[0]:
        raise ValueError("Hierarchy and exact code attachments do not match")
    xyz, _, _, _ = load_geometry(args.geometry_checkpoint, part_ids.size)
    valid_global = np.flatnonzero(part_ids >= 0)
    neighbors, _, resources, backend = build_knn(
        xyz[valid_global],
        args.neighbors,
        args.chunk_size,
        args.faiss_gpu,
        args.knn_workers,
    )
    valid_parts = part_ids[valid_global]
    same = valid_parts[neighbors] == valid_parts[:, None]
    same_count = same.sum(axis=1)
    support = same_count.astype(np.float32) / float(args.neighbors)
    support_global = np.zeros(part_ids.size, dtype=np.float32)
    support_global[valid_global] = support
    count_global = np.zeros(part_ids.size, dtype=np.int16)
    count_global[valid_global] = same_count.astype(np.int16)
    del resources

    source_valid = point_weights[:, 0] > 0
    soft_weights = point_weights.copy()
    soft_weights[:, 0] = np.rint(
        soft_weights[:, 0].astype(np.float32) * support_global
    ).clip(0, 255).astype(np.uint8)
    hard_keep = source_valid & (support_global >= args.interior_fraction) & (
        count_global >= args.minimum_same_neighbors
    )
    hard_ids = point_ids.copy()
    hard_weights = point_weights.copy()
    hard_ids[~hard_keep, 0] = int(manifest["invalid_id"])
    hard_weights[~hard_keep, 0] = 0

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    np.save(os.path.join(output_root, "part_interior_support.npy"), support_global)
    rule = (
        f"8-NN same-part support; hard requires fraction>={args.interior_fraction} "
        f"and count>={args.minimum_same_neighbors}; backend={backend}"
    )
    write_artifact(
        exact_dir,
        os.path.join(output_root, "soft"),
        point_ids,
        soft_weights,
        manifest,
        rule + "; soft membership equals support",
    )
    write_artifact(
        exact_dir,
        os.path.join(output_root, "hard"),
        hard_ids,
        hard_weights,
        manifest,
        rule + "; hard boundary attachments removed",
    )
    summary = {
        "representation": "part_interior_group_code_gate",
        "num_gaussians": int(part_ids.size),
        "source_covered_fraction": float(source_valid.mean()),
        "mean_support_on_source": float(support_global[source_valid].mean()),
        "hard_retained_fraction_of_source": float(hard_keep.sum() / max(1, source_valid.sum())),
        "hard_covered_fraction": float(hard_keep.mean()),
        "knn_backend": backend,
        "args": vars(args),
    }
    with open(os.path.join(output_root, "manifest.json"), "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
