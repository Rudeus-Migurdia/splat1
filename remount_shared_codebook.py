#!/usr/bin/env python
"""Create an isolated fixed-ID artifact view mounted on a new shared codebook."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


ARRAY_KEYS = (
    "point_code_ids",
    "valid_mask",
    "point_code_weights",
    "overflow_point_ids",
    "overflow_code_ids",
    "overflow_slots",
    "overflow_weights",
)


def replace_relative_symlink(source, destination):
    if os.path.lexists(destination):
        os.unlink(destination)
    os.symlink(os.path.relpath(os.path.abspath(source), os.path.dirname(destination)), destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--codebook_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--training_metrics", default=None)
    parser.add_argument("--mode", choices=("base", "candidate"), default=None)
    args = parser.parse_args(sys.argv[1:])

    source_dir = os.path.abspath(args.artifact_dir)
    codebook_path = os.path.abspath(args.codebook_path)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(codebook_path)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    os.makedirs(output_dir, exist_ok=True)

    for key in ARRAY_KEYS:
        filename = manifest.get(key)
        if filename:
            replace_relative_symlink(
                os.path.join(source_dir, filename),
                os.path.join(output_dir, filename),
            )
    codebook_name = manifest["codebook_files"][0]
    replace_relative_symlink(codebook_path, os.path.join(output_dir, codebook_name))
    codebook = np.load(codebook_path, mmap_mode="r")
    if codebook.ndim != 2 or codebook.shape[1] != int(manifest["feature_dim"]):
        raise ValueError("Remounted codebook shape does not match the artifact")
    manifest["num_codes"] = int(codebook.shape[0])
    storage = dict(manifest.get("storage", {}))
    old_codebook_bytes = int(storage.get("codebook_bytes_fp16", 0))
    new_codebook_bytes = int(codebook.size * codebook.dtype.itemsize)
    storage["codebook_bytes_fp16"] = new_codebook_bytes
    if "total_semantic_bytes" in storage:
        storage["total_semantic_bytes"] += new_codebook_bytes - old_codebook_bytes
        storage["compression_ratio_vs_512d_fp16"] = (
            storage["full_per_gaussian_fp16_bytes"]
            / storage["total_semantic_bytes"]
        )
        storage["bytes_per_gaussian_amortized"] = (
            storage["total_semantic_bytes"] / manifest["num_gaussians"]
        )
    manifest["storage"] = storage

    remount = {
        "type": "fixed_id_shared_vocabulary_remount",
        "artifact_dir": source_dir,
        "codebook_path": codebook_path,
    }
    if args.training_metrics:
        metrics_path = os.path.abspath(args.training_metrics)
        remount["training_metrics"] = metrics_path
        if args.mode:
            with open(metrics_path) as source:
                metrics = json.load(source)
            mode_metrics = metrics.get("final_metrics", {}).get(args.mode)
            if mode_metrics:
                manifest["mean_reconstruction_cosine"] = mode_metrics["mean_cosine"]
                remount["mode"] = args.mode
                remount["validation_metrics"] = mode_metrics
    manifest["source_before_remount"] = manifest.get("source")
    manifest["source"] = remount
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(remount, indent=2))


if __name__ == "__main__":
    main()
