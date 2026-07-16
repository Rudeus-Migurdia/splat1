#!/usr/bin/env python
import argparse
import json
import os

import numpy as np


def prune_sparse_codebook_arrays(
    base_ids,
    valid_mask,
    overflow_points,
    overflow_ids,
    overflow_slots,
    overflow_weights,
    keep_mask,
    invalid_id,
):
    kept_valid = np.asarray(valid_mask, dtype=bool) & np.asarray(keep_mask, dtype=bool)
    pruned_base = np.array(base_ids, copy=True)
    pruned_base[~kept_valid] = invalid_id
    keep_overflow = kept_valid[np.asarray(overflow_points, dtype=np.int64)]
    return (
        pruned_base,
        kept_valid,
        np.asarray(overflow_points)[keep_overflow],
        np.asarray(overflow_ids)[keep_overflow],
        np.asarray(overflow_slots)[keep_overflow],
        np.asarray(overflow_weights)[keep_overflow],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--keep_mask", required=True)
    parser.add_argument("--codebook_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source_dir = os.path.abspath(args.artifact_dir)
    output_dir = os.path.abspath(args.output_dir)
    with open(os.path.join(source_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    if manifest.get("storage_layout") != "base_plus_sparse_overflow":
        raise ValueError("Pruning currently requires base_plus_sparse_overflow layout")

    load = lambda name: np.load(os.path.join(source_dir, manifest[name]))
    keep_mask = np.load(os.path.abspath(args.keep_mask)).astype(bool)
    if keep_mask.shape != (int(manifest["num_gaussians"]),):
        raise ValueError("Keep mask does not match the Gaussian count")
    arrays = prune_sparse_codebook_arrays(
        load("point_code_ids"),
        load("valid_mask"),
        load("overflow_point_ids"),
        load("overflow_code_ids"),
        load("overflow_slots"),
        load("overflow_weights"),
        keep_mask,
        int(manifest["invalid_id"]),
    )
    names = (
        "point_code_ids",
        "valid_mask",
        "overflow_point_ids",
        "overflow_code_ids",
        "overflow_slots",
        "overflow_weights",
    )
    os.makedirs(output_dir, exist_ok=True)
    for name, array in zip(names, arrays):
        filename = manifest[name]
        np.save(os.path.join(output_dir, filename), array)

    codebook_name = manifest["codebook_files"][0]
    codebook_output = os.path.join(output_dir, codebook_name)
    if os.path.lexists(codebook_output):
        os.unlink(codebook_output)
    os.symlink(
        os.path.relpath(os.path.abspath(args.codebook_path), output_dir),
        codebook_output,
    )

    base_ids, valid, overflow_points, overflow_ids, overflow_slots, overflow_weights = arrays
    num_valid = int(valid.sum())
    overflow_count = int(overflow_ids.size)
    one_id = max(num_valid - overflow_count, 0)
    average_ids = (num_valid + overflow_count) / max(num_valid, 1)
    codebook_bytes = os.path.getsize(os.path.abspath(args.codebook_path))
    point_id_bytes = int(base_ids.nbytes + overflow_ids.nbytes)
    storage = {
        "codebook_bytes_fp16": codebook_bytes,
        "point_id_bytes": point_id_bytes,
        "overflow_point_bytes": int(overflow_points.nbytes),
        "overflow_slot_bytes": int(overflow_slots.nbytes),
        "point_weight_bytes": int(overflow_weights.nbytes),
        "valid_mask_bytes": int(valid.nbytes),
    }
    storage["total_semantic_bytes"] = codebook_bytes + sum(storage.values()) - codebook_bytes
    storage["full_per_gaussian_fp16_bytes"] = int(manifest["num_gaussians"]) * int(manifest["feature_dim"]) * 2
    storage["compression_ratio_vs_512d_fp16"] = storage["full_per_gaussian_fp16_bytes"] / storage["total_semantic_bytes"]
    storage["bytes_per_gaussian_amortized"] = storage["total_semantic_bytes"] / int(manifest["num_gaussians"])

    manifest.update(
        {
            "num_valid_gaussians": num_valid,
            "valid_fraction": num_valid / int(manifest["num_gaussians"]),
            "minimum_ids_per_valid_gaussian": 1,
            "average_ids_per_valid_gaussian": average_ids,
            "id_count_histogram": {"1": one_id, "2": overflow_count},
            "storage": storage,
            "source": {
                "type": "counterfactual_capacity_pruned_codebook",
                "artifact_dir": source_dir,
                "keep_mask": os.path.abspath(args.keep_mask),
            },
        }
    )
    with open(os.path.join(output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({
        "num_valid_gaussians": num_valid,
        "valid_fraction": manifest["valid_fraction"],
        "average_ids_per_valid_gaussian": average_ids,
        "storage_megabytes": storage["total_semantic_bytes"] / 2**20,
    }, indent=2))


if __name__ == "__main__":
    main()
