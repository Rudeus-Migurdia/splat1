#!/usr/bin/env python
"""Union class-semantic and learned-instance Gaussian association caches."""

import json
import os
import sys
from argparse import ArgumentParser

from scipy import sparse


def combine_signatures(first, second):
    if first.shape != second.shape:
        raise ValueError("Association signatures must have matching shapes")
    return first.maximum(second).tocsr()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--class_cache", required=True)
    parser.add_argument("--instance_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_views", type=int, required=True)
    args = parser.parse_args(sys.argv[1:])
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    totals = {
        "candidate_pairs": 0,
        "selected_pairs": 0,
        "geometry_pairs": 0,
        "semantic_pairs": 0,
        "semantic_rescued_pairs": 0,
    }
    for view_index in range(args.num_views):
        name = f"{view_index:04d}"
        first = sparse.load_npz(os.path.join(args.class_cache, f"{name}.npz"))
        second = sparse.load_npz(os.path.join(args.instance_cache, f"{name}.npz"))
        combined = combine_signatures(first, second)
        with open(os.path.join(args.class_cache, f"{name}.json")) as source:
            first_summary = json.load(source)
        with open(os.path.join(args.instance_cache, f"{name}.json")) as source:
            second_summary = json.load(source)
        geometry_pairs = int(first_summary["geometry_pairs"])
        summary = {
            "candidate_pairs": max(
                int(first_summary["candidate_pairs"]),
                int(second_summary["candidate_pairs"]),
            ),
            "selected_pairs": int(combined.nnz),
            "geometry_pairs": geometry_pairs,
            "semantic_pairs": int(
                first_summary["semantic_pairs"] + second_summary["semantic_pairs"]
            ),
            "semantic_rescued_pairs": max(0, int(combined.nnz) - geometry_pairs),
        }
        sparse.save_npz(os.path.join(output_dir, f"{name}.npz"), combined, compressed=True)
        with open(os.path.join(output_dir, f"{name}.json"), "w") as output:
            json.dump(summary, output, indent=2)
        for key, value in summary.items():
            totals[key] += value
        print(json.dumps({"view": view_index, "selected_pairs": int(combined.nnz)}))
    result = {
        "format_version": 1,
        "representation": "class_instance_semantic_association_union",
        "num_views": args.num_views,
        "class_cache": os.path.abspath(args.class_cache),
        "instance_cache": os.path.abspath(args.instance_cache),
        "totals": totals,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
