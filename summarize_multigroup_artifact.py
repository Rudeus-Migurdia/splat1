#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

import numpy as np


def main():
    parser = ArgumentParser(description="Summarize multi-group token artifacts without using GPU")
    parser.add_argument("--artifact_dir", required=True)
    args = parser.parse_args()

    summary_path = os.path.join(args.artifact_dir, "group_lift_summary.json")
    assignments_path = os.path.join(args.artifact_dir, "point_group_assignments.npz")
    group_features_path = os.path.join(args.artifact_dir, "group_features.npy")
    group_metadata_path = os.path.join(args.artifact_dir, "group_metadata.npz")

    result = {
        "artifact_dir": os.path.abspath(args.artifact_dir),
        "has_summary": os.path.exists(summary_path),
        "has_assignments": os.path.exists(assignments_path),
        "has_group_features": os.path.exists(group_features_path),
        "has_group_metadata": os.path.exists(group_metadata_path),
    }

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for key in [
            "assignment_mode",
            "num_gaussians",
            "num_groups",
            "num_usable_groups",
            "valid_gaussians",
            "valid_ratio",
            "mean_active_groups_per_valid_point",
            "eval",
        ]:
            result[key] = summary.get(key)

    if os.path.exists(assignments_path):
        assignments = np.load(assignments_path)
        scores = assignments["top_group_scores"]
        active = scores > 0
        active_counts = active.sum(axis=1)
        valid_points = active_counts > 0
        result.update(
            {
                "num_points": int(scores.shape[0]),
                "groups_per_point_capacity": int(scores.shape[1]),
                "points_with_any_group": int(valid_points.sum()),
                "mean_active_groups_per_point": float(active_counts[valid_points].mean()) if valid_points.any() else 0.0,
                "active_group_count_histogram": {
                    str(i): int((active_counts == i).sum()) for i in range(scores.shape[1] + 1)
                },
            }
        )

    if os.path.exists(group_features_path):
        group_features = np.load(group_features_path, mmap_mode="r")
        result["group_features_shape"] = list(group_features.shape)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
