#!/usr/bin/env python
"""Relate paper-protocol category IoU to labeled object area for diagnostics."""

import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np


def polygon_area(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def labeled_area_fractions(label_dir):
    values = defaultdict(list)
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(label_dir, name)) as source:
            payload = json.load(source)
        info = payload["info"]
        image_area = float(info["width"] * info["height"])
        by_category = defaultdict(float)
        for item in payload.get("objects", []):
            segmentation = item.get("segmentation", [])
            if segmentation and isinstance(segmentation[0][0], (int, float)):
                polygons = [segmentation]
            else:
                polygons = segmentation
            by_category[item["category"]] += sum(polygon_area(p) for p in polygons)
        for category, area in by_category.items():
            values[category].append(area / max(image_area, 1.0))
    return values


def paper_row(path, threshold):
    with open(path) as source:
        payload = json.load(source)
    return next(
        row for row in payload["threshold_summary"]
        if abs(float(row["selection_threshold"]) - threshold) < 1e-8
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="NAME=metrics.json or NAME@THRESHOLD=metrics.json",
    )
    parser.add_argument("--selection_threshold", type=float, default=0.55)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:])

    methods = {}
    for spec in args.metrics:
        label, path = spec.split("=", 1)
        if "@" in label:
            name, raw_threshold = label.rsplit("@", 1)
            threshold = float(raw_threshold)
        else:
            name = label
            threshold = args.selection_threshold
        methods[name] = paper_row(path, threshold)
    areas = labeled_area_fractions(args.label_dir)
    categories = sorted(set(areas).intersection(*(set(row["per_category"]) for row in methods.values())))
    rows = []
    for category in categories:
        fractions = np.asarray(areas[category], dtype=np.float64)
        row = {
            "category": category,
            "num_labeled_views": int(fractions.size),
            "median_area_fraction": float(np.median(fractions)),
            "mean_area_fraction": float(fractions.mean()),
            "iou": {name: float(value["per_category"][category]) for name, value in methods.items()},
        }
        names = list(methods)
        row["delta_last_vs_first"] = row["iou"][names[-1]] - row["iou"][names[0]]
        rows.append(row)
    rows.sort(key=lambda row: row["median_area_fraction"])
    cutoff = float(np.quantile([row["median_area_fraction"] for row in rows], 0.5)) if rows else 0.0
    small = [row for row in rows if row["median_area_fraction"] <= cutoff]
    summary = {
        "diagnostic_only": True,
        "labels_used_for_training_or_selection": False,
        "selection_threshold": args.selection_threshold,
        "small_definition": "bottom half of categories by median labeled area fraction",
        "small_area_cutoff": cutoff,
        "methods": list(methods),
        "small_category_mean_iou": {
            name: float(np.mean([row["iou"][name] for row in small])) if small else 0.0
            for name in methods
        },
        "categories": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
