#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    if "mIoU" not in data or "best_global_threshold" not in data:
        return None
    best_global = data.get("best_global_threshold") or {}
    return {
        "path": str(path),
        "scene": Path(data.get("source_path", "")).name or "-",
        "method": infer_method(path),
        "global_miou": best_global.get("mIoU"),
        "oracle_miou": data.get("mIoU"),
        "macc_025": data.get("mAcc@0.25"),
        "threshold": best_global.get("threshold"),
        "categories": data.get("num_categories"),
        "label_frames": data.get("num_label_frames"),
        "aggregation": data.get("aggregation"),
        "coarse_blend_mode": data.get("coarse_blend_mode"),
        "coarse_blend": data.get("coarse_blend"),
        "query_temperature": data.get("query_temperature"),
        "query_prior_power": data.get("query_prior_power"),
    }


def infer_method(path):
    parts = list(Path(path).parts)
    if "eval" in parts:
        idx = parts.index("eval")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def fmt_percent(value):
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def fmt_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def row_sort_key(row, metric):
    value = row.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def print_markdown(rows):
    print("| scene | method | aggregation | global mIoU | oracle mIoU | mAcc@0.25 | threshold | coarse mode | coarse blend | query temp |")
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|")
    for row in rows:
        print(
            "| {scene} | {method} | {aggregation} | {global_miou} | {oracle_miou} | {macc_025} | {threshold} | {mode} | {blend} | {query_temp} |".format(
                scene=row["scene"],
                method=row["method"],
                aggregation=row["aggregation"] or "-",
                global_miou=fmt_percent(row["global_miou"]),
                oracle_miou=fmt_percent(row["oracle_miou"]),
                macc_025=fmt_percent(row["macc_025"]),
                threshold=fmt_value(row["threshold"]),
                mode=row["coarse_blend_mode"] or "-",
                blend=fmt_value(row["coarse_blend"]),
                query_temp=fmt_value(row["query_temperature"]),
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize LeRF-OVS metrics.json files as a markdown table.")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories to scan recursively.")
    parser.add_argument(
        "--sort",
        choices=["path", "global_miou", "oracle_miou"],
        default="path",
        help="Sort rows by path or descending metric.",
    )
    parser.add_argument(
        "--best-per-scene",
        action="store_true",
        help="Keep only the best row per scene under the selected metric sort.",
    )
    args = parser.parse_args()

    metric_paths = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            metric_paths.extend(path.rglob("metrics.json"))
        elif path.name == "metrics.json":
            metric_paths.append(path)
    rows = [row for path in sorted(set(metric_paths)) if (row := load_metrics(path)) is not None]
    if args.sort == "path":
        rows.sort(key=lambda item: (item["scene"], item["method"], item["path"]))
    else:
        rows.sort(key=lambda item: (item["scene"], -row_sort_key(item, args.sort), item["method"], item["path"]))
    if args.best_per_scene:
        best_rows = []
        seen = set()
        for row in rows:
            if row["scene"] in seen:
                continue
            best_rows.append(row)
            seen.add(row["scene"])
        rows = best_rows
    print_markdown(rows)


if __name__ == "__main__":
    main()
