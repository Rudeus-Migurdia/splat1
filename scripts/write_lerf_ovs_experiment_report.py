#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def infer_method(path):
    parts = list(Path(path).parts)
    if "eval" in parts:
        idx = parts.index("eval")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def load_metric(path):
    with open(path) as f:
        data = json.load(f)
    if "mIoU" not in data or "best_global_threshold" not in data:
        return None
    best_global = data.get("best_global_threshold") or {}
    return {
        "path": Path(path),
        "scene": Path(data.get("source_path", "")).name or "-",
        "method": infer_method(path),
        "global_miou": best_global.get("mIoU"),
        "oracle_miou": data.get("mIoU"),
        "macc_025": data.get("mAcc@0.25"),
        "threshold": best_global.get("threshold"),
        "aggregation": data.get("aggregation"),
        "coarse_blend_mode": data.get("coarse_blend_mode"),
        "coarse_blend": data.get("coarse_blend"),
        "query_temperature": data.get("query_temperature"),
        "num_categories": data.get("num_categories"),
        "num_label_frames": data.get("num_label_frames"),
        "per_category": data.get("per_category") or {},
        "raw": data,
    }


def metric_paths(raw_paths):
    paths = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(path.rglob("metrics.json"))
        elif path.name == "metrics.json":
            paths.append(path)
    return sorted(set(paths))


def pct(value):
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def val(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def delta(value, base):
    if value is None or base is None:
        return None
    return float(value) - float(base)


def method_label(row):
    parts = [row["method"]]
    if row.get("aggregation"):
        parts.append(f"agg={row['aggregation']}")
    if row.get("coarse_blend_mode"):
        parts.append(f"coarse={row['coarse_blend_mode']}:{val(row.get('coarse_blend'))}")
    if row.get("query_temperature") is not None and row.get("aggregation", "").startswith("query"):
        parts.append(f"temp={val(row['query_temperature'])}")
    return ", ".join(parts)


def write_table(lines, rows, baseline_by_scene):
    lines.append("| scene | method | global mIoU | delta | oracle mIoU | delta | mAcc@0.25 | threshold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        base = baseline_by_scene.get(row["scene"])
        g_delta = delta(row["global_miou"], base["global_miou"]) if base else None
        o_delta = delta(row["oracle_miou"], base["oracle_miou"]) if base else None
        lines.append(
            "| {scene} | {method} | {global_miou} | {g_delta} | {oracle_miou} | {o_delta} | {macc} | {thr} |".format(
                scene=row["scene"],
                method=method_label(row),
                global_miou=pct(row["global_miou"]),
                g_delta=pct(g_delta),
                oracle_miou=pct(row["oracle_miou"]),
                o_delta=pct(o_delta),
                macc=pct(row["macc_025"]),
                thr=val(row["threshold"]),
            )
        )


def best_nonbaseline(rows, baseline_paths):
    candidates = [row for row in rows if row["path"] not in baseline_paths]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["global_miou"] or -1.0))


def write_category_delta(lines, baseline, candidate, limit):
    if not baseline or not candidate:
        return
    base_cats = baseline["per_category"]
    cand_cats = candidate["per_category"]
    shared = sorted(set(base_cats) & set(cand_cats))
    if not shared:
        return
    rows = []
    for category in shared:
        base_iou = base_cats[category].get("best_iou")
        cand_iou = cand_cats[category].get("best_iou")
        if base_iou is None or cand_iou is None:
            continue
        rows.append((float(cand_iou) - float(base_iou), category, float(base_iou), float(cand_iou)))
    if not rows:
        return

    lines.append("")
    lines.append(f"## Per-Category Delta: {candidate['scene']}")
    lines.append("")
    lines.append(f"Baseline: `{baseline['method']}`")
    lines.append("")
    lines.append(f"Candidate: `{method_label(candidate)}`")
    lines.append("")
    lines.append("### Largest Drops")
    lines.append("")
    lines.append("| category | baseline IoU | candidate IoU | delta |")
    lines.append("|---|---:|---:|---:|")
    for d, category, base_iou, cand_iou in sorted(rows)[:limit]:
        lines.append(f"| {category} | {pct(base_iou)} | {pct(cand_iou)} | {pct(d)} |")
    lines.append("")
    lines.append("### Largest Gains")
    lines.append("")
    lines.append("| category | baseline IoU | candidate IoU | delta |")
    lines.append("|---|---:|---:|---:|")
    for d, category, base_iou, cand_iou in sorted(rows, reverse=True)[:limit]:
        lines.append(f"| {category} | {pct(base_iou)} | {pct(cand_iou)} | {pct(d)} |")


def main():
    parser = argparse.ArgumentParser(description="Write a markdown report for LeRF-OVS experiment metrics.")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories to scan.")
    parser.add_argument("--baseline", action="append", default=[], help="Baseline metrics.json path. Can be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--category-limit", type=int, default=8)
    args = parser.parse_args()

    rows = [row for path in metric_paths(args.paths) if (row := load_metric(path)) is not None]
    rows.sort(key=lambda row: (row["scene"], -float(row["global_miou"] or -1.0), row["method"], str(row["path"])))

    baseline_paths = {Path(path).resolve() for path in args.baseline}
    baseline_rows = [row for row in rows if row["path"].resolve() in baseline_paths]
    baseline_by_scene = {}
    for row in baseline_rows:
        baseline_by_scene[row["scene"]] = row

    best_by_scene = {}
    for row in rows:
        best_by_scene.setdefault(row["scene"], row)

    lines = ["# LeRF-OVS Experiment Report", ""]
    lines.append("## Best Per Scene")
    lines.append("")
    write_table(lines, list(best_by_scene.values()), baseline_by_scene)
    lines.append("")
    lines.append("## All Metrics")
    lines.append("")
    write_table(lines, rows, baseline_by_scene)

    for scene, baseline in baseline_by_scene.items():
        scene_rows = [row for row in rows if row["scene"] == scene]
        candidate = best_nonbaseline(scene_rows, baseline_paths)
        write_category_delta(lines, baseline, candidate, args.category_limit)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
