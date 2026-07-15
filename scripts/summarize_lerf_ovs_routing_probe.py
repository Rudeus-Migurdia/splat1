#!/usr/bin/env python
import json
from pathlib import Path


ROOT = Path("/mnt/zju105100171/home/anlanfan/Dr-Splat")
SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def scene_method_out(scene):
    if scene == "figurines":
        return ROOT / "runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
    return ROOT / f"runs/prototypes/mask_group_lift/{scene}_teacher_codebook_k256"


def load_metrics(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    best_global = data.get("best_global_threshold") or {}
    return {
        "path": str(path),
        "method": path.parent.name,
        "global_miou": best_global.get("mIoU"),
        "oracle_miou": data.get("mIoU"),
        "macc_025": data.get("mAcc@0.25"),
        "threshold": best_global.get("threshold"),
        "score_calibration": data.get("score_calibration"),
        "activation_normalization": data.get("activation_normalization"),
        "reverse_group_prior": data.get("reverse_group_prior"),
        "reverse_prior_power": data.get("reverse_prior_power"),
        "reverse_residual_temperature": data.get("reverse_residual_temperature"),
        "query_temperature": data.get("query_temperature"),
        "query_prior_power": data.get("query_prior_power"),
        "reverse_top_codes": data.get("reverse_top_codes"),
        "reverse_residual_weight": data.get("reverse_residual_weight"),
        "reverse_code_blend": data.get("reverse_code_blend"),
    }


def best(rows, metric="global_miou"):
    rows = [row for row in rows if row and row.get(metric) is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: row[metric])


def pct(value):
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def compact(row):
    if row is None:
        return "-"
    return (
        f"{row['method']} | cal={row['score_calibration']} | "
        f"act={row['activation_normalization']} | prior={row['reverse_group_prior']} | "
        f"qt={row['query_temperature']} qp={row['query_prior_power']}"
    )


def main():
    rows = []
    for scene in SCENES:
        out = scene_method_out(scene)
        metric_paths = list((out / "eval").glob("*routing_probe*/metrics.json"))
        scene_rows = [load_metrics(path) for path in metric_paths]
        scene_best = best(scene_rows)
        rows.append((scene, scene_best, len(metric_paths)))

    print("| scene | best routing global | best routing oracle | mAcc@0.25 | n | config |")
    print("|---|---:|---:|---:|---:|---|")
    for scene, row, count in rows:
        print(
            f"| {scene} | {pct(row and row['global_miou'])} | "
            f"{pct(row and row['oracle_miou'])} | {pct(row and row['macc_025'])} | "
            f"{count} | {compact(row)} |"
        )

    valid = [row for _, row, _ in rows if row]
    if valid:
        print()
        print("| average | global | oracle | mAcc@0.25 |")
        print("|---|---:|---:|---:|")
        print(
            "| routing best | "
            f"{pct(sum(row['global_miou'] for row in valid) / len(valid))} | "
            f"{pct(sum(row['oracle_miou'] for row in valid) / len(valid))} | "
            f"{pct(sum(row['macc_025'] for row in valid) / len(valid))} |"
        )


if __name__ == "__main__":
    main()
