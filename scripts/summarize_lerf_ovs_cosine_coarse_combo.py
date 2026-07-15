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
        "reverse_group_prior": data.get("reverse_group_prior"),
        "coarse_blend_mode": data.get("coarse_blend_mode"),
        "coarse_blend": data.get("coarse_blend"),
        "query_temperature": data.get("query_temperature"),
        "query_prior_power": data.get("query_prior_power"),
    }


def best(rows, metric="global_miou"):
    rows = [row for row in rows if row and row.get(metric) is not None]
    return max(rows, key=lambda row: row[metric]) if rows else None


def pct(value):
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def describe(row):
    if row is None:
        return "-"
    return (
        f"{row['method']} | cal={row['score_calibration']} | "
        f"prior={row['reverse_group_prior']} | coarse={row['coarse_blend_mode']}:{row['coarse_blend']} | "
        f"qt={row['query_temperature']} | qp={row['query_prior_power']}"
    )


def main():
    rows = []
    for scene in SCENES:
        metric_paths = list((scene_method_out(scene) / "eval").glob("*combo_cosine_coarse32*/*metrics.json"))
        scene_rows = [load_metrics(path) for path in metric_paths]
        rows.append((scene, best(scene_rows), len(metric_paths)))

    print("| scene | best combo global | best combo oracle | mAcc@0.25 | n | config |")
    print("|---|---:|---:|---:|---:|---|")
    for scene, row, count in rows:
        print(
            f"| {scene} | {pct(row and row['global_miou'])} | "
            f"{pct(row and row['oracle_miou'])} | {pct(row and row['macc_025'])} | "
            f"{count} | {describe(row)} |"
        )

    valid = [row for _, row, _ in rows if row]
    if valid:
        print()
        print("| average | global | oracle | mAcc@0.25 |")
        print("|---|---:|---:|---:|")
        print(
            "| combo best | "
            f"{pct(sum(row['global_miou'] for row in valid) / len(valid))} | "
            f"{pct(sum(row['oracle_miou'] for row in valid) / len(valid))} | "
            f"{pct(sum(row['macc_025'] for row in valid) / len(valid))} |"
        )


if __name__ == "__main__":
    main()
