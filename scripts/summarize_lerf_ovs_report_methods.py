#!/usr/bin/env python
"""Summarize report-method LeRF-OVS metrics across scenes."""

import json
from argparse import ArgumentParser
from pathlib import Path


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def scene_out_dir(root, scene):
    if scene == "figurines":
        return root / "runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
    return root / f"runs/prototypes/mask_group_lift/{scene}_teacher_codebook_k256"


def read_metric(path):
    with path.open() as f:
        data = json.load(f)
    best_global = data.get("best_global_threshold") or {}
    return {
        "path": path,
        "global": float(best_global.get("mIoU", 0.0)) * 100.0,
        "oracle": float(data.get("mIoU", 0.0)) * 100.0,
        "macc": float(data.get("mAcc@0.25", 0.0)) * 100.0,
    }


def better(row, incumbent):
    if incumbent is None:
        return True
    return (row["global"], row["oracle"], row["macc"]) > (
        incumbent["global"],
        incumbent["oracle"],
        incumbent["macc"],
    )


def best_matching(paths, predicate):
    best = None
    for path in paths:
        rel = path.as_posix()
        if not predicate(rel):
            continue
        try:
            row = read_metric(path)
        except Exception as exc:
            print(f"[warn] skip {path}: {exc}")
            continue
        if better(row, best):
            best = row
    return best


def baseline_paths(root, scene):
    return list((root / f"runs/drsplat/{scene}_1_pq_openclip_topk45_weight_128/eval").glob("*/metrics.json"))


def method_paths(root, scene):
    out = scene_out_dir(root, scene)
    return list((out / "eval").glob("*/metrics.json"))


def family_specs(root, scene):
    base_eval = root / f"runs/drsplat/{scene}_1_pq_openclip_topk45_weight_128/eval"
    if scene == "figurines":
        teacher_prefix = "lerf_ovs_teacher_w0p75"
        codebook_weighted = "lerf_ovs_teacher_w0p75_codebook_k256_weighted"
    else:
        teacher_prefix = "lerf_ovs_teacher"
        codebook_weighted = "lerf_ovs_teacher_codebook_k256_weighted"

    return [
        (
            "baseline original",
            lambda p: p == (base_eval / "lerf_ovs_miou/metrics.json").as_posix(),
            baseline_paths(root, scene),
        ),
        (
            "baseline best calibration",
            lambda p: "/eval/lerf_ovs_miou" in p,
            baseline_paths(root, scene),
        ),
        (
            "early multigroup best",
            lambda p: "lerf_ovs_multigroup" in p,
            method_paths(root, scene),
        ),
        (
            "hybrid raw-group best",
            lambda p: "lerf_ovs_hybrid_rawgroup" in p or "lerf_ovs_hybrid" in p,
            method_paths(root, scene),
        ),
        (
            "teacher token best",
            lambda p: f"{teacher_prefix}_weighted" in p and "codebook" not in p,
            method_paths(root, scene),
        ),
        (
            "teacher-codebook original",
            lambda p: p.endswith(f"/{codebook_weighted}/metrics.json"),
            method_paths(root, scene),
        ),
        (
            "teacher-codebook best calibration",
            lambda p: (
                codebook_weighted in p
                and "_coarse32_" not in p
                and "_hier_" not in p
                and "_reverse" not in p
                and "_routing" not in p
                and "_combo" not in p
            ),
            method_paths(root, scene),
        ),
        (
            "hier/coarse best",
            lambda p: (
                ("_coarse32_" in p or "_hier_" in p)
                and "combo_cosine" not in p
                and "routing_probe" not in p
            ),
            method_paths(root, scene),
        ),
        (
            "reverse-mounted best",
            lambda p: "reverse" in p and "routing_probe" not in p and "combo_cosine" not in p,
            method_paths(root, scene),
        ),
        (
            "routing best",
            lambda p: "routing_probe" in p,
            method_paths(root, scene),
        ),
        (
            "cosine+coarse combo best",
            lambda p: "combo_cosine_coarse" in p,
            method_paths(root, scene),
        ),
    ]


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def write_markdown(root, output):
    family_rows = {}
    scene_rows = {scene: {} for scene in SCENES}

    for scene in SCENES:
        for family, predicate, paths in family_specs(root, scene):
            row = best_matching(paths, predicate)
            scene_rows[scene][family] = row
            family_rows.setdefault(family, {})[scene] = row

    lines = [
        "# LeRF-OVS Report Method Multi-Scene Summary",
        "",
        "All values are percentages. Global mIoU is the best single global threshold recorded in each metrics file; oracle mIoU is per-category best threshold.",
        "",
        "## Family Average",
        "",
        "| method family | scenes | avg global | avg oracle | avg mAcc@0.25 | complete 4-scene |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family, by_scene in family_rows.items():
        rows = [row for row in by_scene.values() if row is not None]
        if rows:
            avg_g = sum(row["global"] for row in rows) / len(rows)
            avg_o = sum(row["oracle"] for row in rows) / len(rows)
            avg_a = sum(row["macc"] for row in rows) / len(rows)
        else:
            avg_g = avg_o = avg_a = None
        lines.append(
            f"| {family} | {len(rows)}/4 | {fmt(avg_g)} | {fmt(avg_o)} | {fmt(avg_a)} | {'yes' if len(rows) == 4 else 'no'} |"
        )

    lines.extend(["", "## Per-Scene Best By Family", ""])
    for scene in SCENES:
        baseline = scene_rows[scene].get("baseline best calibration")
        lines.extend(
            [
                f"### {scene}",
                "",
                "| method family | global | oracle | mAcc@0.25 | delta global vs calibrated baseline | metrics |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for family, row in scene_rows[scene].items():
            if row is None:
                lines.append(f"| {family} | - | - | - | - | - |")
                continue
            delta = row["global"] - baseline["global"] if baseline else None
            rel = row["path"].relative_to(root)
            lines.append(
                f"| {family} | {row['global']:.2f} | {row['oracle']:.2f} | {row['macc']:.2f} | {fmt(delta)} | `{rel}` |"
            )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    print(write_markdown(root, output))


if __name__ == "__main__":
    main()
