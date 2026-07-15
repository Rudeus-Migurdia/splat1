#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerf_ovs_paper_protocol import summarize_method_scenes


CANONICAL_SCENES = {"figurines", "ramen", "teatime", "waldo_kitchen"}


def load_scene_result(path):
    with open(path) as source:
        result = json.load(source)
    scene = Path(result.get("source_path", "")).name
    if not scene:
        raise ValueError(f"Cannot infer scene from {path}")
    return scene, result


def format_percent(value):
    return f"{100.0 * float(value):.2f}"


def print_markdown(summary):
    print(f"Shared Gaussian selection threshold: {summary['selection_threshold']:.3g}\n")
    print("| scene | mIoU | Acc@0.25 | Acc@0.5 | samples |")
    print("|---|---:|---:|---:|---:|")
    for scene, row in summary["scenes"].items():
        print(
            f"| {scene} | {format_percent(row['mIoU'])} | "
            f"{format_percent(row['mAcc@0.25'])} | "
            f"{format_percent(row['mAcc@0.5'])} | {row['num_samples']} |"
        )
    print(
        f"| **Mean** | **{format_percent(summary['mIoU'])}** | "
        f"**{format_percent(summary['mAcc@0.25'])}** | "
        f"**{format_percent(summary['mAcc@0.5'])}** | - |"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select one method threshold across scenes and report paper metrics."
    )
    parser.add_argument("metrics", nargs="+", help="One paper-protocol metrics.json per scene")
    parser.add_argument("--output", default=None, help="Optional combined JSON output")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow a diagnostic summary with fewer than the four LeRF-OVS scenes.",
    )
    args = parser.parse_args()

    scene_results = {}
    for raw_path in args.metrics:
        scene, result = load_scene_result(raw_path)
        if scene in scene_results:
            raise ValueError(f"Duplicate scene result: {scene}")
        scene_results[scene] = result
    if not args.allow_partial and set(scene_results) != CANONICAL_SCENES:
        missing = sorted(CANONICAL_SCENES - set(scene_results))
        extra = sorted(set(scene_results) - CANONICAL_SCENES)
        raise ValueError(
            f"Paper summary requires exactly four scenes; missing={missing}, extra={extra}"
        )

    summary = summarize_method_scenes(scene_results)
    summary["scene_metrics"] = {
        scene: str(Path(path).resolve())
        for scene, path in sorted(
            (load_scene_result(path)[0], path) for path in args.metrics
        )
    }
    summary["paper_complete"] = set(scene_results) == CANONICAL_SCENES
    print_markdown(summary)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved combined metrics to {output}")


if __name__ == "__main__":
    main()
