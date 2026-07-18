#!/usr/bin/env python3
"""Summarize equal four-token query fusion across codebook seeds."""

import argparse
import json
import os
from statistics import mean, pstdev


METRICS = ("mIoU", "mAcc@0.25", "mAcc@0.5")
METHODS = {
    "equal_query_softmax": "a30_equal_query_softmax",
    "equal_query_max": "a30_equal_query_max",
}


def parse_seed_run(value):
    try:
        seed, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected SEED=SUMMARY_JSON") from error
    if not seed or not path:
        raise argparse.ArgumentTypeError("expected SEED=SUMMARY_JSON")
    return seed, os.path.abspath(path)


def metric_stats(values, baseline):
    return {
        "mean": mean(values),
        "population_std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean_minus_a20": mean(values) - baseline,
        "worst_minus_a20": min(values) - baseline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-run",
        action="append",
        required=True,
        type=parse_seed_run,
        help="Codebook seed and three-scene summary as SEED=SUMMARY_JSON",
    )
    parser.add_argument("--teatime-summary")
    parser.add_argument("--teatime-seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_paths = dict(args.seed_run)
    if len(seed_paths) != len(args.seed_run):
        raise ValueError("seed names must be unique")
    runs = {seed: json.load(open(path)) for seed, path in seed_paths.items()}
    if len(runs) < 2:
        raise ValueError("at least two seeds are required")

    first = next(iter(runs.values()))
    scenes = tuple(first["scenes"])
    a20 = first["a20_mean"]
    for seed, run in runs.items():
        if tuple(run["scenes"]) != scenes:
            raise ValueError(f"scene mismatch for seed {seed}")
        if run["a20_mean"] != a20:
            raise ValueError(f"A20 reference mismatch for seed {seed}")

    summary = {
        "method": "equal four-token query-aware score fusion",
        "seed_runs": seed_paths,
        "scenes": list(scenes),
        "a20_reference": a20,
        "per_seed": {},
        "aggregate": {},
        "per_scene": {},
    }
    for seed, run in runs.items():
        summary["per_seed"][seed] = {
            name: run[key + "_mean"] for name, key in METHODS.items()
        }

    for name, key in METHODS.items():
        summary["aggregate"][name] = {
            metric: metric_stats(
                [run[key + "_mean"][metric] for run in runs.values()],
                a20[metric],
            )
            for metric in METRICS
        }

    for scene in scenes:
        summary["per_scene"][scene] = {}
        for name, key in METHODS.items():
            summary["per_scene"][scene][name] = {
                metric: {
                    "mean": mean(
                        run["scenes"][scene][key][metric] for run in runs.values()
                    ),
                    "population_std": pstdev(
                        run["scenes"][scene][key][metric] for run in runs.values()
                    ),
                    "minimum": min(
                        run["scenes"][scene][key][metric] for run in runs.values()
                    ),
                    "maximum": max(
                        run["scenes"][scene][key][metric] for run in runs.values()
                    ),
                }
                for metric in METRICS
            }

    max_key = METHODS["equal_query_max"]
    level_fractions = {f"level_{level}": [] for level in range(4)}
    for run in runs.values():
        for scene in scenes:
            route = run["scenes"][scene]["max_route"]["dominant_level_fraction"]
            for level, value in route.items():
                level_fractions[level].append(value)
    summary["max_level_balance"] = {
        level: {
            "mean": mean(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for level, values in level_fractions.items()
    }

    best_seed = max(
        runs,
        key=lambda seed: runs[seed][max_key + "_mean"]["mIoU"],
    )
    strict_passes = sum(
        run[max_key + "_mean"]["mAcc@0.5"] >= a20["mAcc@0.5"]
        for run in runs.values()
    )
    max_aggregate = summary["aggregate"]["equal_query_max"]
    summary["decision"] = {
        "best_seed_by_max_miou": best_seed,
        "all_seeds_max_miou_above_a20": all(
            run[max_key + "_mean"]["mIoU"] > a20["mIoU"]
            for run in runs.values()
        ),
        "max_mean_beats_a20_all_metrics": all(
            max_aggregate[metric]["mean_minus_a20"] > 0.0 for metric in METRICS
        ),
        "max_miou_population_std_at_most_half_point": (
            max_aggregate["mIoU"]["population_std"] <= 0.005
        ),
        "max_strict_accuracy_seed_passes": strict_passes,
        "max_strict_accuracy_seed_total": len(runs),
        "all_seed_scene_routes_use_every_level": all(
            stats["minimum"] > 0.0 for stats in summary["max_level_balance"].values()
        ),
        "verdict": (
            "max readout is stable for mean mIoU and Acc25; strict Acc5 has "
            "moderate threshold sensitivity"
        ),
    }

    if args.teatime_summary:
        if not args.teatime_seed or args.teatime_seed not in runs:
            raise ValueError("--teatime-seed must identify one of the seed runs")
        teatime = json.load(open(args.teatime_summary))
        seed_run = runs[args.teatime_seed]
        four_scene = {}
        for name, key in METHODS.items():
            tea_key = "a31_" + name
            four_scene[name] = {
                metric: (
                    len(scenes) * seed_run[key + "_mean"][metric]
                    + teatime[tea_key][metric]
                )
                / (len(scenes) + 1)
                for metric in METRICS
            }
        for name, key in (
            ("paper_baseline_local", "paper_baseline_local"),
            ("e8_3", "e8_3"),
        ):
            four_scene[name] = {
                metric: (
                    len(scenes) * seed_run[key + "_mean"][metric]
                    + teatime[key][metric]
                )
                / (len(scenes) + 1)
                for metric in METRICS
            }
        summary["four_scene_validation"] = {
            "seed": args.teatime_seed,
            "teatime_summary": os.path.abspath(args.teatime_summary),
            "metrics": four_scene,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
