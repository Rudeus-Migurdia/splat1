#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A34 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A34}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A34}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SEEDS=${SEEDS:-"20260717 20260718 20260719"}
MEMORY_ROOTS=${MEMORY_ROOTS:-"$ROOT/runs/a30_equal_four_token_query_fusion_20260718_101242 $ROOT/runs/a32_equal_four_token_seed_replication_20260718_112041 $ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948"}
MARGIN_VALUES=${MARGIN_VALUES:-"0.0025 0.005 0.01"}
MARGIN_TAGS=${MARGIN_TAGS:-"m0025 m005 m01"}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=20260717 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$RUN_ROOT" "$LOG_DIR"

evaluate_scene() {
  local scene=$1
  local seed=$2
  local memory_root=$3
  local margin=$4
  local tag=$5
  local output=$RUN_ROOT/seed_${seed}/$scene/eval_margin_top2_${tag}
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory_root/$scene/equal_four_token_memory" \
    --group_topk 4 --group_readout equal_query_margin_top2 \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --group_query_tie_margin "$margin" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output" \
    > "$LOG_DIR/${scene}_seed_${seed}_${tag}_eval.log" 2>&1
}

run_scene() {
  local scene=$1
  read -r -a seeds <<< "$SEEDS"
  read -r -a memory_roots <<< "$MEMORY_ROOTS"
  read -r -a margins <<< "$MARGIN_VALUES"
  read -r -a tags <<< "$MARGIN_TAGS"
  for seed_index in "${!seeds[@]}"; do
    for margin_index in "${!margins[@]}"; do
      evaluate_scene \
        "$scene" \
        "${seeds[$seed_index]}" \
        "${memory_roots[$seed_index]}" \
        "${margins[$margin_index]}" \
        "${tags[$margin_index]}"
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$ROOT/scripts/gpu_guard.py"; do
  [[ -f "$required" ]] || { echo "Missing required source: $required" >&2; exit 2; }
done

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
read -r -a seeds <<< "$SEEDS"
read -r -a memory_roots <<< "$MEMORY_ROOTS"
read -r -a margins <<< "$MARGIN_VALUES"
read -r -a tags <<< "$MARGIN_TAGS"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || {
  echo "SCENES and GPU_LIST must have equal lengths" >&2
  exit 2
}
[[ "${#seeds[@]}" -eq "${#memory_roots[@]}" ]] || {
  echo "SEEDS and MEMORY_ROOTS must have equal lengths" >&2
  exit 2
}
[[ "${#margins[@]}" -eq "${#tags[@]}" ]] || {
  echo "MARGIN_VALUES and MARGIN_TAGS must have equal lengths" >&2
  exit 2
}

for seed_index in "${!seeds[@]}"; do
  summary=${memory_roots[$seed_index]}/three_scene_summary.json
  [[ -f "$summary" ]] || { echo "Missing seed summary: $summary" >&2; exit 2; }
  for scene in "${scenes[@]}"; do
    memory=${memory_roots[$seed_index]}/$scene/equal_four_token_memory
    [[ -f "$memory/manifest.json" ]] || {
      echo "Missing seed memory: $memory" >&2
      exit 2
    }
    "$PYTHON_BIN" - "$memory/manifest.json" "${seeds[$seed_index]}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert manifest["resident_slots_required"] == 4
assert manifest["reproducibility"]["seed"] == int(sys.argv[2])
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [
    2048, 4096, 8192, 16384
]
assert all(parent == -1 for parent in __import__("numpy").load(
    __import__("os").path.join(
        __import__("os").path.dirname(sys.argv[1]), manifest["group_parent_ids"]
    ), mmap_mode="r"
))
PY
  done
done

for scene in "${scenes[@]}"; do
  for required in \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$A20_ROOT/$scene/eval_fine_part/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

script_path=${BASH_SOURCE[0]}
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$script_path" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A20_ROOT" "$SEEDS" "$MEMORY_ROOTS" \
  "$MARGIN_VALUES" "$MARGIN_TAGS" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import statistics
import sys

root, a20_root, raw_seeds, raw_memory_roots, raw_margins, raw_tags, raw_t, *scenes = sys.argv[1:]
seeds = raw_seeds.split()
memory_roots = raw_memory_roots.split()
margins = [float(value) for value in raw_margins.split()]
tags = raw_tags.split()
threshold = float(raw_t)
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def row(path):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metric_names}


def route(path):
    diagnostics = json.load(open(path)).get("route_diagnostics", {})
    tie_fractions = [
        float(item.get("tie_blended_fraction_covered", 0.0))
        for item in diagnostics.values()
    ]
    margins_by_query = [
        float(item.get("mean_top2_adjusted_margin", 0.0))
        for item in diagnostics.values()
    ]
    dominant = {f"level_{level}": 0 for level in range(4)}
    for item in diagnostics.values():
        for level, count in item.get("dominant_level_counts", {}).items():
            dominant[level] += int(count)
    total = sum(dominant.values())
    return {
        "mean_tie_blended_fraction_covered": (
            statistics.mean(tie_fractions) if tie_fractions else 0.0
        ),
        "mean_top2_adjusted_margin": (
            statistics.mean(margins_by_query) if margins_by_query else 0.0
        ),
        "dominant_level_fraction": {
            level: count / max(1, total) for level, count in dominant.items()
        },
    }


references = {}
for seed, memory_root in zip(seeds, memory_roots):
    references[seed] = json.load(
        open(os.path.join(memory_root, "three_scene_summary.json"))
    )
a20 = next(iter(references.values()))["a20_mean"]

summary = {
    "method": "A34 margin-aware top-2 fusion across four peer semantic tokens",
    "seeds": seeds,
    "scenes": scenes,
    "query_temperature": 0.05,
    "selection_threshold": threshold,
    "margins": margins,
    "a20_mean": a20,
    "reference_equal_query_max": {},
    "variants": {},
}

for seed, reference in references.items():
    summary["reference_equal_query_max"][seed] = reference[
        "a30_equal_query_max_mean"
    ]

for margin, tag in zip(margins, tags):
    variant = {"margin": margin, "per_seed": {}, "per_scene": {}}
    for seed in seeds:
        per_scene = {}
        routes = {}
        for scene in scenes:
            path = os.path.join(
                root, f"seed_{seed}", scene, f"eval_margin_top2_{tag}", "metrics.json"
            )
            per_scene[scene] = row(path)
            routes[scene] = route(path)
        variant["per_seed"][seed] = {
            "mean": {
                metric: statistics.mean(per_scene[scene][metric] for scene in scenes)
                for metric in metric_names
            },
            "scenes": per_scene,
            "routes": routes,
        }
    for scene in scenes:
        variant["per_scene"][scene] = {
            metric: {
                "mean": statistics.mean(
                    variant["per_seed"][seed]["scenes"][scene][metric]
                    for seed in seeds
                ),
                "population_std": statistics.pstdev(
                    variant["per_seed"][seed]["scenes"][scene][metric]
                    for seed in seeds
                ),
            }
            for metric in metric_names
        }
    variant["aggregate"] = {}
    for metric in metric_names:
        values = [variant["per_seed"][seed]["mean"][metric] for seed in seeds]
        variant["aggregate"][metric] = {
            "mean": statistics.mean(values),
            "population_std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean_minus_a20": statistics.mean(values) - a20[metric],
        }
    tie_values = [
        variant["per_seed"][seed]["routes"][scene][
            "mean_tie_blended_fraction_covered"
        ]
        for seed in seeds for scene in scenes
    ]
    variant["mean_tie_blended_fraction_covered"] = statistics.mean(tie_values)
    summary["variants"][tag] = variant

reference_values = {
    metric: [
        summary["reference_equal_query_max"][seed][metric] for seed in seeds
    ]
    for metric in metric_names
}
reference_stats = {
    metric: {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }
    for metric, values in reference_values.items()
}
summary["reference_equal_query_max_aggregate"] = reference_stats

accepted = []
for tag, variant in summary["variants"].items():
    stats = variant["aggregate"]
    checks = {
        "miou_within_0_1_point_of_reference": (
            stats["mIoU"]["mean"] >= reference_stats["mIoU"]["mean"] - 0.001
        ),
        "acc25_within_0_25_point_of_reference": (
            stats["mAcc@0.25"]["mean"]
            >= reference_stats["mAcc@0.25"]["mean"] - 0.0025
        ),
        "acc5_std_reduced_by_25_percent": (
            stats["mAcc@0.5"]["population_std"]
            <= 0.75 * reference_stats["mAcc@0.5"]["population_std"]
        ),
        "worst_acc5_not_below_reference": (
            stats["mAcc@0.5"]["minimum"]
            >= reference_stats["mAcc@0.5"]["minimum"] - 1e-12
        ),
    }
    checks["accepted"] = all(checks.values())
    variant["acceptance"] = checks
    if checks["accepted"]:
        accepted.append(tag)

summary["decision"] = {
    "accepted_variants": accepted,
    "selected_variant": max(
        accepted,
        key=lambda tag: summary["variants"][tag]["aggregate"]["mIoU"]["mean"],
    ) if accepted else None,
    "best_variant_by_mean_miou": max(
        tags,
        key=lambda tag: summary["variants"][tag]["aggregate"]["mIoU"]["mean"],
    ),
}

with open(os.path.join(root, "cross_seed_margin_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A34 margin-aware equal-token validation complete: $RUN_ROOT"
