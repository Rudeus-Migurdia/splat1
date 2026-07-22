#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
GPU_GUARD=${GPU_GUARD:?GPU_GUARD is required}
DATA_ROOT=${DATA_ROOT:-$ROOT}
GEOMETRY_ROOT=${GEOMETRY_ROOT:?GEOMETRY_ROOT is required}
A14_DISC_ROOT=${A14_DISC_ROOT:?A14_DISC_ROOT is required}
MEMORY_ROOTS=${MEMORY_ROOTS:?MEMORY_ROOTS is required}
TEATIME_MEMORY_ROOT=${TEATIME_MEMORY_ROOT:?TEATIME_MEMORY_ROOT is required}
SEEDS=${SEEDS:-"20260717 20260718 20260719"}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU=${GPU:-1}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}

SITE=${SITE:-$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:${PYTHONPATH:-}"
export PYTHONHASHSEED=20260717 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$RUN_ROOT" "$LOG_DIR"

memory_for_seed() {
  local requested=$1
  read -r -a seeds <<< "$SEEDS"
  read -r -a memories <<< "$MEMORY_ROOTS"
  for index in "${!seeds[@]}"; do
    [[ "${seeds[$index]}" == "$requested" ]] && {
      printf '%s\n' "${memories[$index]}"
      return
    }
  done
  return 1
}

evaluate() {
  local scene=$1 seed=$2 memory_root=$3
  local output=$RUN_ROOT/seed_${seed}/$scene/eval_equal_query_max
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" -m "$GEOMETRY_ROOT/$scene" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory_root/$scene/equal_four_token_memory" \
    --group_topk 4 --group_readout equal_query_max --group_query_temperature 0.05 \
    --label_dir "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${scene}_seed_${seed}_raw_max.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  for scene in $SCENES; do
    for seed in $SEEDS; do evaluate "$scene" "$seed" "$(memory_for_seed "$seed")"; done
  done
  evaluate teatime 20260717 "$TEATIME_MEMORY_ROOT"
  exit 0
fi

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$MEMORY_ROOTS" "$SEEDS" "$SELECTION_THRESHOLD" $SCENES <<'PY'
import json
import os
import statistics
import sys

root, raw_memories, raw_seeds, raw_threshold, *scenes = sys.argv[1:]
memories, seeds = raw_memories.split(), raw_seeds.split()
threshold = float(raw_threshold)
names = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def row(path):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(item[name]) for name in names}


reference = {
    seed: json.load(open(os.path.join(memory, "three_scene_summary.json")))[
        "a30_equal_query_max_mean"
    ]
    for seed, memory in zip(seeds, memories)
}
per_seed = {}
for seed in seeds:
    scene_rows = {
        scene: row(os.path.join(root, f"seed_{seed}", scene, "eval_equal_query_max", "metrics.json"))
        for scene in scenes
    }
    mean = {name: statistics.mean(item[name] for item in scene_rows.values()) for name in names}
    per_seed[seed] = {
        "mean": mean,
        "scenes": scene_rows,
        "delta_from_171_reference": {
            name: mean[name] - float(reference[seed][name]) for name in names
        },
    }
summary = {
    "method": "A35 same-hardware equal-query-max control",
    "seeds": seeds,
    "scenes": scenes,
    "per_seed": per_seed,
    "aggregate": {
        name: {
            "mean": statistics.mean(per_seed[seed]["mean"][name] for seed in seeds),
            "population_std": statistics.pstdev(per_seed[seed]["mean"][name] for seed in seeds),
            "mean_hardware_delta": statistics.mean(
                per_seed[seed]["delta_from_171_reference"][name] for seed in seeds
            ),
        }
        for name in names
    },
    "teatime_seed_20260717": row(
        os.path.join(root, "seed_20260717", "teatime", "eval_equal_query_max", "metrics.json")
    ),
}
with open(os.path.join(root, "hardware_control_summary.json"), "w") as target:
    json.dump(summary, target, indent=2)
open(os.path.join(root, "PROBE_COMPLETE"), "w").close()
print(json.dumps(summary, indent=2))
PY
