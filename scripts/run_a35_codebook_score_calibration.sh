#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A35 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-$ROOT}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
ENV_SCRIPT=${ENV_SCRIPT:-$ROOT/scripts/drsplat_env.sh}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A35}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A35}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2"}
SEEDS=${SEEDS:-"20260717 20260718 20260719"}
MEMORY_ROOTS=${MEMORY_ROOTS:-"$ROOT/runs/a30_equal_four_token_query_fusion_20260718_101242 $ROOT/runs/a32_equal_four_token_seed_replication_20260718_112041 $ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948"}
TEATIME_MEMORY_ROOT=${TEATIME_MEMORY_ROOT:-$ROOT/runs/a31_teatime_equal_four_token_validation_20260718_104609}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
GPU_WAIT_TIMEOUT=${GPU_WAIT_TIMEOUT:-21600}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
if [[ "$ENV_SCRIPT" != "none" ]]; then
  source "$ENV_SCRIPT"
fi
SITE=${SITE:-$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=20260717 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

memory_for_seed() {
  local requested=$1
  read -r -a seed_values <<< "$SEEDS"
  read -r -a memory_values <<< "$MEMORY_ROOTS"
  for index in "${!seed_values[@]}"; do
    if [[ "${seed_values[$index]}" == "$requested" ]]; then
      printf '%s\n' "${memory_values[$index]}"
      return
    fi
  done
  return 1
}

evaluate_scene() {
  local scene=$1
  local seed=$2
  local memory_root=$3
  local mode=$4
  local output=$RUN_ROOT/seed_${seed}/$scene/eval_${mode}
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" -m "$GEOMETRY_ROOT/$scene" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory_root/$scene/equal_four_token_memory" \
    --group_topk 4 --group_readout "$mode" \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --label_dir "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output" \
    > "$LOG_DIR/${scene}_seed_${seed}_${mode}.log" 2>&1
}

run_scene() {
  local scene=$1
  read -r -a seed_values <<< "$SEEDS"
  for seed in "${seed_values[@]}"; do
    local memory_root
    memory_root=$(memory_for_seed "$seed")
    evaluate_scene "$scene" "$seed" "$memory_root" equal_query_percentile_max
    evaluate_scene "$scene" "$seed" "$memory_root" equal_query_tail_max
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  for scene in "$@"; do run_scene "$scene"; done
  exit 0
fi
if [[ "${1:-}" == "--teatime-worker" ]]; then
  evaluate_scene teatime 20260717 "$TEATIME_MEMORY_ROOT" equal_query_percentile_max
  evaluate_scene teatime 20260717 "$TEATIME_MEMORY_ROOT" equal_query_tail_max
  exit 0
fi

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/tests/test_semantic_hypothesis_routing.py" \
  "$GPU_GUARD"; do
  [[ -f "$required" ]] || { echo "Missing required source: $required" >&2; exit 2; }
done

read -r -a seed_values <<< "$SEEDS"
read -r -a memory_values <<< "$MEMORY_ROOTS"
[[ "${#seed_values[@]}" -eq "${#memory_values[@]}" ]] || {
  echo "SEEDS and MEMORY_ROOTS must have equal lengths" >&2
  exit 2
}

for index in "${!seed_values[@]}"; do
  seed=${seed_values[$index]}
  memory_root=${memory_values[$index]}
  [[ -f "$memory_root/three_scene_summary.json" ]] || {
    echo "Missing seed summary: $memory_root" >&2
    exit 2
  }
  for scene in $SCENES; do
    manifest=$memory_root/$scene/equal_four_token_memory/manifest.json
    [[ -f "$manifest" ]] || { echo "Missing memory: $manifest" >&2; exit 2; }
    "$PYTHON_BIN" - "$manifest" "$seed" <<'PY'
import json
import os
import sys

import numpy as np

manifest = json.load(open(sys.argv[1]))
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert manifest["resident_slots_required"] == 4
assert manifest["reproducibility"]["seed"] == int(sys.argv[2])
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [
    2048, 4096, 8192, 16384
]
parents = np.load(os.path.join(os.path.dirname(sys.argv[1]), manifest["group_parent_ids"]))
assert np.all(parents == -1)
PY
  done
done

for scene in $SCENES teatime; do
  for required in \
    "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done
[[ -f "$TEATIME_MEMORY_ROOT/teatime/equal_four_token_memory/manifest.json" ]] || {
  echo "Missing teatime memory" >&2
  exit 2
}

(
cd "$SOURCE_DIR"
"$PYTHON_BIN" - <<'PY'
import torch

from semantic_hypothesis_routing import fuse_calibrated_equal_query_tokens

base = torch.tensor([[0.2], [0.3]])
candidates = torch.tensor([[0.81, 0.78], [0.75, 0.70]])
memberships = torch.ones_like(candidates)
valid = torch.tensor([[True, True], [False, False]])

percentile_output, percentile_stats = fuse_calibrated_equal_query_tokens(
    base,
    candidates,
    torch.tensor([[0.70, 0.99], [0.0, 0.0]]),
    memberships,
    memberships,
    valid,
    "percentile",
    0.05,
)
assert torch.allclose(percentile_output, torch.tensor([[0.78], [0.3]]))
assert percentile_stats["covered_points"] == 1
assert percentile_stats["fallback_points"] == 1

tail_output, tail_stats = fuse_calibrated_equal_query_tokens(
    base[:1],
    candidates[:1],
    torch.tensor([[4.0, 4.4]]),
    memberships[:1],
    torch.tensor([[1.0, 0.5]]),
    valid[:1],
    "tail_evidence",
    0.05,
)
assert torch.allclose(tail_output, torch.tensor([[0.81]]))
assert tail_stats["level_score_calibration"] == "tail_evidence"
print("A35_CONTRACT_OK")
PY
) > "$LOG_DIR/unit_tests.log" 2>&1

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -ge 1 && "${#gpus[@]}" -le 2 ]] || {
  echo "A35 expects one or two isolated GPUs" >&2
  exit 2
}
script_path=${BASH_SOURCE[0]}
pids=()
if [[ "${#gpus[@]}" -eq 1 ]]; then
  "$PYTHON_BIN" "$GPU_GUARD" --gpu "${gpus[0]}" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout "$GPU_WAIT_TIMEOUT" --poll-interval 5 -- \
    bash "$script_path" --worker "${scenes[@]}" \
    > "$LOG_DIR/worker_gpu_${gpus[0]}.log" 2>&1 &
  pids+=("$!")
else
  "$PYTHON_BIN" "$GPU_GUARD" --gpu "${gpus[0]}" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout "$GPU_WAIT_TIMEOUT" --poll-interval 5 -- \
    bash "$script_path" --worker "${scenes[0]}" "${scenes[2]}" \
    > "$LOG_DIR/worker_gpu_${gpus[0]}.log" 2>&1 &
  pids+=("$!")
  "$PYTHON_BIN" "$GPU_GUARD" --gpu "${gpus[1]}" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout "$GPU_WAIT_TIMEOUT" --poll-interval 5 -- \
    bash "$script_path" --worker "${scenes[1]}" \
    > "$LOG_DIR/worker_gpu_${gpus[1]}.log" 2>&1 &
  pids+=("$!")
fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

teatime_gpu=${gpus[$((${#gpus[@]} - 1))]}
"$PYTHON_BIN" "$GPU_GUARD" --gpu "$teatime_gpu" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout "$GPU_WAIT_TIMEOUT" --poll-interval 5 -- \
  bash "$script_path" --teatime-worker \
  > "$LOG_DIR/worker_teatime_gpu_${teatime_gpu}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$A20_ROOT" "$SEEDS" "$MEMORY_ROOTS" \
  "$SELECTION_THRESHOLD" $SCENES <<'PY'
import json
import os
import statistics
import sys

root, a20_root, raw_seeds, raw_memories, raw_threshold, *scenes = sys.argv[1:]
seeds = raw_seeds.split()
memories = raw_memories.split()
threshold = float(raw_threshold)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")
modes = ("equal_query_percentile_max", "equal_query_tail_max")


def row(path):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metrics}


def mean_rows(rows):
    return {name: statistics.mean(item[name] for item in rows) for name in metrics}


references = {
    seed: json.load(open(os.path.join(memory, "three_scene_summary.json")))
    for seed, memory in zip(seeds, memories)
}
a20 = next(iter(references.values()))["a20_mean"]
summary = {
    "method": "A35 complete-codebook calibrated four-token retrieval",
    "seeds": seeds,
    "scenes": scenes,
    "selection_threshold": threshold,
    "query_temperature": 0.05,
    "a20_mean": a20,
    "reference_equal_query_max": {
        seed: references[seed]["a30_equal_query_max_mean"] for seed in seeds
    },
    "variants": {},
}

for mode in modes:
    variant = {"per_seed": {}, "per_scene": {}}
    for seed in seeds:
        scene_rows = {
            scene: row(os.path.join(root, f"seed_{seed}", scene, f"eval_{mode}", "metrics.json"))
            for scene in scenes
        }
        variant["per_seed"][seed] = {
            "mean": mean_rows(list(scene_rows.values())),
            "scenes": scene_rows,
        }
    for scene in scenes:
        variant["per_scene"][scene] = {
            name: {
                "mean": statistics.mean(
                    variant["per_seed"][seed]["scenes"][scene][name] for seed in seeds
                ),
                "population_std": statistics.pstdev(
                    variant["per_seed"][seed]["scenes"][scene][name] for seed in seeds
                ),
            }
            for name in metrics
        }
    variant["aggregate"] = {
        name: {
            "mean": statistics.mean(
                variant["per_seed"][seed]["mean"][name] for seed in seeds
            ),
            "population_std": statistics.pstdev(
                variant["per_seed"][seed]["mean"][name] for seed in seeds
            ),
            "mean_minus_a20": statistics.mean(
                variant["per_seed"][seed]["mean"][name] for seed in seeds
            ) - float(a20[name]),
        }
        for name in metrics
    }
    acc5 = variant["aggregate"]["mAcc@0.5"]
    variant["acceptance"] = {
        "miou_at_least_47": variant["aggregate"]["mIoU"]["mean"] >= 0.47,
        "acc5_at_least_a33": acc5["mean"] >= 0.5031730788773042,
        "acc5_std_below_0_8_point": acc5["population_std"] <= 0.008,
    }
    variant["acceptance"]["accepted"] = all(variant["acceptance"].values())
    summary["variants"][mode] = variant

teatime = {
    mode: row(os.path.join(root, "seed_20260717", "teatime", f"eval_{mode}", "metrics.json"))
    for mode in modes
}
summary["teatime_seed_20260717"] = teatime
summary["four_scene_seed_20260717"] = {
    mode: mean_rows(
        [summary["variants"][mode]["per_seed"]["20260717"]["scenes"][scene] for scene in scenes]
        + [teatime[mode]]
    )
    for mode in modes
}
with open(os.path.join(root, "cross_seed_calibration_summary.json"), "w") as target:
    json.dump(summary, target, indent=2)
open(os.path.join(root, "PROBE_COMPLETE"), "w").close()
print(json.dumps(summary, indent=2))
PY
