#!/usr/bin/env bash
set -euo pipefail

# A41: identify useful SFS-filtered levels by composing whole independently
# trained A33/A40 level memories.  Four token slots remain peer residents.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
SCENE=${SCENE:-ramen}
GPU=${GPU:-4}
SEED=${SEED:-20260719}
A33_MEMORY=${A33_MEMORY:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/equal_four_token_memory}
A40_MEMORY=${A40_MEMORY:-$ROOT/runs/a40_post_aggregation_filter_20260719_162740/$SCENE/filtered_four_token_memory}
A33_METRICS=${A33_METRICS:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/eval_equal_query_max/metrics.json}
A40_METRICS=${A40_METRICS:-$ROOT/runs/a40_post_aggregation_filter_20260719_162740/$SCENE/eval_equal_query_max/metrics.json}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
LABEL_ROOT=${LABEL_ROOT:-$ROOT/drsplat_data/lerf_ovs/label}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

compose_variant() {
  local name=$1 l0=$2 l1=$3 l2=$4 l3=$5
  local memory=$RUN_ROOT/$SCENE/$name/memory
  [[ -f "$memory/manifest.json" ]] && return
  "$PYTHON_BIN" "$SOURCE_DIR/compose_independent_hierarchical_memory.py" \
    --level_memory_dirs "$l0" "$l1" "$l2" "$l3" \
    --output_dir "$memory" --seed "$SEED" \
    > "$LOG_DIR/${name}_compose.log" 2>&1
  "$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
    --artifact_dir "$memory" --required base sam_l0 sam_l1 sam_l2 sam_l3 \
    > "$LOG_DIR/${name}_contract.log" 2>&1
}

evaluate_variant() {
  local name=$1
  local memory=$RUN_ROOT/$SCENE/$name/memory
  local output=$RUN_ROOT/$SCENE/$name/eval
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/$SCENE" -m "$GEOMETRY_ROOT/$SCENE" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory" --group_topk 4 \
    --group_readout equal_query_max --group_query_temperature 0.05 \
    --label_dir "$LABEL_ROOT/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${name}_eval.log" 2>&1
}

run_worker() {
  compose_variant filtered_l0 "$A40_MEMORY" "$A33_MEMORY" "$A33_MEMORY" "$A33_MEMORY"
  compose_variant filtered_l1 "$A33_MEMORY" "$A40_MEMORY" "$A33_MEMORY" "$A33_MEMORY"
  compose_variant filtered_l2 "$A33_MEMORY" "$A33_MEMORY" "$A40_MEMORY" "$A33_MEMORY"
  compose_variant filtered_l3 "$A33_MEMORY" "$A33_MEMORY" "$A33_MEMORY" "$A40_MEMORY"
  compose_variant filtered_l0_l2 "$A40_MEMORY" "$A33_MEMORY" "$A40_MEMORY" "$A33_MEMORY"
  compose_variant filtered_l0_l3 "$A40_MEMORY" "$A33_MEMORY" "$A33_MEMORY" "$A40_MEMORY"
  compose_variant filtered_l0_l2_l3 "$A40_MEMORY" "$A33_MEMORY" "$A40_MEMORY" "$A40_MEMORY"
  local name
  for name in filtered_l0 filtered_l1 filtered_l2 filtered_l3 \
    filtered_l0_l2 filtered_l0_l3 filtered_l0_l2_l3; do
    evaluate_variant "$name"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for required in \
  "$SOURCE_DIR/compose_independent_hierarchical_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" \
  "$A33_MEMORY/manifest.json" "$A40_MEMORY/manifest.json" \
  "$A33_METRICS" "$A40_METRICS" \
  "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" "$LABEL_ROOT/$SCENE" \
  "$A14_DISC_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids/manifest.json"; do
  [[ -e "$required" ]] || { echo "Missing A41 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$SCENE" "$A33_METRICS" "$A40_METRICS" "$SEED" <<'PY'
import json
import os
import sys

root, scene, a33_path, a40_path, raw_seed = sys.argv[1:]
variants = (
    "filtered_l0",
    "filtered_l1",
    "filtered_l2",
    "filtered_l3",
    "filtered_l0_l2",
    "filtered_l0_l3",
    "filtered_l0_l2_l3",
)


def row(path):
    payload = json.load(open(path))
    return next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - 0.55) < 1e-8
    )


a33 = row(a33_path)
a40 = row(a40_path)
summary = {
    "method": "A41 whole-level selective post-filter memory composition",
    "scene": scene,
    "seed": int(raw_seed),
    "selection_threshold": 0.55,
    "control_a33": {key: a33[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
    "control_a40": {key: a40[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
    "variants": {},
}
for name in variants:
    current = row(os.path.join(root, scene, name, "eval", "metrics.json"))
    metrics = {key: current[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}
    summary["variants"][name] = {
        "metrics": metrics,
        "delta_from_a33": {key: metrics[key] - a33[key] for key in metrics},
        "per_category_delta_from_a33": {
            category: current["per_category"][category] - a33["per_category"][category]
            for category in a33["per_category"]
        },
    }
best_name = max(variants, key=lambda name: summary["variants"][name]["metrics"]["mIoU"])
best = summary["variants"][best_name]
summary["best_variant"] = best_name
summary["best_metrics"] = best["metrics"]
summary["decision"] = {
    "beats_a33_miou": best["delta_from_a33"]["mIoU"] > 0.0,
    "improves_miou_by_half_point": best["delta_from_a33"]["mIoU"] >= 0.005,
    "preserves_strict_accuracy": best["delta_from_a33"]["mAcc@0.5"] >= 0.0,
}
summary["decision"]["promote_best"] = all(summary["decision"].values())
with open(os.path.join(root, "ramen_level_ablation_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A41 Ramen level-selective filter probe complete: $RUN_ROOT"
