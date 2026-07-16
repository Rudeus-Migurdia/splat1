#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/runs/a6_query_margin_joint32k_20260715}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/e9_query_preserving_joint32k_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/e9_query_preserving_joint32k_20260716}
QUERY_BANK_ROOT=${QUERY_BANK_ROOT:-$RUN_ROOT}
GPU_LIST=${GPU_LIST:-"1 2"}
SCENES=${SCENES:-"figurines waldo_kitchen"}
ITERATIONS=${ITERATIONS:-1200}
LEARNING_RATE=${LEARNING_RATE:-0.0002}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

base_path() {
  [[ "$1" == "waldo_kitchen" ]] && printf '%s\n' "$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt" || printf '%s\n' "$ROOT/runs/multiscale_split_consistency/$1/fused_w1p5_t005.pt"
}

candidate_path() {
  [[ "$1" == "waldo_kitchen" ]] && printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt" || printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/consensus_alpha050.pt"
}

cache_path() {
  [[ "$1" == "waldo_kitchen" ]] && printf '%s\n' "$ROOT/runs/query_routing/waldo_multiscale/cache_l2_raw" || printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$1/cache_l2_raw"
}

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene
  local vocabulary=$output/trained_vocabulary
  local query_bank=$QUERY_BANK_ROOT/$scene/training_query_bank.npy
  local base_artifact=$SOURCE_ROOT/$scene/base_ids
  local candidate_artifact=$SOURCE_ROOT/$scene/candidate_ids
  local initial_codebook=$SOURCE_ROOT/$scene/joint_vocabulary/codebook_shared.npy
  mkdir -p "$output"

  "$PYTHON_BIN" -u build_training_semantic_query_bank.py \
    --cache_dir "$(cache_path "$scene")" \
    --num_queries 512 \
    --max_features_per_view 64 \
    --iterations 25 \
    --faiss_gpu \
    --seed 20260716 \
    --output "$query_bank" \
    > "$LOG_DIR/${scene}_query_bank.log" 2>&1

  "$PYTHON_BIN" -u train_joint_query_preserving_vocabulary.py \
    --base_consensus "$(base_path "$scene")" \
    --candidate_consensus "$(candidate_path "$scene")" \
    --base_artifact_dir "$base_artifact" \
    --candidate_artifact_dir "$candidate_artifact" \
    --initial_codebook "$initial_codebook" \
    --query_bank "$query_bank" \
    --iterations "$ITERATIONS" \
    --batch_gaussians 4096 \
    --learning_rate "$LEARNING_RATE" \
    --cosine_weight 1 \
    --query_kl_weight 0.1 \
    --query_margin_weight 0.05 \
    --codebook_anchor_weight 0.01 \
    --prediction_anchor_weight 0.05 \
    --query_temperature 0.07 \
    --validation_samples 65536 \
    --seed 20260716 \
    --output_dir "$vocabulary" \
    > "$LOG_DIR/${scene}_train.log" 2>&1

  "$PYTHON_BIN" remount_shared_codebook.py \
    --artifact_dir "$base_artifact" \
    --codebook_path "$vocabulary/codebook_shared.npy" \
    --training_metrics "$vocabulary/training_metrics.json" \
    --mode base \
    --output_dir "$output/base_ids" \
    > "$LOG_DIR/${scene}_remount_base.log" 2>&1
  "$PYTHON_BIN" remount_shared_codebook.py \
    --artifact_dir "$candidate_artifact" \
    --codebook_path "$vocabulary/codebook_shared.npy" \
    --training_metrics "$vocabulary/training_metrics.json" \
    --mode candidate \
    --output_dir "$output/candidate_ids" \
    > "$LOG_DIR/${scene}_remount_candidate.log" 2>&1

  "$PYTHON_BIN" build_novelty_route_mask.py \
    --base_consensus "$(base_path "$scene")" \
    --candidate_consensus "$(candidate_path "$scene")" \
    --base_codebook_dir "$output/base_ids" \
    --candidate_codebook_dir "$output/candidate_ids" \
    --noise_ratio 1 \
    --output "$output/candidate_mask.npy" \
    > "$LOG_DIR/${scene}_mask.log" 2>&1
  "$PYTHON_BIN" prune_gaussian_codebook.py \
    --artifact_dir "$output/candidate_ids" \
    --keep_mask "$output/candidate_mask.npy" \
    --codebook_path "$vocabulary/codebook_shared.npy" \
    --output_dir "$output/pruned_candidate_ids" \
    > "$LOG_DIR/${scene}_prune.log" 2>&1

  mkdir -p "$output/eval"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$output/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$output/base_ids" \
    --codebook_query_route query_positive \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --occupancy_threshold 0.7 \
    --output "$output/eval" \
    > "$LOG_DIR/${scene}_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || { echo "SCENES and GPU_LIST must have equal lengths" >&2; exit 2; }
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_e9_query_preserving_joint32k.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$BASELINE_ROOT" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, baseline_root, *scenes = sys.argv[1:]
threshold = 0.55
summary = {"fixed_selection_threshold": threshold, "scenes": {}}
for scene in scenes:
    rows = {}
    for name, root in (("e8_3", baseline_root), ("e9", run_root)):
        path = os.path.join(root, scene, "eval", "metrics.json")
        metrics = json.load(open(path))
        row = next(
            item
            for item in metrics["threshold_summary"]
            if abs(item["selection_threshold"] - threshold) < 1e-8
        )
        rows[name] = {
            key: row[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
        }
    rows["delta"] = {key: rows["e9"][key] - rows["e8_3"][key] for key in rows["e9"]}
    summary["scenes"][scene] = rows
for name in ("e8_3", "e9", "delta"):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
with open(os.path.join(run_root, "fixed_threshold_probe.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "E9 query-preserving joint vocabulary probe complete: $RUN_ROOT"
