#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-$ROOT/runs/a6_query_margin_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a6_novelty_joint32k_20260715}
GPU_LIST=${GPU_LIST:-"0 1 2 3"}

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

prepare_local_artifact() {
  local scene=$1
  local mode=$2
  local source=$SOURCE_ROOT/$scene/${mode}_ids
  local output=$RUN_ROOT/$scene/source_artifacts/${mode}_ids
  mkdir -p "$output"
  cp "$source/manifest.json" "$output/manifest.json"
  for name in point_code_ids.npy overflow_point_ids.npy overflow_code_ids.npy overflow_slots.npy overflow_weights.npy valid_mask.npy; do
    ln -sfn "$source/$name" "$output/$name"
  done
  ln -sfn "$SOURCE_ROOT/$scene/joint_vocabulary/codebook_shared.npy" "$output/codebook_shared.npy"
}

build_mask() {
  local scene=$1
  mkdir -p "$RUN_ROOT/$scene"
  prepare_local_artifact "$scene" base
  prepare_local_artifact "$scene" candidate
  if [[ ! -f "$RUN_ROOT/$scene/candidate_mask.json" ]]; then
    "$PYTHON_BIN" build_novelty_route_mask.py \
      --base_consensus "$(base_path "$scene")" \
      --candidate_consensus "$(candidate_path "$scene")" \
      --base_codebook_dir "$RUN_ROOT/$scene/source_artifacts/base_ids" \
      --candidate_codebook_dir "$RUN_ROOT/$scene/source_artifacts/candidate_ids" \
      --noise_ratio 1 \
      --output "$RUN_ROOT/$scene/candidate_mask.npy" \
      > "$LOG_DIR/${scene}_mask.log" 2>&1
  fi
  "$PYTHON_BIN" prune_gaussian_codebook.py \
    --artifact_dir "$RUN_ROOT/$scene/source_artifacts/candidate_ids" \
    --keep_mask "$RUN_ROOT/$scene/candidate_mask.npy" \
    --codebook_path "$SOURCE_ROOT/$scene/joint_vocabulary/codebook_shared.npy" \
    --output_dir "$RUN_ROOT/$scene/pruned_candidate_ids" \
    > "$LOG_DIR/${scene}_prune.log" 2>&1
}

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene/eval
  mkdir -p "$output"
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" \
    -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$RUN_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$RUN_ROOT/$scene/source_artifacts/base_ids" \
    --codebook_query_route query_positive \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --occupancy_threshold 0.7 \
    --output "$output" \
    > "$LOG_DIR/${scene}_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for scene in figurines ramen teatime waldo_kitchen; do
  build_mask "$scene"
done

scenes=(figurines ramen teatime waldo_kitchen)
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -ge 4 ]] || { echo "GPU_LIST must contain four GPUs" >&2; exit 2; }
pids=()
for index in 0 1 2 3; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_a6_novelty_joint32k.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" scripts/summarize_lerf_ovs_paper.py \
  "$RUN_ROOT/figurines/eval/metrics.json" \
  "$RUN_ROOT/ramen/eval/metrics.json" \
  "$RUN_ROOT/teatime/eval/metrics.json" \
  "$RUN_ROOT/waldo_kitchen/eval/metrics.json" \
  --output "$RUN_ROOT/four_scene_metrics.json" \
  > "$RUN_ROOT/four_scene_table.md"

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "A6 novelty-gated joint-32k evaluation complete: $RUN_ROOT"
