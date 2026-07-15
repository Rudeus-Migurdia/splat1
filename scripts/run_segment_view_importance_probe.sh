#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENES=${SCENES:-"figurines waldo_kitchen"}
SEMANTIC_DIM=${SEMANTIC_DIM:-64}
TOPK=${TOPK:-16}
MAX_PIXELS_PER_VIEW=${MAX_PIXELS_PER_VIEW:-32768}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-5000}
BATCH_PIXELS=${BATCH_PIXELS:-4096}
LOVO_WEIGHT=${LOVO_WEIGHT:-0.5}
LOVO_TOPK=${LOVO_TOPK:-4}
NUISANCE_RANK=${NUISANCE_RANK:-4}
IMPORTANCE_GROUPS=${IMPORTANCE_GROUPS:-16}
IMPORTANCE_TEMPERATURE=${IMPORTANCE_TEMPERATURE:-1.0}
IMPORTANCE_UNIFORM_MIX=${IMPORTANCE_UNIFORM_MIX:-0.25}
IMPORTANCE_MAX_STEP_KL=${IMPORTANCE_MAX_STEP_KL:-0.02}
IMPORTANCE_MAX_BASE_KL=${IMPORTANCE_MAX_BASE_KL:-0.5}
IMPORTANCE_UPDATE_INTERVAL=${IMPORTANCE_UPDATE_INTERVAL:-100}
IMPORTANCE_EMA_DECAY=${IMPORTANCE_EMA_DECAY:-0.95}
IMPORTANCE_RATIO_CLIP=${IMPORTANCE_RATIO_CLIP:-5.0}
IMPORTANCE_RARITY_WEIGHT=${IMPORTANCE_RARITY_WEIGHT:-0.1}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
RUN_QUERY_EVAL=${RUN_QUERY_EVAL:-1}
FORCE=${FORCE:-0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/segment_view_importance}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args+=(--force)
fi

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

for scene in $SCENES; do
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  geometry="$ROOT/runs/3dgs/$scene/chkpnt30000.pth"
  labels="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  run_root="$ROOT/runs/semantic_core/${scene}_d${SEMANTIC_DIM}_k${TOPK}_p${MAX_PIXELS_PER_VIEW}"
  cache="$run_root/cache"
  output="$run_root/lovo${LOVO_WEIGHT}_nuisance${NUISANCE_RANK}_lt${LOVO_TOPK}_svi_g${IMPORTANCE_GROUPS}${OUTPUT_SUFFIX}"
  IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$scene")"

  for path in "$dataset" "$geometry" "$labels" "$cache/manifest.json"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done

  if [[ "$RUN_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -u train_semantic_field.py \
      --cache_dir "$cache" \
      --output "$output" \
      --iterations "$TRAIN_ITERATIONS" \
      --batch_pixels "$BATCH_PIXELS" \
      --lovo_weight "$LOVO_WEIGHT" \
      --lovo_topk "$LOVO_TOPK" \
      --nuisance_rank "$NUISANCE_RANK" \
      --view_sampling segment_importance \
      --importance_groups "$IMPORTANCE_GROUPS" \
      --importance_temperature "$IMPORTANCE_TEMPERATURE" \
      --importance_uniform_mix "$IMPORTANCE_UNIFORM_MIX" \
      --importance_max_step_kl "$IMPORTANCE_MAX_STEP_KL" \
      --importance_max_base_kl "$IMPORTANCE_MAX_BASE_KL" \
      --importance_update_interval "$IMPORTANCE_UPDATE_INTERVAL" \
      --importance_ema_decay "$IMPORTANCE_EMA_DECAY" \
      --importance_ratio_clip "$IMPORTANCE_RATIO_CLIP" \
      --importance_rarity_weight "$IMPORTANCE_RARITY_WEIGHT" \
      "${force_args[@]}" \
      > "$LOG_DIR/${scene}_train.log" 2>&1
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    eval_output="$output/eval_${calibration}_l${calibration_low}_h${calibration_high}"
    "$PYTHON_BIN" -u eval_lerf_ovs_semantic_field_miou.py \
      -s "$dataset" -m "$cache" \
      --geometry_checkpoint "$geometry" \
      --semantic_artifact "$output/semantic_field.pt" \
      --label_dir "$labels" \
      --score_calibration "$calibration" \
      --calibration_low "$calibration_low" \
      --calibration_high "$calibration_high" \
      --thresholds $THRESHOLDS \
      --output "$eval_output" \
      > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi

  if [[ "$RUN_QUERY_EVAL" == "1" ]]; then
    "$PYTHON_BIN" -u eval_semantic_field_consistency.py \
      --cache_dir "$cache" \
      --semantic_artifact "$output/semantic_field.pt" \
      --label_dir "$labels" \
      --output "$output/query_consistency.json" \
      --samples_per_view 256 \
      --lovo_topk "$LOVO_TOPK" \
      > "$LOG_DIR/${scene}_query_eval.log" 2>&1
  fi
done

echo "segment-wise view importance probe complete: scenes=$SCENES"
