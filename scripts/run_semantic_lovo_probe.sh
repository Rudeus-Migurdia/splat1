#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
GPU_ID=${GPU_ID:-0}
SCENES=${SCENES:-"figurines ramen"}
SEMANTIC_DIM=${SEMANTIC_DIM:-16}
TOPK=${TOPK:-8}
MAX_PIXELS_PER_VIEW=${MAX_PIXELS_PER_VIEW:-16384}
MAX_VIEWS=${MAX_VIEWS:-0}
CODEC_EPOCHS=${CODEC_EPOCHS:-15}
CODEC_HIDDEN_DIMS=${CODEC_HIDDEN_DIMS:-"256 128"}
CODEC_LR=${CODEC_LR:-0.0003}
MIN_CODEC_COSINE=${MIN_CODEC_COSINE:-0.9}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-5000}
BATCH_PIXELS=${BATCH_PIXELS:-4096}
LOVO_WEIGHT=${LOVO_WEIGHT:-0.5}
LOVO_TOPK=${LOVO_TOPK:-4}
NUISANCE_RANK=${NUISANCE_RANK:-4}
RUN_PREPARE=${RUN_PREPARE:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
FORCE=${FORCE:-0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/semantic_lovo_probe}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
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
  cache="$ROOT/runs/semantic_core/${scene}_d${SEMANTIC_DIM}_k${TOPK}_p${MAX_PIXELS_PER_VIEW}/cache"
  baseline="$ROOT/runs/semantic_core/${scene}_d${SEMANTIC_DIM}_k${TOPK}_p${MAX_PIXELS_PER_VIEW}/baseline"
  core="$ROOT/runs/semantic_core/${scene}_d${SEMANTIC_DIM}_k${TOPK}_p${MAX_PIXELS_PER_VIEW}/lovo${LOVO_WEIGHT}_nuisance${NUISANCE_RANK}_lt${LOVO_TOPK}"
  IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$scene")"

  for path in "$dataset" "$geometry" "$labels"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done

  if [[ "$RUN_PREPARE" == "1" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$cache" \
      --geometry_checkpoint "$geometry" \
      --feature_level 1 \
      --semantic_dim "$SEMANTIC_DIM" \
      --codec_hidden_dims $CODEC_HIDDEN_DIMS \
      --codec_epochs "$CODEC_EPOCHS" \
      --codec_lr "$CODEC_LR" \
      --min_codec_validation_cosine "$MIN_CODEC_COSINE" \
      --max_pixels_per_view "$MAX_PIXELS_PER_VIEW" \
      --max_views "$MAX_VIEWS" \
      --topk "$TOPK" \
      "${force_args[@]}" \
      > "$LOG_DIR/${scene}_prepare.log" 2>&1
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    "$PYTHON_BIN" -u train_semantic_field.py \
      --cache_dir "$cache" \
      --output "$baseline" \
      --iterations "$TRAIN_ITERATIONS" \
      --batch_pixels "$BATCH_PIXELS" \
      --lovo_weight 0 \
      --nuisance_rank 0 \
      "${force_args[@]}" \
      > "$LOG_DIR/${scene}_baseline_train.log" 2>&1

    "$PYTHON_BIN" -u train_semantic_field.py \
      --cache_dir "$cache" \
      --output "$core" \
      --iterations "$TRAIN_ITERATIONS" \
      --batch_pixels "$BATCH_PIXELS" \
      --lovo_weight "$LOVO_WEIGHT" \
      --lovo_topk "$LOVO_TOPK" \
      --nuisance_rank "$NUISANCE_RANK" \
      "${force_args[@]}" \
      > "$LOG_DIR/${scene}_core_train.log" 2>&1
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    for variant in baseline core; do
      if [[ "$variant" == "baseline" ]]; then
        artifact_dir="$baseline"
      else
        artifact_dir="$core"
      fi
      output="$artifact_dir/eval_${calibration}_l${calibration_low}_h${calibration_high}"
      "$PYTHON_BIN" -u eval_lerf_ovs_semantic_field_miou.py \
        -s "$dataset" -m "$cache" \
        --geometry_checkpoint "$geometry" \
        --semantic_artifact "$artifact_dir/semantic_field.pt" \
        --label_dir "$labels" \
        --score_calibration "$calibration" \
        --calibration_low "$calibration_low" \
        --calibration_high "$calibration_high" \
        --thresholds $THRESHOLDS \
        --output "$output" \
        > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    done
  fi
done

echo "semantic LOVO probe complete: scenes=$SCENES"
