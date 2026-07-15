#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENES=${SCENES:-waldo_kitchen}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-language_features_multiscale}
FINE_FEATURE_LEVEL=${FINE_FEATURE_LEVEL:-0}
OBJECT_FEATURE_LEVEL=${OBJECT_FEATURE_LEVEL:-3}
FINE_CODES=${FINE_CODES:-"4096 4096"}
OBJECT_CODES=${OBJECT_CODES:-"1024"}
TOPK=${TOPK:-45}
MAX_PIXELS_PER_VIEW=${MAX_PIXELS_PER_VIEW:-32768}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-5000}
OBJECT_FEATURE_WEIGHT=${OBJECT_FEATURE_WEIGHT:-0.5}
OBJECT_LOSS_WEIGHT=${OBJECT_LOSS_WEIGHT:-0.5}
RGR_ALPHA=${RGR_ALPHA:-0.0}
LOG_DIR=${LOG_DIR:-$ROOT/logs/hierarchical_multiscale_codebook}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

fine_tag=${FINE_CODES// /x}
object_tag=${OBJECT_CODES// /x}

for scene in $SCENES; do
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  labels="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  geometry="$ROOT/runs/3dgs/$scene/chkpnt30000.pth"
  feature_dir="$dataset/$FEATURE_DIR_NAME"
  run_root="$ROOT/runs/hierarchical_multiscale_codebook/${scene}_fine${fine_tag}_obj${object_tag}_p${MAX_PIXELS_PER_VIEW}_topk${TOPK}"
  fine_cache="$run_root/fine_cache_l${FINE_FEATURE_LEVEL}"
  object_cache="$run_root/object_cache_l${OBJECT_FEATURE_LEVEL}"
  fine_initial="$run_root/fine_initial"
  object_initial="$run_root/object_initial"
  query_bank="$run_root/query_bank_256.npy"
  trained="$run_root/joint"
  IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$scene")"

  for path in "$dataset" "$labels" "$geometry" "$feature_dir"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  mkdir -p "$run_root"

  if [[ ! -f "$fine_cache/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$fine_cache" \
      --geometry_checkpoint "$geometry" --feature_dir "$feature_dir" \
      --feature_level "$FINE_FEATURE_LEVEL" --semantic_dim 512 --identity_codec \
      --topk "$TOPK" --max_pixels_per_view "$MAX_PIXELS_PER_VIEW" \
      > "$LOG_DIR/${scene}_fine_cache.log" 2>&1
  fi
  if [[ ! -f "$object_cache/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$object_cache" \
      --geometry_checkpoint "$geometry" --feature_dir "$feature_dir" \
      --feature_level "$OBJECT_FEATURE_LEVEL" --semantic_dim 512 --identity_codec \
      --topk "$TOPK" --max_pixels_per_view "$MAX_PIXELS_PER_VIEW" \
      > "$LOG_DIR/${scene}_object_cache.log" 2>&1
  fi
  if [[ ! -f "$fine_initial/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
      --consensus "$fine_cache/consensus.pt" --codes_per_level $FINE_CODES \
      --train_samples 262144 --iterations 25 --assignment_chunk 16384 --faiss_gpu \
      --output_dir "$fine_initial" > "$LOG_DIR/${scene}_fine_initialize.log" 2>&1
  fi
  if [[ ! -f "$object_initial/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
      --consensus "$object_cache/consensus.pt" --codes_per_level $OBJECT_CODES \
      --train_samples 262144 --iterations 25 --assignment_chunk 16384 --faiss_gpu \
      --output_dir "$object_initial" > "$LOG_DIR/${scene}_object_initialize.log" 2>&1
  fi
  if [[ ! -f "$query_bank" ]]; then
    "$PYTHON_BIN" -u build_semantic_query_bank.py \
      --feature_dir "$feature_dir" --num_queries 256 --max_features 200000 \
      --iterations 25 --faiss_gpu --output "$query_bank" \
      > "$LOG_DIR/${scene}_query_bank.log" 2>&1
  fi
  if [[ ! -f "$trained/training_metrics.json" ]]; then
    "$PYTHON_BIN" -u train_hierarchical_gaussian_codebook.py \
      --fine_cache_dir "$fine_cache" --object_cache_dir "$object_cache" \
      --fine_initial_codebook_dir "$fine_initial" --object_initial_codebook_dir "$object_initial" \
      --output "$trained" --iterations "$TRAIN_ITERATIONS" --batch_pixels 4096 \
      --codebook_lr 0.001 --object_feature_weight "$OBJECT_FEATURE_WEIGHT" \
      --object_loss_weight "$OBJECT_LOSS_WEIGHT" --lovo_weight 0.5 \
      --query_bank "$query_bank" --query_kl_weight 0.1 --lovo_query_kl_weight 0.1 \
      > "$LOG_DIR/${scene}_joint_train.log" 2>&1
  fi

  output="$trained/eval_composed_${calibration}_l${calibration_low}_h${calibration_high}"
  if [[ ! -f "$output/metrics.json" ]]; then
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$run_root" --geometry_checkpoint "$geometry" \
      --codebook_dir "$trained/fine_artifact" --object_codebook_dir "$trained/object_artifact" \
      --object_feature_weight "$OBJECT_FEATURE_WEIGHT" --label_dir "$labels" \
      --rgr_alpha "$RGR_ALPHA" --score_calibration "$calibration" \
      --calibration_low "$calibration_low" --calibration_high "$calibration_high" \
      --thresholds $THRESHOLDS --output "$output" \
      > "$LOG_DIR/${scene}_composed_eval.log" 2>&1
  fi
done

echo "hierarchical multiscale codebook probe complete: scenes=$SCENES"
