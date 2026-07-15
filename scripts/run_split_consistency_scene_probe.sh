#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
SCENE=${SCENE:?Set SCENE}
TOPK=${TOPK:-45}
TEMPERATURE=${TEMPERATURE:-0.05}
MAX_AUX_WEIGHT=${MAX_AUX_WEIGHT:-1.5}
LOG_DIR=${LOG_DIR:-$ROOT/logs/multiscale_split_consistency_multiscene}

case "$SCENE" in
  figurines|teatime)
    CALIBRATION=frame_minmax
    CALIBRATION_LOW=0
    CALIBRATION_HIGH=100
    ;;
  ramen)
    CALIBRATION=category_percentile
    CALIBRATION_LOW=1
    CALIBRATION_HIGH=99
    ;;
  *)
    echo "Unsupported scene calibration: $SCENE" >&2
    exit 2
    ;;
esac

cd "$ROOT"
source scripts/drsplat_env.sh
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
run_root="$ROOT/runs/multiscale_split_consistency/$SCENE"
base_cache="$run_root/base_split2"
aux_cache="$run_root/l2_split2"
fused="$run_root/fused_w1p5_t005.pt"
artifact="$run_root/fused_w1p5_t005_codebook_k4096x2"
eval_model="$run_root/eval_model"
eval_output="$run_root/eval_codebook_${CALIBRATION}_l${CALIBRATION_LOW}_h${CALIBRATION_HIGH}"
mkdir -p "$LOG_DIR" "$run_root" "$eval_model"

for path in "$dataset" "$labels" "$geometry" \
  "$dataset/language_features" "$dataset/language_features_multiscale"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

prepare_cache() {
  local output=$1
  local feature_dir=$2
  local feature_level=$3
  local tag=$4
  if [[ ! -f "$output/manifest.json" ]]; then
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$output" \
      --geometry_checkpoint "$geometry" \
      --feature_dir "$feature_dir" --feature_level "$feature_level" \
      --semantic_dim 512 --identity_codec \
      --max_pixels_per_view 0 --topk "$TOPK" --raw_contribution_weights \
      --consensus_only --consensus_chunk_pixels 1024 --consensus_splits 2 \
      > "$LOG_DIR/${SCENE}_${tag}_prepare.log" 2>&1
  fi
}

prepare_cache "$base_cache" "$dataset/language_features" 1 base
prepare_cache "$aux_cache" "$dataset/language_features_multiscale" 2 l2

if [[ ! -f "$fused" ]]; then
  "$PYTHON_BIN" -u build_split_consistency_fusion.py \
    --base_consensus "$base_cache/consensus.pt" \
    --aux_consensus "$aux_cache/consensus.pt" \
    --output "$fused" --max_aux_weight "$MAX_AUX_WEIGHT" \
    --temperature "$TEMPERATURE" \
    > "$LOG_DIR/${SCENE}_fusion.log" 2>&1
fi

if [[ ! -f "$artifact/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
    --consensus "$fused" --codes_per_level 4096 4096 \
    --train_samples 262144 --iterations 25 --assignment_chunk 16384 --faiss_gpu \
    --output_dir "$artifact" \
    > "$LOG_DIR/${SCENE}_codebook.log" 2>&1
fi

if [[ ! -f "$eval_output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$eval_model" --geometry_checkpoint "$geometry" \
    --codebook_dir "$artifact" --label_dir "$labels" --rgr_alpha 0 \
    --score_calibration "$CALIBRATION" \
    --calibration_low "$CALIBRATION_LOW" --calibration_high "$CALIBRATION_HIGH" \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$eval_output" \
    > "$LOG_DIR/${SCENE}_eval.log" 2>&1
fi

echo "split-consistency scene probe complete: scene=$SCENE output=$eval_output"
