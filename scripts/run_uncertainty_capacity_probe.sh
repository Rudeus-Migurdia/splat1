#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SCENE=${SCENE:?Set SCENE}
VARIANT=${VARIANT:?Set VARIANT to a7.0, a7.1, a7.2, or a7.3}
RUN_TAG=${RUN_TAG:-$VARIANT}
GPU_ID=${GPU_ID:-0}
NUM_CODES=${NUM_CODES:-16384}
FINE_FRACTION=${FINE_FRACTION:-0.15}
FINE_WEIGHT=${FINE_WEIGHT:-0.50}
FINE_SCORE_MODE=${FINE_SCORE_MODE:-stability}
LOG_DIR=${LOG_DIR:-$ROOT/logs/uncertainty_capacity}

case "$SCENE" in
  figurines|teatime)
    CALIBRATION=frame_minmax; CALIBRATION_LOW=0; CALIBRATION_HIGH=100 ;;
  ramen|waldo_kitchen)
    CALIBRATION=category_percentile; CALIBRATION_LOW=1; CALIBRATION_HIGH=99 ;;
  *) echo "Unsupported scene: $SCENE" >&2; exit 2 ;;
esac

cd "$ROOT"
source scripts/drsplat_env.sh
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
if [[ "$SCENE" == waldo_kitchen ]]; then
  source_root="$ROOT/runs/multiscale_split_consistency"
  base_cache="$source_root/waldo_base_split2"
  l2_cache="$source_root/waldo_l2_split2"
else
  source_root="$ROOT/runs/multiscale_split_consistency/$SCENE"
  base_cache="$source_root/base_split2"
  l2_cache="$source_root/l2_split2"
fi
run_root="$ROOT/runs/uncertainty_capacity/$SCENE"
l1_cache="$run_root/l1_split2"
consensus="$run_root/${RUN_TAG}.pt"
artifact="$run_root/${RUN_TAG}_shared_k${NUM_CODES}"
shared_base="$run_root/a7.0_shared_k${NUM_CODES}"
eval_model="$run_root/eval_model_${RUN_TAG}"
eval_output="$run_root/eval_${RUN_TAG}_shared_k${NUM_CODES}_${CALIBRATION}_l${CALIBRATION_LOW}_h${CALIBRATION_HIGH}"
mkdir -p "$LOG_DIR" "$run_root" "$eval_model"

for path in "$dataset" "$labels" "$geometry" "$base_cache/consensus.pt" \
  "$l2_cache/consensus.pt" "$dataset/language_features_multiscale"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

if [[ ! -f "$l1_cache/manifest.json" ]]; then
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$l1_cache" --geometry_checkpoint "$geometry" \
    --feature_dir "$dataset/language_features_multiscale" --feature_level 1 \
    --semantic_dim 512 --identity_codec --max_pixels_per_view 0 --topk 45 \
    --raw_contribution_weights --consensus_only --consensus_chunk_pixels 1024 \
    --consensus_splits 2 > "$LOG_DIR/${SCENE}_l1_prepare.log" 2>&1
fi

if [[ ! -f "$consensus" ]]; then
  "$PYTHON_BIN" -u build_uncertainty_capacity_fusion.py \
    --base_consensus "$base_cache/consensus.pt" \
    --aux_consensus "$l2_cache/consensus.pt" \
    --fine_consensus "$l1_cache/consensus.pt" --output "$consensus" \
    --variant "$VARIANT" --max_aux_weight 1.5 --temperature 0.05 \
    --fallback_reliability 0.50 --fallback_margin 0.03 \
    --fallback_ambiguous_ceiling 0.75 --fine_min_reliability 0.60 \
    --fine_min_disagreement 0.10 --fine_fraction "$FINE_FRACTION" \
    --fine_weight "$FINE_WEIGHT" --fine_score_mode "$FINE_SCORE_MODE" \
    > "$LOG_DIR/${SCENE}_${RUN_TAG}_fusion.log" 2>&1
fi

if [[ "$VARIANT" == a7.0 ]]; then
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$consensus" --num_codes "$NUM_CODES" --min_ids 2 --max_ids 2 \
      --min_cosine_gain 0 --target_cosine 1 --train_samples 262144 \
      --iterations 25 --assignment_chunk 4096 --faiss_gpu --output_dir "$artifact" \
      > "$LOG_DIR/${SCENE}_${RUN_TAG}_codebook.log" 2>&1
  fi
else
  [[ -f "$shared_base/manifest.json" ]] || {
    echo "Run VARIANT=a7.0 first to establish the shared vocabulary" >&2; exit 1;
  }
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_adaptive_codebook.py \
      --consensus "$consensus" --codebook "$shared_base/codebook_shared.npy" \
      --num_codes "$NUM_CODES" --min_ids 2 --max_ids 3 --use_consensus_capacity \
      --fill_consensus_capacity \
      --min_cosine_gain 0.002 --target_cosine 0.995 --assignment_chunk 4096 \
      --faiss_gpu --output_dir "$artifact" \
      > "$LOG_DIR/${SCENE}_${RUN_TAG}_codebook.log" 2>&1
  fi
fi

if [[ ! -f "$eval_output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$eval_model" --geometry_checkpoint "$geometry" \
    --codebook_dir "$artifact" --label_dir "$labels" --rgr_alpha 0 \
    --score_calibration "$CALIBRATION" --calibration_low "$CALIBRATION_LOW" \
    --calibration_high "$CALIBRATION_HIGH" \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$eval_output" > "$LOG_DIR/${SCENE}_${RUN_TAG}_eval.log" 2>&1
fi

echo "uncertainty-capacity probe complete: scene=$SCENE variant=$VARIANT output=$eval_output"
