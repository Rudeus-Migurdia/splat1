#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-figurines}
GPU_ID=${GPU_ID:-1}
COARSE_CODES=${COARSE_CODES:-32}
FINE_CODES=${FINE_CODES:-8}
BLENDS=${BLENDS:-"0.15 0.30 0.45 0.60"}
BLEND_MODES=${BLEND_MODES:-"fixed query_adaptive"}
AGGREGATIONS=${AGGREGATIONS:-"weighted"}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
EVAL_HIERARCHICAL_RECON=${EVAL_HIERARCHICAL_RECON:-1}
COARSE_MIN_BLEND=${COARSE_MIN_BLEND:-0.05}
COARSE_SPECIFICITY_TOPK=${COARSE_SPECIFICITY_TOPK:-16}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
export WANDB_MODE=offline

DATASET=${DATASET:-$ROOT/drsplat_data/lerf_ovs/$SCENE}
LABEL_DIR=${LABEL_DIR:-$ROOT/drsplat_data/lerf_ovs/label/$SCENE}
if [[ "$SCENE" == "figurines" ]]; then
  OUT=${OUT:-$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens}
else
  OUT=${OUT:-$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256}
fi

CHECKPOINT=${CHECKPOINT:-$OUT/chkpnt0.pth}
ASSIGNMENTS=${ASSIGNMENTS:-$OUT/point_group_assignments.npz}
GROUP_FEATURES=${GROUP_FEATURES:-$OUT/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy}
HIER_DIR=${HIER_DIR:-$OUT/teacher_distilled/hier_k${COARSE_CODES}_f${FINE_CODES}_from_codebook}
LOG_DIR=${LOG_DIR:-$ROOT/logs/coarse_fine_sweep/$SCENE}

mkdir -p "$HIER_DIR" "$LOG_DIR"

if [[ ! -f "$GROUP_FEATURES" ]]; then
  echo "Missing GROUP_FEATURES: $GROUP_FEATURES" >&2
  exit 1
fi
if [[ ! -f "$ASSIGNMENTS" ]]; then
  echo "Missing ASSIGNMENTS: $ASSIGNMENTS" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing CHECKPOINT: $CHECKPOINT" >&2
  exit 1
fi

if [[ -f "$HIER_DIR/coarse_codebook.npy" && -f "$HIER_DIR/coarse_ids.npy" && -f "$HIER_DIR/group_features_hierarchical.npy" ]]; then
  echo "[$(date +%FT%T)] reuse hierarchical coarse/fine tokens: $HIER_DIR"
else
  echo "[$(date +%FT%T)] build hierarchical coarse/fine tokens for $SCENE"
  "$PYTHON_BIN" hierarchical_group_codebook.py \
    --group_features "$GROUP_FEATURES" \
    --assignments "$ASSIGNMENTS" \
    --coarse_codes "$COARSE_CODES" \
    --fine_codes "$FINE_CODES" \
    --iterations 120 \
    --seed 71 \
    --usage_weighted \
    --output_dir "$HIER_DIR" \
    > "$LOG_DIR/00_hierarchical_codebook.log" 2>&1
fi

for aggregation in $AGGREGATIONS; do
  if [[ "$EVAL_HIERARCHICAL_RECON" == "1" ]]; then
    eval_dir="$OUT/eval/lerf_ovs_teacher_codebook_k256_${aggregation}_hier_k${COARSE_CODES}_f${FINE_CODES}"
    echo "[$(date +%FT%T)] eval $SCENE aggregation=$aggregation hierarchical reconstructed tokens"
    "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
      -s "$DATASET" -m "$OUT" \
      --checkpoint "$CHECKPOINT" \
      --label_dir "$LABEL_DIR" \
      --group_features "$HIER_DIR/group_features_hierarchical.npy" \
      --assignments "$ASSIGNMENTS" \
      --aggregation "$aggregation" \
      --score_power 1.0 \
      --query_temperature "$QUERY_TEMPERATURE" \
      --query_prior_power "$QUERY_PRIOR_POWER" \
      --thresholds $THRESHOLDS \
      --output "$eval_dir" \
      > "$LOG_DIR/01_eval_${aggregation}_hier_k${COARSE_CODES}_f${FINE_CODES}.log" 2>&1
  fi

  for mode in $BLEND_MODES; do
    for blend in $BLENDS; do
      blend_tag=${blend/./p}
      eval_dir="$OUT/eval/lerf_ovs_teacher_codebook_k256_${aggregation}_coarse${COARSE_CODES}_${mode}_blend_${blend_tag}"
      echo "[$(date +%FT%T)] eval $SCENE aggregation=$aggregation coarse_blend_mode=$mode coarse_blend=$blend"
      "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
        -s "$DATASET" -m "$OUT" \
        --checkpoint "$CHECKPOINT" \
        --label_dir "$LABEL_DIR" \
        --group_features "$GROUP_FEATURES" \
        --assignments "$ASSIGNMENTS" \
        --coarse_features "$HIER_DIR/coarse_codebook.npy" \
        --group_to_coarse "$HIER_DIR/coarse_ids.npy" \
        --coarse_blend "$blend" \
        --coarse_blend_mode "$mode" \
        --coarse_min_blend "$COARSE_MIN_BLEND" \
        --coarse_specificity_topk "$COARSE_SPECIFICITY_TOPK" \
        --aggregation "$aggregation" \
        --score_power 1.0 \
        --query_temperature "$QUERY_TEMPERATURE" \
        --query_prior_power "$QUERY_PRIOR_POWER" \
        --thresholds $THRESHOLDS \
        --output "$eval_dir" \
        > "$LOG_DIR/01_eval_${aggregation}_${mode}_blend_${blend_tag}.log" 2>&1
    done
  done
done

echo "[$(date +%FT%T)] coarse/fine sweep done for $SCENE"
