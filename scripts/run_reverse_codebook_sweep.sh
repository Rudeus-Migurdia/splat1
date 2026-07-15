#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-waldo_kitchen}
GPU_ID=${GPU_ID:-0}
TOP_CODES=${TOP_CODES:-"8 16 32 64 0"}
RESIDUAL_WEIGHTS=${RESIDUAL_WEIGHTS:-"0.0 0.5 1.0"}
CODE_BLENDS=${CODE_BLENDS:-"0.0 0.25"}
AGGREGATIONS=${AGGREGATIONS:-"query_softmax weighted"}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.07}
QUERY_PRIOR_POWER=${QUERY_PRIOR_POWER:-1.0}
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
OUT=${OUT:-$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256}
CHECKPOINT=${CHECKPOINT:-$OUT/chkpnt0.pth}
ASSIGNMENTS=${ASSIGNMENTS:-$OUT/point_group_assignments.npz}
GROUP_FEATURES=${GROUP_FEATURES:-$OUT/teacher_distilled/group_features_teacher_w0p75.npy}
CODEBOOK_DIR=${CODEBOOK_DIR:-$OUT/teacher_distilled/codebook_teacher_w0p75_k256_usage}
REVERSE_DIR=${REVERSE_DIR:-$OUT/teacher_distilled/reverse_codebook_teacher_w0p75_k256_usage}
LOG_DIR=${LOG_DIR:-$ROOT/logs/reverse_codebook_sweep/$SCENE}

mkdir -p "$LOG_DIR" "$REVERSE_DIR"

for path in "$CHECKPOINT" "$ASSIGNMENTS" "$GROUP_FEATURES" "$CODEBOOK_DIR/codebook.npy" "$CODEBOOK_DIR/group_to_code.npy"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$REVERSE_DIR/reverse_codebook_summary.json" ]]; then
  echo "[$(date +%FT%T)] build reverse-mounted codebook index"
  "$PYTHON_BIN" build_reverse_codebook_index.py \
    --codebook_dir "$CODEBOOK_DIR" \
    --group_features "$GROUP_FEATURES" \
    --assignments "$ASSIGNMENTS" \
    --output_dir "$REVERSE_DIR" \
    > "$LOG_DIR/00_build_reverse_index.log" 2>&1
else
  echo "[$(date +%FT%T)] reuse reverse-mounted codebook index: $REVERSE_DIR"
fi

for aggregation in $AGGREGATIONS; do
  for top_codes in $TOP_CODES; do
    for residual_weight in $RESIDUAL_WEIGHTS; do
      for code_blend in $CODE_BLENDS; do
        top_tag=${top_codes/./p}
        residual_tag=${residual_weight/./p}
        blend_tag=${code_blend/./p}
        eval_dir="$OUT/eval/lerf_ovs_teacher_codebook_k256_reverse_${aggregation}_top${top_tag}_res${residual_tag}_codeblend${blend_tag}"
        echo "[$(date +%FT%T)] eval reverse codebook aggregation=$aggregation top_codes=$top_codes residual=$residual_weight code_blend=$code_blend"
        "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
          -s "$DATASET" -m "$OUT" \
          --checkpoint "$CHECKPOINT" \
          --label_dir "$LABEL_DIR" \
          --group_features "$GROUP_FEATURES" \
          --assignments "$ASSIGNMENTS" \
          --reverse_codebook_dir "$REVERSE_DIR" \
          --reverse_top_codes "$top_codes" \
          --reverse_residual_weight "$residual_weight" \
          --reverse_code_blend "$code_blend" \
          --aggregation "$aggregation" \
          --score_power 1.0 \
          --query_temperature "$QUERY_TEMPERATURE" \
          --query_prior_power "$QUERY_PRIOR_POWER" \
          --thresholds $THRESHOLDS \
          --output "$eval_dir" \
          > "$LOG_DIR/01_eval_${aggregation}_top${top_tag}_res${residual_tag}_codeblend${blend_tag}.log" 2>&1
      done
    done
  done
done

echo "[$(date +%FT%T)] reverse codebook sweep done for $SCENE"
