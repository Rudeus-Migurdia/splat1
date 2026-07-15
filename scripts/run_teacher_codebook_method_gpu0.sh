#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENE=${SCENE:-figurines}
DATASET=${DATASET:-$ROOT/drsplat_data/lerf_ovs/$SCENE}
LABEL_DIR=${LABEL_DIR:-$ROOT/drsplat_data/lerf_ovs/label/$SCENE}
START_CKPT=${START_CKPT:-$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth}
DRSPLAT_CKPT=${DRSPLAT_CKPT:-$ROOT/runs/drsplat/${SCENE}_1_pq_openclip_topk45_weight_128/chkpnt0.pth}
PQ_INDEX=${PQ_INDEX:-$ROOT/ckpts/pq_index.faiss}
OUT=${OUT:-$ROOT/runs/prototypes/mask_group_lift/${SCENE}_teacher_codebook_k256}
LOG_DIR=${LOG_DIR:-$ROOT/logs/teacher_codebook_method_${SCENE}}
GPU_ID=${GPU_ID:-0}

mkdir -p "$LOG_DIR"
cd "$ROOT"

source scripts/drsplat_env.sh
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SITE=${SITE:-$VENV_PATH/lib/python3.9/site-packages}
export ROOT VENV_PATH PYTHON_BIN
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}

run_inner() {
  rm -rf "$OUT"
  mkdir -p "$OUT"

  echo "[$(date +%FT%T)] lift multi-group tokens for $SCENE"
  "$PYTHON_BIN" -u prototype_mask_group_lift.py \
    -s "$DATASET" -m "$OUT" \
    --start_checkpoint "$START_CKPT" \
    --pq_index "$PQ_INDEX" \
    --output_model "$OUT" \
    --label_dir "$LABEL_DIR" \
    --feature_level 1 \
    --topk 10 \
    --mask_keep_gaussians 2048 \
    --group_keep_gaussians 8192 \
    --candidate_seed_gaussians 64 \
    --index_group_gaussians 160 \
    --min_group_observations 1 \
    --assignment_mode soft \
    --keep_point_groups 4 \
    --soft_score_power 1.0 \
    > "$LOG_DIR/01_lift.log" 2>&1

  echo "[$(date +%FT%T)] summarize artifact"
  "$PYTHON_BIN" summarize_multigroup_artifact.py --artifact_dir "$OUT" \
    > "$LOG_DIR/02_summary.log" 2>&1

  echo "[$(date +%FT%T)] distill Dr.Splat teacher into group tokens"
  "$PYTHON_BIN" distill_group_tokens_from_drsplat.py \
    --artifact_dir "$OUT" \
    --drsplat_checkpoint "$DRSPLAT_CKPT" \
    --pq_index "$PQ_INDEX" \
    --score_power 1.0 \
    --teacher_weights 0.75 \
    --output_dir "$OUT/teacher_distilled" \
    > "$LOG_DIR/03_distill.log" 2>&1

  echo "[$(date +%FT%T)] quantize distilled group tokens"
  mkdir -p "$OUT/teacher_distilled/artifact_w0p75"
  ln -sf ../group_features_teacher_w0p75.npy "$OUT/teacher_distilled/artifact_w0p75/group_features.npy"
  ln -sf ../../point_group_assignments.npz "$OUT/teacher_distilled/artifact_w0p75/point_group_assignments.npz"
  "$PYTHON_BIN" quantize_multigroup_codebook.py \
    --artifact_dir "$OUT/teacher_distilled/artifact_w0p75" \
    --num_codes 256 \
    --levels 1 \
    --iterations 120 \
    --seed 31 \
    --usage_weighted \
    --output_dir "$OUT/teacher_distilled/codebook_teacher_w0p75_k256_usage" \
    > "$LOG_DIR/04_quantize.log" 2>&1

  echo "[$(date +%FT%T)] evaluate teacher-distilled codebook"
  "$PYTHON_BIN" -u eval_lerf_ovs_multigroup_miou.py \
    -s "$DATASET" -m "$OUT" \
    --checkpoint "$OUT/chkpnt0.pth" \
    --label_dir "$LABEL_DIR" \
    --group_features "$OUT/teacher_distilled/codebook_teacher_w0p75_k256_usage/group_features_quantized.npy" \
    --assignments "$OUT/point_group_assignments.npz" \
    --aggregation weighted \
    --score_power 1.0 \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$OUT/eval/lerf_ovs_teacher_codebook_k256_weighted" \
    > "$LOG_DIR/05_eval.log" 2>&1

  echo "[$(date +%FT%T)] done"
}

if [[ "${1:-}" == "--inner" ]]; then
  run_inner
  exit 0
fi

"$PYTHON_BIN" -u scripts/gpu_guard.py \
  --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 -- \
  bash -lc "set -euo pipefail; cd '$ROOT'; ROOT='$ROOT' VENV_PATH='$VENV_PATH' PYTHON_BIN='$PYTHON_BIN' GPU_ID='$GPU_ID' bash scripts/run_teacher_codebook_method_gpu0.sh --inner"
