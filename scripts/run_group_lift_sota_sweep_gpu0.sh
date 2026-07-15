#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/anlanfan/Dr-Splat
DATASET=$ROOT/drsplat_data/lerf_ovs/figurines
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/figurines
START_CKPT=$ROOT/runs/3dgs/figurines/chkpnt30000.pth
PQ_INDEX=$ROOT/ckpts/pq_index.faiss
BASE_OUT=$ROOT/runs/prototypes/mask_group_lift
LOG_DIR=$ROOT/logs/group_lift_sota_sweep

mkdir -p "$LOG_DIR"
cd "$ROOT"

source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

run_eval_pair() {
  local out=$1
  local name=$2

  .venv/bin/python -u eval_lerf_ovs_multigroup_miou.py \
    -s "$DATASET" -m "$out" \
    --checkpoint "$out/chkpnt0.pth" \
    --label_dir "$LABEL_DIR" \
    --group_features "$out/group_features.npy" \
    --assignments "$out/point_group_assignments.npz" \
    --aggregation weighted --score_power 1.0 \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$out/eval/lerf_ovs_multigroup_weighted" \
    > "$LOG_DIR/${name}_weighted.log" 2>&1

  .venv/bin/python -u eval_lerf_ovs_multigroup_miou.py \
    -s "$DATASET" -m "$out" \
    --checkpoint "$out/chkpnt0.pth" \
    --label_dir "$LABEL_DIR" \
    --group_features "$out/group_features.npy" \
    --assignments "$out/point_group_assignments.npz" \
    --aggregation weighted_maxblend --score_power 1.0 --blend_alpha 0.75 \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$out/eval/lerf_ovs_multigroup_blend_a075" \
    > "$LOG_DIR/${name}_blend_a075.log" 2>&1
}

run_variant() {
  local name=$1
  shift
  local out=$BASE_OUT/$name

  echo "[$(date +%FT%T)] start $name"
  rm -rf "$out"
  .venv/bin/python -u prototype_mask_group_lift.py \
    -s "$DATASET" -m "$out" \
    --start_checkpoint "$START_CKPT" \
    --pq_index "$PQ_INDEX" \
    --output_model "$out" \
    --label_dir "$LABEL_DIR" \
    --assignment_mode soft \
    --keep_point_groups 4 \
    --soft_score_power 1.0 \
    --min_group_observations 1 \
    "$@" > "$LOG_DIR/${name}_lift.log" 2>&1

  .venv/bin/python summarize_multigroup_artifact.py --artifact_dir "$out" \
    > "$LOG_DIR/${name}_summary.log" 2>&1
  run_eval_pair "$out" "$name"
  echo "[$(date +%FT%T)] done $name"
}

if [[ "${1:-}" == "--inner" ]]; then
  run_variant group_soft_topk10_merge045 \
    --topk 10 --mask_keep_gaussians 2048 --group_keep_gaussians 8192 \
    --candidate_seed_gaussians 64 --index_group_gaussians 160 \
    --merge_score 0.45 --min_group_iou 0.015 --min_group_cosine 0.70

  run_variant group_soft_topk10_merge055 \
    --topk 10 --mask_keep_gaussians 2048 --group_keep_gaussians 8192 \
    --candidate_seed_gaussians 64 --index_group_gaussians 160 \
    --merge_score 0.55 --min_group_iou 0.015 --min_group_cosine 0.70

  run_variant group_soft_topk10_cos075_merge045 \
    --topk 10 --mask_keep_gaussians 2048 --group_keep_gaussians 8192 \
    --candidate_seed_gaussians 64 --index_group_gaussians 160 \
    --merge_score 0.45 --min_group_iou 0.015 --min_group_cosine 0.75

  run_variant group_soft_topk20_dense_merge045 \
    --topk 20 --mask_keep_gaussians 3072 --group_keep_gaussians 12288 \
    --candidate_seed_gaussians 128 --index_group_gaussians 256 \
    --merge_score 0.45 --min_group_iou 0.015 --min_group_cosine 0.70

  exit 0
fi

.venv/bin/python -u scripts/gpu_guard.py \
  --gpu 0 --hold-mb 512 --max-used-mb 256 --max-utilization 5 -- \
  bash -lc 'set -euo pipefail; source scripts/run_group_lift_sota_sweep_gpu0.sh --inner'
