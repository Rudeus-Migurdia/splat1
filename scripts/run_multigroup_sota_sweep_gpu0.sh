#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/anlanfan/Dr-Splat
DATASET=$ROOT/drsplat_data/lerf_ovs/figurines
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/figurines
OUT=$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens
LOG_DIR=$ROOT/logs/multigroup_sota_sweep

mkdir -p "$LOG_DIR"
cd "$ROOT"

source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PYTHONPATH="$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

run_eval() {
  local name=$1
  shift
  echo "[$(date '+%F %T')] $name"
  .venv/bin/python -u eval_lerf_ovs_multigroup_miou.py \
    -s "$DATASET" -m "$OUT" \
    --checkpoint "$OUT/chkpnt0.pth" \
    --label_dir "$LABEL_DIR" \
    --group_features "$OUT/group_features.npy" \
    --assignments "$OUT/point_group_assignments.npz" \
    --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \
    --output "$OUT/eval/$name" \
    "$@" > "$LOG_DIR/$name.log" 2>&1
}

if [[ "${1:-}" == "--inner" ]]; then
  run_eval lerf_ovs_multigroup_weighted_top2 --aggregation weighted --score_power 1.0 --eval_topk 2
  run_eval lerf_ovs_multigroup_weighted_top3 --aggregation weighted --score_power 1.0 --eval_topk 3
  run_eval lerf_ovs_multigroup_weighted_frame_p01_99 --aggregation weighted --score_power 1.0 --score_calibration frame_percentile --calibration_low 1 --calibration_high 99
  run_eval lerf_ovs_multigroup_weighted_frame_p05_95 --aggregation weighted --score_power 1.0 --score_calibration frame_percentile --calibration_low 5 --calibration_high 95
  run_eval lerf_ovs_multigroup_weighted_cat_p01_99 --aggregation weighted --score_power 1.0 --score_calibration category_percentile --calibration_low 1 --calibration_high 99
  run_eval lerf_ovs_multigroup_weighted_cat_p05_95 --aggregation weighted --score_power 1.0 --score_calibration category_percentile --calibration_low 5 --calibration_high 95
  run_eval lerf_ovs_multigroup_weighted_p05_cat_p01_99 --aggregation weighted --score_power 0.5 --score_calibration category_percentile --calibration_low 1 --calibration_high 99
  run_eval lerf_ovs_multigroup_blend_a075 --aggregation weighted_maxblend --score_power 1.0 --blend_alpha 0.75
  run_eval lerf_ovs_multigroup_blend_a050 --aggregation weighted_maxblend --score_power 1.0 --blend_alpha 0.5
  run_eval lerf_ovs_multigroup_blend_a075_cat_p01_99 --aggregation weighted_maxblend --score_power 1.0 --blend_alpha 0.75 --score_calibration category_percentile --calibration_low 1 --calibration_high 99
  run_eval lerf_ovs_multigroup_blend_a050_cat_p01_99 --aggregation weighted_maxblend --score_power 1.0 --blend_alpha 0.5 --score_calibration category_percentile --calibration_low 1 --calibration_high 99
  run_eval lerf_ovs_multigroup_noisy_or --aggregation noisy_or --score_power 1.0
  exit 0
fi

.venv/bin/python -u scripts/gpu_guard.py \
  --gpu 0 --hold-mb 512 --max-used-mb 256 --max-utilization 5 -- \
  bash -lc '
    set -euo pipefail
    source scripts/run_multigroup_sota_sweep_gpu0.sh --inner
  '

exit 0
