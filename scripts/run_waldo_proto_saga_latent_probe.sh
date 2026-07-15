#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/proto_saga_latent/waldo_l1_d16_i10000}
GROUP_DIR=${GROUP_DIR:-$ROOT/runs/track_membership_sota/waldo_learned_classifier_i10000}
EVAL_DIR=${EVAL_DIR:-$ROOT/runs/track_membership_sota/waldo_fixed/eval_learned_classifier_i10000}
LOG_DIR=${LOG_DIR:-$ROOT/logs/proto_saga_latent_waldo}
GPU_ID=${GPU_ID:-0}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}

dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth
codebook=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2
cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l1_raw
association_cache=$GROUP_DIR/association_signatures
mkdir -p "$LOG_DIR"

train_latent() {
  [[ -f "$RUN_ROOT/manifest.json" ]] && return
  "$PYTHON_BIN" -u train_view_specific_gaussian_classifier.py \
    --cache_dir "$cache" --codebook_dir "$codebook" \
    --output_dir "$RUN_ROOT" --latent_dim 16 --temperature 10 \
    --iterations 10000 --pixels_per_step 2048 --steps_per_loaded_view 4 \
    --feature_lr 0.0025 --classifier_lr 0.0005 --seed 0 \
    > "$LOG_DIR/train.log" 2>&1
}

build_groups() {
  [[ -f "$GROUP_DIR/manifest.json" ]] && return
  "$PYTHON_BIN" -u build_full_contribution_group_membership.py \
    --cache_dir "$cache" --output_dir "$GROUP_DIR" \
    --track_linking mutual_soft_overlap --track_window 3 \
    --similarity_threshold 0.82 --min_soft_overlap 0.05 \
    --min_track_support 32 --min_track_views 3 \
    --membership_mode saga_union --semantic_classifier_dir "$RUN_ROOT" \
    --association_cache_dir "$association_cache" \
    --association_fraction 0.2 --association_max_candidates 2048 \
    --top_m 3 --membership_threshold 0.5 --min_foreground 0.0001 \
    --min_view_contribution 0.0001 --view_foreground_ratio 0.5 \
    --view_weighting information_kl --importance_temperature 1.0 \
    --max_view_kl 0.02 --importance_ratio_clip 5 \
    --agreement_power 1 --information_weight 1 \
    > "$LOG_DIR/build.log" 2>&1
}

evaluate_groups() {
  [[ -f "$EVAL_DIR/metrics.json" ]] && return
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" --codebook_dir "$codebook" \
    --group_hierarchy_dir "$GROUP_DIR" --group_topk 1 \
    --group_readout hypothesis --group_route_fraction 0.1 \
    --group_route_priority reliability_gain \
    --label_dir "$labels" --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --output "$EVAL_DIR" \
    > "$LOG_DIR/eval.log" 2>&1
}

case "${1:-all}" in
  train) train_latent ;;
  build) build_groups ;;
  eval) evaluate_groups ;;
  all)
    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- bash "$0" train
    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- bash "$0" build
    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "$GPU_ID" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- bash "$0" eval
    ;;
  *) echo "Unknown mode: $1" >&2; exit 2 ;;
esac
