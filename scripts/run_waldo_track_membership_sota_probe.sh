#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/track_membership_sota/waldo_fixed}
LOG_DIR=${LOG_DIR:-$ROOT/logs/track_membership_sota_waldo}
GPU_IDS=(${GPU_IDS:-0 1 2 3})

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

if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four physical GPU indices" >&2
  exit 2
fi

dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth
codebook=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2
cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l1_raw
association_cache=$RUN_ROOT/saga_signatures
identity_dir=$RUN_ROOT/identity_trace_memory
membership_dir=$RUN_ROOT/membership_saga_local
both_dir=$RUN_ROOT/identity_trace_memory_membership_saga
mkdir -p "$RUN_ROOT" "$LOG_DIR"

for required in "$dataset" "$labels" "$geometry" "$codebook/manifest.json" "$cache/manifest.json"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

build_identity() {
  [[ -f "$identity_dir/manifest.json" ]] && return
  "$PYTHON_BIN" -u build_full_contribution_group_membership.py \
    --cache_dir "$cache" --output_dir "$identity_dir" \
    --track_linking trace_memory_mutual --memory_signature_points 512 \
    --similarity_threshold 0.82 --min_soft_overlap 0.05 \
    --min_track_support 32 --min_track_views 3 \
    --top_m 3 --membership_threshold 0.5 --min_foreground 0.0001 \
    --min_view_contribution 0.0001 --view_foreground_ratio 0.5 \
    --view_weighting information_kl --importance_temperature 1.0 \
    --max_view_kl 0.02 --importance_ratio_clip 5 \
    --agreement_power 1 --information_weight 1 \
    > "$LOG_DIR/build_identity.log" 2>&1
}

build_membership() {
  [[ -f "$membership_dir/manifest.json" ]] && return
  "$PYTHON_BIN" -u build_full_contribution_group_membership.py \
    --cache_dir "$cache" --output_dir "$membership_dir" \
    --track_linking mutual_soft_overlap --track_window 3 \
    --similarity_threshold 0.82 --min_soft_overlap 0.05 \
    --min_track_support 32 --min_track_views 3 \
    --membership_mode saga_union --semantic_codebook_dir "$codebook" \
    --association_cache_dir "$association_cache" \
    --association_fraction 0.2 --association_max_candidates 2048 \
    --top_m 3 --membership_threshold 0.5 --min_foreground 0.0001 \
    --min_view_contribution 0.0001 --view_foreground_ratio 0.5 \
    --view_weighting information_kl --importance_temperature 1.0 \
    --max_view_kl 0.02 --importance_ratio_clip 5 \
    --agreement_power 1 --information_weight 1 \
    > "$LOG_DIR/build_membership.log" 2>&1
}

build_both() {
  [[ -f "$both_dir/manifest.json" ]] && return
  "$PYTHON_BIN" -u build_full_contribution_group_membership.py \
    --cache_dir "$cache" --output_dir "$both_dir" \
    --track_linking trace_memory_mutual --memory_signature_points 512 \
    --similarity_threshold 0.82 --min_soft_overlap 0.05 \
    --min_track_support 32 --min_track_views 3 \
    --membership_mode saga_union --semantic_codebook_dir "$codebook" \
    --association_cache_dir "$association_cache" \
    --association_fraction 0.2 --association_max_candidates 2048 \
    --top_m 3 --membership_threshold 0.5 --min_foreground 0.0001 \
    --min_view_contribution 0.0001 --view_foreground_ratio 0.5 \
    --view_weighting information_kl --importance_temperature 1.0 \
    --max_view_kl 0.02 --importance_ratio_clip 5 \
    --agreement_power 1 --information_weight 1 \
    > "$LOG_DIR/build_both.log" 2>&1
}

evaluate_variant() {
  local name=$1
  local hierarchy=$2
  local output=$RUN_ROOT/eval_$name
  [[ -f "$output/metrics.json" ]] && return
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" --codebook_dir "$codebook" \
    --group_hierarchy_dir "$hierarchy" --group_topk 1 \
    --group_readout hypothesis --group_route_fraction 0.1 \
    --group_route_priority reliability_gain \
    --label_dir "$labels" --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --output "$output" \
    > "$LOG_DIR/eval_$name.log" 2>&1
}

case "${1:-all}" in
  build-identity) build_identity ;;
  build-membership) build_membership ;;
  build-both) build_both ;;
  eval) evaluate_variant "$2" "$3" ;;
  all)
    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "${GPU_IDS[0]}" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- bash "$0" build-membership \
      > "$LOG_DIR/worker_build_membership.log" 2>&1 &
    membership_worker=$!
    bash "$0" build-identity > "$LOG_DIR/worker_build_identity.log" 2>&1 &
    identity_worker=$!
    status=0
    wait "$membership_worker" || status=$?
    wait "$identity_worker" || status=$?
    [[ "$status" -eq 0 ]] || exit "$status"

    "$PYTHON_BIN" scripts/gpu_guard.py \
      --gpu "${GPU_IDS[1]}" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
      --wait-timeout 0 -- bash "$0" build-both \
      > "$LOG_DIR/worker_build_both.log" 2>&1

    variants=(identity membership both)
    hierarchies=("$identity_dir" "$membership_dir" "$both_dir")
    workers=()
    for index in 0 1 2; do
      "$PYTHON_BIN" scripts/gpu_guard.py \
        --gpu "${GPU_IDS[$index]}" --hold-mb 512 --max-used-mb 256 --max-utilization 5 \
        --wait-timeout 0 -- bash "$0" eval "${variants[$index]}" "${hierarchies[$index]}" \
        > "$LOG_DIR/worker_eval_${variants[$index]}.log" 2>&1 &
      workers+=("$!")
    done
    status=0
    for worker in "${workers[@]}"; do
      wait "$worker" || status=$?
    done
    [[ "$status" -eq 0 ]] || exit "$status"
    ;;
  *) echo "Unknown mode: $1" >&2; exit 2 ;;
esac
