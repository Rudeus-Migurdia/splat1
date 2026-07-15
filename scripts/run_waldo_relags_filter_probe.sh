#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/relags_conditioned/waldo_mwp5e4_rofa2}
LOG_DIR=${LOG_DIR:-$ROOT/logs/relags_conditioned_waldo}
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
a6_continuous=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt
a6_discrete=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2
l1_cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l1_raw
mwp_dir=$RUN_ROOT/max_contribution
rofa_dir=$RUN_ROOT/l1_mutual_rofa_tau2
keep_mask=$mwp_dir/keep_gt_5e-04.npy
mkdir -p "$RUN_ROOT" "$LOG_DIR"

for required in "$dataset" "$labels" "$geometry" "$a6_continuous" \
  "$a6_discrete/manifest.json" "$l1_cache/manifest.json"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

scan_mwp() {
  if [[ -f "$mwp_dir/manifest.json" ]]; then
    echo "[$(date +%FT%T)] reuse MWP scan"
    return
  fi
  "$PYTHON_BIN" -u compute_gaussian_max_contribution.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" \
    --output_dir "$mwp_dir" --thresholds 0.0005 \
    > "$LOG_DIR/mwp_scan.log" 2>&1
}

build_rofa() {
  if [[ -f "$rofa_dir/manifest.json" ]]; then
    echo "[$(date +%FT%T)] reuse ROFA hierarchy"
    return
  fi
  "$PYTHON_BIN" -u build_full_contribution_group_membership.py \
    --cache_dir "$l1_cache" --output_dir "$rofa_dir" \
    --similarity_threshold 0.82 --min_track_support 32 \
    --track_linking mutual_soft_overlap --track_window 3 \
    --min_soft_overlap 0.05 --min_track_views 3 \
    --top_m 3 --membership_threshold 0.5 --min_foreground 0.0001 \
    --min_view_contribution 0.0001 --view_foreground_ratio 0.5 \
    --view_weighting information_kl --importance_temperature 1.0 \
    --max_view_kl 0.02 --importance_ratio_clip 5 \
    --agreement_power 1 --information_weight 1 --rofa_tau 2 \
    > "$LOG_DIR/rofa_build.log" 2>&1
}

eval_mwp() {
  local source=$1
  local output=$2
  shift 2
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" "$@" \
    --semantic_keep_mask "$keep_mask" \
    --label_dir "$labels" --evaluation_protocol drsplat_3d_selection \
    --output "$output" \
    > "$LOG_DIR/eval_${source}.log" 2>&1
}

eval_rofa() {
  local output=$RUN_ROOT/eval_l1_rofa_reliability_top10
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" --codebook_dir "$a6_discrete" \
    --group_hierarchy_dir "$rofa_dir" --group_topk 1 \
    --group_readout hypothesis --group_route_fraction 0.1 \
    --group_route_priority reliability_gain \
    --label_dir "$labels" --evaluation_protocol drsplat_3d_selection \
    --output "$output" \
    > "$LOG_DIR/eval_rofa.log" 2>&1
}

if [[ "${1:-}" == "--scan-mwp" ]]; then
  scan_mwp
  exit 0
fi
if [[ "${1:-}" == "--build-rofa" ]]; then
  build_rofa
  exit 0
fi
if [[ "${1:-}" == "--eval-mwp" ]]; then
  eval_mwp "$2" "$3" "${@:4}"
  exit 0
fi
if [[ "${1:-}" == "--eval-rofa" ]]; then
  eval_rofa
  exit 0
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "${GPU_IDS[0]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_relags_filter_probe.sh" --scan-mwp \
  > "$LOG_DIR/worker_mwp.log" 2>&1 &
mwp_worker=$!

bash "$ROOT/scripts/run_waldo_relags_filter_probe.sh" --build-rofa \
  > "$LOG_DIR/worker_rofa.log" 2>&1 &
rofa_worker=$!

status=0
wait "$mwp_worker" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "MWP scan failed with status=$status" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "${GPU_IDS[1]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_relags_filter_probe.sh" \
    --eval-mwp continuous "$RUN_ROOT/eval_mwp_continuous" \
    --consensus_path "$a6_continuous" \
  > "$LOG_DIR/worker_eval_mwp_continuous.log" 2>&1 &
eval_continuous=$!

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "${GPU_IDS[2]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_relags_filter_probe.sh" \
    --eval-mwp discrete "$RUN_ROOT/eval_mwp_discrete" \
    --codebook_dir "$a6_discrete" \
  > "$LOG_DIR/worker_eval_mwp_discrete.log" 2>&1 &
eval_discrete=$!

wait "$rofa_worker" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "ROFA build failed with status=$status" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/gpu_guard.py \
  --gpu "${GPU_IDS[3]}" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
  --wait-timeout 0 -- \
  bash "$ROOT/scripts/run_waldo_relags_filter_probe.sh" --eval-rofa \
  > "$LOG_DIR/worker_eval_rofa.log" 2>&1 &
eval_rofa_worker=$!

for worker in "$eval_continuous" "$eval_discrete" "$eval_rofa_worker"; do
  wait "$worker" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "A ReLaGS-conditioned evaluator failed with status=$status" >&2
  exit "$status"
fi

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "ReLaGS-conditioned Waldo probe complete: $RUN_ROOT"
