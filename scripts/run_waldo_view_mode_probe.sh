#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv-236/bin/python}
GPU_ID=${GPU_ID:-0}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/view_semantic_modes/waldo_l2_split_reproducible}
LOG_DIR=${LOG_DIR:-$ROOT/logs/view_semantic_modes_waldo}

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

mkdir -p "$RUN_ROOT" "$LOG_DIR"
base=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt
cache=$ROOT/runs/query_routing/waldo_multiscale/cache_l2_raw
codebook=$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2
dataset=$ROOT/drsplat_data/lerf_ovs/waldo_kitchen
labels=$ROOT/drsplat_data/lerf_ovs/label/waldo_kitchen
geometry=$ROOT/runs/3dgs/waldo_kitchen/chkpnt30000.pth

for required in "$base" "$cache/manifest.json" "$codebook/manifest.json" "$geometry" "$labels"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

if [[ ! -f "$RUN_ROOT/hypothesis/manifest.json" ]]; then
  "$PYTHON_BIN" -u build_split_reproducible_semantic_modes.py \
    --base_consensus "$base" \
    --view_cache_dir "$cache" \
    --output_dir "$RUN_ROOT/hypothesis" \
    --device cuda \
    --deviation_cosine_max 0.75 \
    --min_views_per_split 3 \
    --support_saturation 6 \
    --min_compactness 0.88 \
    --min_cross_split_cosine 0.90 \
    --max_base_cosine 0.90 \
    > "$LOG_DIR/build.log" 2>&1
fi

for spec in blend:blend blend_margin:blend_margin switch:switch; do
  tag=${spec%%:*}
  readout=${spec##*:}
  output=$RUN_ROOT/eval_$tag
  [[ -f "$output/metrics.json" ]] && continue
  args=(--hypothesis_readout reliability_blend)
  [[ "$readout" == "switch" ]] && args=(--hypothesis_readout switch)
  [[ "$readout" == "blend_margin" ]] && args+=(--hypothesis_query_margin)
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$codebook" \
    --hypothesis_dir "$RUN_ROOT/hypothesis" \
    "${args[@]}" \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --output "$output" \
    > "$LOG_DIR/eval_${tag}.log" 2>&1
done

directional=$RUN_ROOT/hypothesis_directional
if [[ ! -f "$directional/manifest.json" ]]; then
  "$PYTHON_BIN" -u reweight_sparse_semantic_hypothesis.py \
    --hypothesis_dir "$RUN_ROOT/hypothesis" \
    --base_consensus "$base" \
    --output_dir "$directional" \
    > "$LOG_DIR/reweight_directional.log" 2>&1
fi

output=$RUN_ROOT/eval_directional_blend_margin
if [[ ! -f "$output/metrics.json" ]]; then
  "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s "$dataset" -m "$ROOT/runs/3dgs/waldo_kitchen" \
    --geometry_checkpoint "$geometry" \
    --codebook_dir "$codebook" \
    --hypothesis_dir "$directional" \
    --hypothesis_readout reliability_blend \
    --hypothesis_query_margin \
    --label_dir "$labels" \
    --evaluation_protocol drsplat_3d_selection \
    --output "$output" \
    > "$LOG_DIR/eval_directional_blend_margin.log" 2>&1
fi

date +%FT%T > "$RUN_ROOT/COMPLETE"
echo "Waldo split-reproducible view-mode probe complete: $RUN_ROOT"
