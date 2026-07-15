#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

HYPOTHESIS_DIR=${HYPOTHESIS_DIR:-runs/query_routing/a8/waldo_l1_continuous}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/query_routing/a8/waldo_eval}
LOG_ROOT=${LOG_ROOT:-logs/query_routing/a8}
SPECS=${SPECS:-"switch:0:switch reliability_blend:0:blend reliability_blend:1:blend_margin"}
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for spec in $SPECS; do
  readout=${spec%%:*}
  remainder=${spec#*:}
  margin=${remainder%%:*}
  tag=${remainder##*:}
  margin_args=()
  [[ "$margin" == "1" ]] && margin_args+=(--hypothesis_query_margin)
  echo "[$(date +%FT%T)] $tag readout=$readout margin=$margin start"
  .venv/bin/python -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s drsplat_data/lerf_ovs/waldo_kitchen \
    -m runs/3dgs/waldo_kitchen \
    --geometry_checkpoint runs/3dgs/waldo_kitchen/chkpnt30000.pth \
    --codebook_dir runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2 \
    --hypothesis_dir "$HYPOTHESIS_DIR" \
    --hypothesis_readout "$readout" \
    "${margin_args[@]}" \
    --label_dir drsplat_data/lerf_ovs/label/waldo_kitchen \
    --evaluation_protocol drsplat_3d_selection \
    --output "$OUTPUT_ROOT/$tag" \
    > "$LOG_ROOT/waldo_${tag}.log" 2>&1
  echo "[$(date +%FT%T)] $tag done"
done
