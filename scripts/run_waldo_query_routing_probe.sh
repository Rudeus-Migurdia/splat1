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

OUTPUT_ROOT=${OUTPUT_ROOT:-runs/query_routing/waldo_g1}
LOG_ROOT=${LOG_ROOT:-logs/query_routing}
GROUP_DIR=${GROUP_DIR:-runs/full_contribution_group/waldo_g1_top45raw_p05_top3}
GROUP_TOPK=${GROUP_TOPK:-3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-g1}
SPECS=${SPECS:-"1.0:query_gain:max_all 0.1:query_gain:query_top10 0.1:membership_gain:membership_top10"}
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for spec in $SPECS; do
  fraction=${spec%%:*}
  remainder=${spec#*:}
  priority=${remainder%%:*}
  tag=${remainder##*:}
  echo "[$(date +%FT%T)] $tag fraction=$fraction priority=$priority start"
  .venv/bin/python -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s drsplat_data/lerf_ovs/waldo_kitchen \
    -m runs/3dgs/waldo_kitchen \
    --geometry_checkpoint runs/3dgs/waldo_kitchen/chkpnt30000.pth \
    --codebook_dir runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2 \
    --group_hierarchy_dir "$GROUP_DIR" \
    --group_topk "$GROUP_TOPK" \
    --group_readout hypothesis \
    --group_route_fraction "$fraction" \
    --group_route_priority "$priority" \
    --label_dir drsplat_data/lerf_ovs/label/waldo_kitchen \
    --evaluation_protocol drsplat_3d_selection \
    --output "$OUTPUT_ROOT/$tag" \
    > "$LOG_ROOT/waldo_${EXPERIMENT_TAG}_${tag}.log" 2>&1
  echo "[$(date +%FT%T)] $tag done"
done
