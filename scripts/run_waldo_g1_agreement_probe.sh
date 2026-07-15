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

SPECS=${SPECS:-"0.25:0.50 0.25:0.70"}
GROUP_DIR=${GROUP_DIR:-runs/full_contribution_group/waldo_g1_top45raw_p05_top3}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-g1}
LOG_DIR=${LOG_DIR:-logs/full_contribution_group}
mkdir -p "$LOG_DIR"
for spec in $SPECS; do
  alpha=${spec%%:*}
  floor=${spec##*:}
  alpha_tag=${alpha/./p}
  floor_tag=${floor/./p}
  echo "[$(date +%FT%T)] alpha=$alpha agreement_floor=$floor start"
  .venv/bin/python -u eval_lerf_ovs_gaussian_codebook_miou.py \
    -s drsplat_data/lerf_ovs/waldo_kitchen \
    -m runs/3dgs/waldo_kitchen \
    --geometry_checkpoint runs/3dgs/waldo_kitchen/chkpnt30000.pth \
    --codebook_dir runs/multiscale_split_consistency/fused_w1p5_t005_codebook_k4096x2 \
    --group_hierarchy_dir "$GROUP_DIR" \
    --group_topk 3 \
    --group_aggregation weighted \
    --group_score_power 1.0 \
    --group_membership_confidence \
    --point_gate_floor 0.0 \
    --point_gate_power 1.0 \
    --group_feature_agreement_floor "$floor" \
    --group_feature_agreement_power 1.0 \
    --rgr_mode positive \
    --rgr_alpha "$alpha" \
    --label_dir drsplat_data/lerf_ovs/label/waldo_kitchen \
    --evaluation_protocol drsplat_3d_selection \
    --output "$GROUP_DIR/eval_a${alpha_tag}_agree_${floor_tag}" \
    > "$LOG_DIR/waldo_${EXPERIMENT_TAG}_eval_a${alpha_tag}_agree_${floor_tag}.log" 2>&1
  echo "[$(date +%FT%T)] alpha=$alpha agreement_floor=$floor done"
done
