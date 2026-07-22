#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?}
SOURCE_DIR=${SOURCE_DIR:?}
RUN_ROOT=${RUN_ROOT:?}
LOG_DIR=${LOG_DIR:?}
SCENE=${SCENE:?}
GPU=${GPU:?}
MEMORY=${MEMORY:?}
SPATIAL=${SPATIAL:?}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_DISC=${A14_DISC:-$ROOT/runs/a14_e8_joint32k_20260716}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=20260719 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

common=(
  -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE"
  --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
  --codebook_dir "$A14_DISC/$SCENE/pruned_candidate_ids"
  --query_route_base_codebook_dir "$A14_DISC/$SCENE/base_ids"
  --codebook_query_route query_positive
  --group_hierarchy_dir "$MEMORY" --group_topk 4
  --group_query_temperature 0.05
  --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
  --evaluation_protocol drsplat_3d_selection
  --selection_thresholds 0.55 --occupancy_threshold 0.7
)

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "${common[@]}" --group_readout equal_query_max \
  --output "$RUN_ROOT/$SCENE/eval_control_equal_query_max" \
  > "$LOG_DIR/${SCENE}_control_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "${common[@]}" --spatial_group_posterior_dir "$SPATIAL" \
  --group_readout equal_query_global_anchor_entmax15 \
  --global_group_temperature 0.05 --global_group_semantic_weight 0.75 \
  --global_group_ring_contrast_strength 0.50 \
  --global_group_maximum_penalty 0.08 --global_group_entropy_relaxation 0.50 \
  --global_group_anchor_quantile 0.20 \
  --global_group_anchor_temperature 0.02 \
  --global_group_semantic_preservation_quantile 0.75 \
  --spatial_posterior_core_membership 0.30 \
  --output "$RUN_ROOT/$SCENE/eval_mass_conserving_anchor" \
  > "$LOG_DIR/${SCENE}_a55_eval.log" 2>&1
