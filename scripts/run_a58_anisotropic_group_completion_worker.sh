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
SHAPE=${SHAPE:?}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_DISC=${A14_DISC:-$ROOT/runs/a14_e8_joint32k_20260716}
SEED=${SEED:-20260719}

cd "$ROOT"
export DRSPLAT_CACHE_DIR="$RUN_ROOT/.cache/$SCENE"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

OUTPUT=$RUN_ROOT/$SCENE/eval_anisotropic_group_completion
MODEL_DIR=$RUN_ROOT/$SCENE/model_shadow
mkdir -p "$OUTPUT" "$MODEL_DIR"

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$MODEL_DIR" \
  --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
  --codebook_dir "$A14_DISC/$SCENE/pruned_candidate_ids" \
  --query_route_base_codebook_dir "$A14_DISC/$SCENE/base_ids" \
  --codebook_query_route query_positive \
  --group_hierarchy_dir "$MEMORY" --group_topk 4 \
  --spatial_group_posterior_dir "$SPATIAL" \
  --group_anisotropic_geometry_dir "$SHAPE" \
  --group_readout equal_query_anisotropic_group_completion \
  --group_query_temperature 0.05 \
  --global_group_temperature 0.05 --global_group_semantic_weight 0.75 \
  --global_group_ring_contrast_strength 0.50 \
  --global_group_maximum_penalty 0.08 --global_group_entropy_relaxation 0.50 \
  --global_group_anchor_quantile 0.20 \
  --global_group_anchor_temperature 0.02 \
  --global_group_semantic_preservation_quantile 0.75 \
  --spatial_posterior_core_membership 0.30 \
  --group_completion_seed_quantile 0.75 \
  --group_completion_seed_support 0.95 \
  --group_completion_seed_score_floor 0.55 \
  --group_completion_target_quantile 0.20 \
  --group_completion_boundary_membership 0.05 \
  --group_completion_semantic_delta 0.15 \
  --group_completion_agreement_temperature 0.02 \
  --group_completion_strength 1.00 \
  --group_completion_max_expansion_ratio 2.0 \
  --group_completion_minimum_seed_points 8 \
  --group_completion_minimum_contact 0.20 \
  --group_completion_maximum_hops 32 \
  --group_completion_anisotropic_axis_floor 0.05 \
  --group_completion_anisotropic_budget_floor 0.25 \
  --group_completion_anisotropic_semantic_floor 0.50 \
  --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
  --evaluation_protocol drsplat_3d_selection \
  --selection_thresholds 0.55 --occupancy_threshold 0.7 \
  --output "$OUTPUT" > "$LOG_DIR/${SCENE}_eval.log" 2>&1

echo "A58_SCENE_COMPLETE $SCENE"
