#!/usr/bin/env bash
set -euo pipefail

# Complete A27 waldo on a 96 GB PRO 6000 after the 24 GB worker OOM.
# Everything written by this runner stays below /home/anlanfan/a27_*.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
INPUT_ROOT=${INPUT_ROOT:?INPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.conda/envs/drsplat-py39/bin/python3.9}
SEED=${SEED:-20260717}
SCENE=waldo_kitchen
LOG_DIR="$RUN_ROOT/logs"
MEMORY="$RUN_ROOT/$SCENE/hierarchical_memory"
OUTPUT="$RUN_ROOT/$SCENE/eval"

export PYTHONPATH="$SOURCE_DIR:$ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED="$INPUT_ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
mkdir -p "$LOG_DIR" "$RUN_ROOT/$SCENE"

for required in \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  "$ROOT/drsplat_data/lerf_ovs/$SCENE" \
  "$INPUT_ROOT/3dgs/$SCENE/chkpnt30000.pth" \
  "$INPUT_ROOT/a14_cont/$SCENE/old_split2/consensus.pt" \
  "$INPUT_ROOT/a14_disc/$SCENE/base_ids/manifest.json" \
  "$INPUT_ROOT/a14_disc/$SCENE/pruned_candidate_ids/manifest.json" \
  "$OPENCLIP_PRETRAINED"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done
for level in 0 1 2 3; do
  [[ -f "$RUN_ROOT/$SCENE/sam_l${level}_split2/consensus.pt" ]] || {
    echo "Missing isolated SAM consensus for level $level" >&2
    exit 2
  }
done

if [[ ! -f "$MEMORY/manifest.json" ]]; then
  "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
    --geometry_checkpoint "$INPUT_ROOT/3dgs/$SCENE/chkpnt30000.pth" \
    --old_consensus "$INPUT_ROOT/a14_cont/$SCENE/old_split2/consensus.pt" \
    --sam_l0_consensus "$RUN_ROOT/$SCENE/sam_l0_split2/consensus.pt" \
    --sam_l1_consensus "$RUN_ROOT/$SCENE/sam_l1_split2/consensus.pt" \
    --sam_l2_consensus "$RUN_ROOT/$SCENE/sam_l2_split2/consensus.pt" \
    --sam_l3_consensus "$RUN_ROOT/$SCENE/sam_l3_split2/consensus.pt" \
    --output_dir "$MEMORY" --device cuda --seed "$SEED" --neighbors 8 \
    --semantic_thresholds 0.76 0.82 0.87 0.91 \
    --maximum_group_sizes 2048 512 128 32 \
    --minimum_group_sizes 16 8 4 2 \
    --codes_per_level 2048 4096 8192 16384 \
    --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 2000000 \
    --stability_floor 0.5 --minimum_reliability 0.25 \
    --source_agreement_floor 0.80 --source_margin 0.0 \
    --faiss_gpu > "$LOG_DIR/${SCENE}_memory_build.log" 2>&1
fi
"$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  --artifact_dir "$MEMORY" --required base sam_l0 sam_l1 sam_l2 sam_l3 \
  > "$LOG_DIR/${SCENE}_memory_contract.log" 2>&1

if [[ ! -f "$OUTPUT/metrics.json" ]]; then
  mkdir -p "$OUTPUT"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$INPUT_ROOT/3dgs/$SCENE" \
    --geometry_checkpoint "$INPUT_ROOT/3dgs/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$INPUT_ROOT/a14_disc/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$INPUT_ROOT/a14_disc/$SCENE/base_ids" \
    --codebook_query_route query_positive --group_hierarchy_dir "$MEMORY" \
    --group_topk 4 --group_readout calibrated_hierarchical_memory \
    --group_query_temperature 0.10 \
    --group_level_margin_threshold 0.25 \
    --group_level_margin_temperature 0.10 \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --occupancy_threshold 0.7 \
    --output "$OUTPUT" > "$LOG_DIR/${SCENE}_eval.log" 2>&1
fi

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
