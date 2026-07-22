#!/usr/bin/env bash
set -euo pipefail

# A40: SFS-style post-aggregation mask filtering, followed by fresh L0--L3
# consensus aggregation, four independent codebook trainings, and peer-token
# query-score fusion.  Ramen is the pre-registered first-stage probe.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
SCENE=${SCENE:-ramen}
GPU=${GPU:-4}
SEED=${SEED:-20260719}
PROVISIONAL_MEMORY=${PROVISIONAL_MEMORY:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/equal_four_token_memory}
CONTROL_METRICS=${CONTROL_METRICS:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/eval_equal_query_max/metrics.json}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
LABEL_ROOT=${LABEL_ROOT:-$ROOT/drsplat_data/lerf_ovs/label}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLDS=${SELECTION_THRESHOLDS:-"0.50 0.55"}
PRIMARY_THRESHOLD=${PRIMARY_THRESHOLD:-0.55}
FILTER_DIR=$RUN_ROOT/$SCENE/post_aggregation_filter
MEMORY_DIR=$RUN_ROOT/$SCENE/filtered_four_token_memory
EVAL_DIR=$RUN_ROOT/$SCENE/eval_equal_query_max
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

run_worker() {
  if [[ ! -f "$FILTER_DIR/FILTER_COMPLETE" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
      -s "$DATA_ROOT/$SCENE" -m "$RUN_ROOT/$SCENE/filter_model_context" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
      --feature_dir "$DATA_ROOT/$SCENE/language_features_multiscale" \
      --provisional_memory_dir "$PROVISIONAL_MEMORY" \
      --output_dir "$FILTER_DIR" --seed "$SEED" --topk 45 \
      --scaffold_clusters 256 --scaffold_train_samples 200000 \
      --scaffold_iterations 25 --mask_iou_threshold 0.8 \
      --consensus_chunk_pixels 1024 --projection_chunk_pixels 8192 \
      --faiss_gpu > "$LOG_DIR/${SCENE}_post_aggregation_filter.log" 2>&1
  fi

  if [[ ! -f "$MEMORY_DIR/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
      --old_consensus "$A14_CONT_ROOT/$SCENE/old_split2/consensus.pt" \
      --sam_l0_consensus "$FILTER_DIR/sam_l0_split2/consensus.pt" \
      --sam_l1_consensus "$FILTER_DIR/sam_l1_split2/consensus.pt" \
      --sam_l2_consensus "$FILTER_DIR/sam_l2_split2/consensus.pt" \
      --sam_l3_consensus "$FILTER_DIR/sam_l3_split2/consensus.pt" \
      --output_dir "$MEMORY_DIR" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
      --stability_floor 0.50 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --fallback_reliability 0.05 --faiss_gpu \
      > "$LOG_DIR/${SCENE}_four_codebooks_retrain.log" 2>&1
  fi

  "$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
    --artifact_dir "$MEMORY_DIR" --required base sam_l0 sam_l1 sam_l2 sam_l3 \
    > "$LOG_DIR/${SCENE}_memory_contract.log" 2>&1

  if [[ ! -f "$EVAL_DIR/metrics.json" ]]; then
    mkdir -p "$EVAL_DIR"
    "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$DATA_ROOT/$SCENE" -m "$GEOMETRY_ROOT/$SCENE" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
      --codebook_dir "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_DISC_ROOT/$SCENE/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$MEMORY_DIR" --group_topk 4 \
      --group_readout equal_query_max \
      --group_query_temperature "$QUERY_TEMPERATURE" \
      --label_dir "$LABEL_ROOT/$SCENE" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds $SELECTION_THRESHOLDS --occupancy_threshold 0.7 \
      --output "$EVAL_DIR" > "$LOG_DIR/${SCENE}_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" \
  "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
  "$DATA_ROOT/$SCENE/language_features_multiscale" \
  "$LABEL_ROOT/$SCENE" "$PROVISIONAL_MEMORY/manifest.json" \
  "$A14_CONT_ROOT/$SCENE/old_split2/consensus.pt" \
  "$A14_DISC_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids/manifest.json" \
  "$CONTROL_METRICS"; do
  [[ -e "$required" ]] || { echo "Missing A40 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$SCENE" "$PROVISIONAL_MEMORY" \
  "$CONTROL_METRICS" "$PRIMARY_THRESHOLD" "$SEED" <<'PY'
import hashlib
import json
import os
import sys

root, scene, provisional_dir, control_path, raw_threshold, raw_seed = sys.argv[1:]
threshold = float(raw_threshold)
seed = int(raw_seed)
filter_dir = os.path.join(root, scene, "post_aggregation_filter")
memory_dir = os.path.join(root, scene, "filtered_four_token_memory")
eval_path = os.path.join(root, scene, "eval_equal_query_max", "metrics.json")


def row(path):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - threshold) < 1e-8
    )
    return {metric: float(item[metric]) for metric in ("mIoU", "mAcc@0.25", "mAcc@0.5")}


filter_manifest = json.load(open(os.path.join(filter_dir, "manifest.json")))
memory = json.load(open(os.path.join(memory_dir, "manifest.json")))
control = row(control_path)
result = row(eval_path)
fresh_hashes = {}
control_hashes = {}
for entry in memory["level_codebooks"]:
    fresh_hashes[entry["name"]] = hashlib.sha256(
        open(os.path.join(memory_dir, entry["codebook"]), "rb").read()
    ).hexdigest()
for entry in json.load(open(os.path.join(provisional_dir, "manifest.json")))["level_codebooks"]:
    control_hashes[entry["name"]] = hashlib.sha256(
        open(os.path.join(provisional_dir, entry["codebook"]), "rb").read()
    ).hexdigest()

delta = {metric: result[metric] - control[metric] for metric in result}
summary = {
    "method": "A40 SFS-style post-aggregation filtering plus fresh four-codebook memory",
    "scene": scene,
    "seed": seed,
    "selection_threshold": threshold,
    "filter": {
        "mask_iou_threshold": filter_manifest["mask_iou_threshold"],
        "scaffold": filter_manifest["scaffold"],
        "levels": [entry["filter_diagnostics"] for entry in filter_manifest["levels"]],
    },
    "control_a33_equal_query_max": control,
    "a40_equal_query_max": result,
    "delta_from_a33": delta,
    "codebooks": {
        "fresh": fresh_hashes,
        "provisional": control_hashes,
        "all_retrained": all(fresh_hashes[name] != control_hashes[name] for name in fresh_hashes),
        "sizes": [entry["num_codes"] for entry in memory["level_codebooks"]],
    },
    "decision": {
        "all_four_codebooks_retrained": all(
            fresh_hashes[name] != control_hashes[name] for name in fresh_hashes
        ),
        "keeps_masks_at_every_level": all(
            entry["filter_diagnostics"]["kept_masks"] > 0
            for entry in filter_manifest["levels"]
        ),
        "improves_miou_by_half_point": delta["mIoU"] >= 0.005,
        "preserves_strict_accuracy": delta["mAcc@0.5"] >= 0.0,
    },
}
summary["decision"]["expand_three_scene"] = all(summary["decision"].values())
with open(os.path.join(root, "ramen_probe_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A40 Ramen post-aggregation filter probe complete: $RUN_ROOT"
