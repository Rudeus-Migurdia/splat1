#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A51 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A51 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
GPU=${GPU:-1}
A47_RUN=${A47_RUN:-$ROOT/runs/a47_raw_entity_tomography_20260721_181309}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A33_MEMORY=${A33_MEMORY:-$A33_RUN/ramen/equal_four_token_memory}
A14_DISCRETE=${A14_DISCRETE:-$ROOT/runs/a14_e8_joint32k_20260716}
SCENE=ramen
SCENE_ROOT=$RUN_ROOT/$SCENE
MEMORY_ROOT=$SCENE_ROOT/full_group_addressed_memory
GEOMETRY=$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/$SCENE
CACHE_ROOT=$RUN_ROOT/.cache

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch HF_HOME=$CACHE_ROOT/huggingface
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$SCENE_ROOT" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

for required in \
  "$SOURCE_DIR/build_full_group_addressed_memory.py" \
  "$SOURCE_DIR/build_group_addressed_spatial_memory_audit.py" \
  "$SOURCE_DIR/build_persistent_entity_tomography.py" \
  "$SOURCE_DIR/build_multi_hypothesis_entity_tomography.py" \
  "$SOURCE_DIR/build_geometry_conditioned_tracklet_partition.py" \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$A47_RUN/PROBE_COMPLETE" \
  "$A47_RUN/$SCENE/entity_identifiability_audit/manifest.json" \
  "$GEOMETRY" "$LABEL_DIR"; do
  [[ -e "$required" ]] || { echo "Missing A51 input: $required" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
  "$SOURCE_DIR/build_full_group_addressed_memory.py" \
  --a47_audit_dir "$A47_RUN/$SCENE/entity_identifiability_audit" \
  --geometry_checkpoint "$GEOMETRY" \
  --reference_memory "$A33_MEMORY" \
  --output_dir "$MEMORY_ROOT" --seed "$SEED" \
  --coverage_threshold 0.30 \
  --minimum_spatial_jaccard 0.35 --minimum_semantic_cosine 0.75 \
  --minimum_association 0.40 --spatial_weight 0.85 \
  --temporal_neighbors 2 --minimum_persistence_views 3 \
  --merge_jaccard 0.85 --merge_semantic_cosine 0.90 --maximum_slots 4096 \
  --atom_neighbors 8 --minimum_atom_contact 0.05 \
  --core_coverage_threshold 0.30 --boundary_coverage_threshold 0.05 \
  --minimum_owner_membership 0.05 --boundary_margin 0.20 \
  --boundary_reliability_floor 0.25 --exterior_semantic_weight 0.35 \
  --codes_per_level 2048 4096 8192 16384 \
  --kmeans_iterations 25 --assignment_chunk_size 8192 --faiss_gpu \
  > "$LOG_DIR/${SCENE}_full_memory_train.log" 2>&1

validate_memory() {
  local memory=$1
  "$PYTHON_BIN" - "$memory" "$SEED" <<'PY'
import json
import os
import sys
import numpy as np

root, seed = sys.argv[1:]
manifest = json.load(open(os.path.join(root, "manifest.json")))
ids = np.load(os.path.join(root, manifest["point_group_ids"]), mmap_mode="r")
semantic = np.load(os.path.join(root, manifest["group_semantic_code_ids"]), mmap_mode="r")
levels = np.load(os.path.join(root, manifest["group_level"]), mmap_mode="r")
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert manifest["method"] in {
    "full_group_addressed_hierarchical_semantic_memory",
    "group_first_gaussian_refined_hierarchical_memory",
}
assert ids.shape == (manifest["num_gaussians"], 4)
assert semantic.shape == (manifest["num_spatial_groups"], 1)
assert levels.shape == (manifest["num_spatial_groups"],)
assert set(np.unique(levels).tolist()) == {0, 1, 2, 3}
assert all(item["num_codes"] > 0 for item in manifest["level_codebooks"])
assert manifest["reproducibility"]["seed"] == int(seed)
assert manifest["leakage_control"]["evaluation_queries_or_labels_used"] is False
print("A51_FULL_GROUP_ADDRESS_CONTRACT_OK", manifest["variant"], manifest["covered_fraction"])
PY
}

evaluate_memory() {
  local name=$1
  local memory=$2
  local output=$SCENE_ROOT/eval_$name
  validate_memory "$memory" > "$LOG_DIR/${SCENE}_${name}_contract.log" 2>&1
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
    "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE" \
    --geometry_checkpoint "$GEOMETRY" \
    --codebook_dir "$A14_DISCRETE/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISCRETE/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory" --group_topk 4 \
    --group_readout equal_query_max --group_query_temperature 0.05 \
    --label_dir "$LABEL_DIR" --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${SCENE}_${name}_eval.log" 2>&1
}

evaluate_memory core_key "$MEMORY_ROOT/core_key_memory"
evaluate_memory residual_key "$MEMORY_ROOT/residual_key_memory"
evaluate_memory gated_core_key "$MEMORY_ROOT/gated_core_key_memory"
evaluate_memory gated_residual_key "$MEMORY_ROOT/gated_residual_key_memory"
evaluate_memory composite_refine "$MEMORY_ROOT/composite_refine_memory"
evaluate_memory group_conditioned_refine "$MEMORY_ROOT/group_conditioned_refine_memory"

"$PYTHON_BIN" - "$RUN_ROOT" "$A33_RUN" <<'PY'
import json
import os
import sys

root, a33_root = sys.argv[1:]

def row(path):
    payload = json.load(open(path))
    item = payload["threshold_summary"][0]
    return {
        "mIoU": item["mIoU"],
        "mAcc@0.25": item["mAcc@0.25"],
        "mAcc@0.5": item["mAcc@0.5"],
        "per_category": item["per_category"],
    }

paths = {
    "a33_equal_four_token": os.path.join(a33_root, "ramen", "eval_equal_query_max", "metrics.json"),
    "a51_core_key": os.path.join(root, "ramen", "eval_core_key", "metrics.json"),
    "a51_residual_key": os.path.join(root, "ramen", "eval_residual_key", "metrics.json"),
    "a51_gated_core_key": os.path.join(root, "ramen", "eval_gated_core_key", "metrics.json"),
    "a51_gated_residual_key": os.path.join(root, "ramen", "eval_gated_residual_key", "metrics.json"),
    "a51_composite_refine": os.path.join(root, "ramen", "eval_composite_refine", "metrics.json"),
    "a51_group_conditioned_refine": os.path.join(root, "ramen", "eval_group_conditioned_refine", "metrics.json"),
}
metrics = {name: row(path) for name, path in paths.items()}
reference = metrics["a33_equal_four_token"]
for name in (
    "a51_core_key",
    "a51_residual_key",
    "a51_gated_core_key",
    "a51_gated_residual_key",
    "a51_composite_refine",
    "a51_group_conditioned_refine",
):
    metrics[name]["delta_from_a33"] = {
        key: metrics[name][key] - reference[key]
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
    metrics[name]["ramen_bowl_delta_from_a33"] = (
        metrics[name]["per_category"].get("bowl", 0.0)
        - reference["per_category"].get("bowl", 0.0)
    )
winner = max(
    (
        "a51_core_key",
        "a51_residual_key",
        "a51_gated_core_key",
        "a51_gated_residual_key",
        "a51_composite_refine",
        "a51_group_conditioned_refine",
    ),
    key=lambda name: metrics[name]["mIoU"],
)
summary = {
    "experiment": "A51_full_group_addressed_hierarchical_memory",
    "fixed_seed": 20260719,
    "evaluation": "TopK45, selection=0.55, occupancy=0.7",
    "metrics": metrics,
    "best_a51_variant": winner,
    "best_a51_beats_a33_miou": metrics[winner]["mIoU"] > reference["mIoU"],
    "best_a51_improves_bowl": metrics[winner]["ramen_bowl_delta_from_a33"] > 0.0,
}
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A51 full Group-addressed memory complete: $RUN_ROOT"
