#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A52 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A52 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU=${GPU:-1}
SEED=${SEED:-20260719}
SCENE=ramen
A47_RUN=${A47_RUN:-$ROOT/runs/a47_raw_entity_tomography_20260721_181309}
A27_RUN=${A27_RUN:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
A14_CONT=${A14_CONT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC=${A14_DISC:-$ROOT/runs/a14_e8_joint32k_20260716}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A51_RUN=${A51_RUN:-$ROOT/runs/a51_2_composite_group_refinement_20260721_211504}
SCENE_ROOT=$RUN_ROOT/$SCENE
MEMORY=$SCENE_ROOT/fresh_equal_four_token_memory
SPATIAL=$SCENE_ROOT/query_conditioned_spatial_posterior
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
  "$SOURCE_DIR/build_query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$A47_RUN/PROBE_COMPLETE" "$GEOMETRY" "$LABEL_DIR"; do
  [[ -e "$required" ]] || { echo "Missing A52 input: $required" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  --geometry_checkpoint "$GEOMETRY" \
  --old_consensus "$A14_CONT/$SCENE/old_split2/consensus.pt" \
  --sam_l0_consensus "$A27_RUN/$SCENE/sam_l0_split2/consensus.pt" \
  --sam_l1_consensus "$A27_RUN/$SCENE/sam_l1_split2/consensus.pt" \
  --sam_l2_consensus "$A27_RUN/$SCENE/sam_l2_split2/consensus.pt" \
  --sam_l3_consensus "$A27_RUN/$SCENE/sam_l3_split2/consensus.pt" \
  --output_dir "$MEMORY" --device cuda --seed "$SEED" --neighbors 8 \
  --semantic_thresholds 0.76 0.82 0.87 0.91 \
  --maximum_group_sizes 2048 512 128 32 \
  --minimum_group_sizes 16 8 4 2 \
  --codes_per_level 2048 4096 8192 16384 \
  --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
  --stability_floor 0.50 --minimum_reliability 0.25 \
  --source_agreement_floor 0.80 --source_margin 0.0 --fallback_reliability 0.05 \
  --faiss_gpu > "$LOG_DIR/${SCENE}_fresh_four_codebooks.log" 2>&1

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u \
  "$SOURCE_DIR/build_query_conditioned_spatial_posterior.py" \
  --a47_audit_dir "$A47_RUN/$SCENE/entity_identifiability_audit" \
  --geometry_checkpoint "$GEOMETRY" --output_dir "$SPATIAL" --seed "$SEED" \
  --coverage_threshold 0.30 --minimum_spatial_jaccard 0.35 \
  --minimum_semantic_cosine 0.75 --minimum_association 0.40 \
  --spatial_weight 0.85 --temporal_neighbors 2 --minimum_persistence_views 3 \
  --merge_jaccard 0.85 --merge_semantic_cosine 0.90 --maximum_slots 4096 \
  --atom_neighbors 8 --minimum_atom_contact 0.05 \
  --core_coverage_threshold 0.30 --boundary_coverage_threshold 0.05 \
  --minimum_owner_membership 0.02 \
  > "$LOG_DIR/${SCENE}_spatial_posterior_build.log" 2>&1

"$PYTHON_BIN" - "$MEMORY" "$SPATIAL" "$SEED" <<'PY' \
  > "$LOG_DIR/${SCENE}_contract.log" 2>&1
import json, os, sys, numpy as np
memory, spatial, seed = sys.argv[1:]
m = json.load(open(os.path.join(memory, "manifest.json")))
s = json.load(open(os.path.join(spatial, "manifest.json")))
assert m["representation"] == "hierarchical_independent_group_codebooks"
assert [item["num_codes"] for item in m["level_codebooks"]] == [2048, 4096, 8192, 16384]
assert m["reproducibility"]["seed"] == int(seed)
ids = np.load(os.path.join(spatial, s["point_group_ids"]), mmap_mode="r")
memberships = np.load(os.path.join(spatial, s["point_group_memberships"]), mmap_mode="r")
assert ids.shape == memberships.shape == (s["num_gaussians"], 4, 2)
assert s["source_contract"]["semantic_tokens_not_modified"] is True
assert s["source_contract"]["spatial_posterior_applied_after_semantic_retrieval"] is True
assert s["source_contract"]["evaluation_queries_or_labels_used"] is False
assert s["source_contract"]["fixed_seed"] == int(seed)
print("A52_FULL_CONTRACT_OK", s["coverage"])
PY

evaluate() {
  local name=$1
  local readout=$2
  local output=$SCENE_ROOT/eval_$name
  if [[ -f "$output/metrics.json" ]]; then
    echo "Reuse completed A52 evaluation: $name"
    return
  fi
  mkdir -p "$output"
  local spatial_args=()
  if [[ "$readout" == equal_query_spatial_* ]]; then
    spatial_args=(
      --spatial_group_posterior_dir "$SPATIAL"
      --spatial_posterior_maximum_penalty 0.06
      --spatial_posterior_ring_weight 1.0
      --spatial_posterior_contrast_temperature 0.05
      --spatial_posterior_core_membership 0.30
      --spatial_posterior_entropy_relaxation 0.75
      --spatial_posterior_geodesic_delta 0.05
      --spatial_posterior_recovery_factor 0.20
    )
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
    "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE" \
    --geometry_checkpoint "$GEOMETRY" \
    --codebook_dir "$A14_DISC/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY" --group_topk 4 \
    --group_readout "$readout" --group_query_temperature 0.05 \
    "${spatial_args[@]}" \
    --label_dir "$LABEL_DIR" --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${SCENE}_${name}_eval.log" 2>&1
}

evaluate control_equal_query_max equal_query_max
evaluate spatial_posterior equal_query_spatial_posterior
evaluate spatial_geodesic equal_query_spatial_geodesic

"$PYTHON_BIN" - "$RUN_ROOT" "$A33_RUN" "$A51_RUN" <<'PY'
import json, os, sys
root, a33, a51 = sys.argv[1:]

def row(path):
    x = json.load(open(path))["threshold_summary"][0]
    return {k: x[k] for k in ("mIoU", "mAcc@0.25", "mAcc@0.5", "per_category")}

paths = {
    "a33": os.path.join(a33, "ramen", "eval_equal_query_max", "metrics.json"),
    "a51_composite_refine": os.path.join(a51, "ramen", "eval_composite_refine", "metrics.json"),
    "a52_fresh_control": os.path.join(root, "ramen", "eval_control_equal_query_max", "metrics.json"),
    "a52_spatial_posterior": os.path.join(root, "ramen", "eval_spatial_posterior", "metrics.json"),
    "a52_spatial_geodesic": os.path.join(root, "ramen", "eval_spatial_geodesic", "metrics.json"),
}
metrics = {name: row(path) for name, path in paths.items()}
control = metrics["a52_fresh_control"]
for name in ("a52_spatial_posterior", "a52_spatial_geodesic"):
    metrics[name]["delta_from_fresh_control"] = {
        key: metrics[name][key] - control[key]
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
    metrics[name]["bowl_delta_from_control"] = (
        metrics[name]["per_category"]["bowl"] - control["per_category"]["bowl"]
    )
winner = max(("a52_spatial_posterior", "a52_spatial_geodesic"), key=lambda n: metrics[n]["mIoU"])
summary = {
    "experiment": "A52_query_conditioned_spatial_posterior",
    "fixed_seed": 20260719,
    "evaluation": "TopK45, selection=0.55, occupancy=0.7",
    "metrics": metrics,
    "best_a52_variant": winner,
    "checks": {
        "fresh_control_matches_a33_within_0.001": abs(control["mIoU"] - metrics["a33"]["mIoU"]) <= 0.001,
        "best_recovers_a33_miou": metrics[winner]["mIoU"] >= metrics["a33"]["mIoU"],
        "best_preserves_a51_bowl": metrics[winner]["per_category"]["bowl"] >= 0.45,
        "best_preserves_strict_accuracy": metrics[winner]["mAcc@0.5"] >= metrics["a33"]["mAcc@0.5"],
    },
}
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A52 query-conditioned spatial posterior complete: $RUN_ROOT"
