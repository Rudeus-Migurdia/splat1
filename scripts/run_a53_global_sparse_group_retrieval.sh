#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A53 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A53 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU=${GPU:-1}
SEED=${SEED:-20260719}
SCENE=ramen
A52_RUN=${A52_RUN:-$ROOT/runs/a52_query_conditioned_spatial_posterior_20260721_213802}
A51_RUN=${A51_RUN:-$ROOT/runs/a51_2_composite_group_refinement_20260721_211504}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A14_DISC=${A14_DISC:-$ROOT/runs/a14_e8_joint32k_20260716}
MEMORY=$A52_RUN/$SCENE/fresh_equal_four_token_memory
SPATIAL=$A52_RUN/$SCENE/query_conditioned_spatial_posterior
GEOMETRY=$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth
LABEL_DIR=$ROOT/drsplat_data/lerf_ovs/label/$SCENE
SCENE_ROOT=$RUN_ROOT/$SCENE
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
  "$A52_RUN/PROBE_COMPLETE" "$MEMORY/manifest.json" "$SPATIAL/manifest.json" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$GEOMETRY" "$LABEL_DIR"; do
  [[ -e "$required" ]] || { echo "Missing A53 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$MEMORY" "$SPATIAL" "$SEED" <<'PY' > "$LOG_DIR/input_contract.log" 2>&1
import json, os, sys
memory, spatial, seed = sys.argv[1:]
m = json.load(open(os.path.join(memory, "manifest.json")))
s = json.load(open(os.path.join(spatial, "manifest.json")))
assert m["representation"] == "hierarchical_independent_group_codebooks"
assert [x["num_codes"] for x in m["level_codebooks"]] == [2048, 4096, 8192, 16384]
assert m["reproducibility"]["seed"] == int(seed)
assert s["representation"] == "query_conditioned_top2_spatial_group_posterior"
assert s["source_contract"]["evaluation_queries_or_labels_used"] is False
assert s["source_contract"]["top2_overlapping_groups"] is True
print("A53_GLOBAL_GROUP_INPUT_CONTRACT_OK")
PY

evaluate() {
  local name=$1
  local readout=$2
  local output=$SCENE_ROOT/eval_$name
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
    "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE" \
    --geometry_checkpoint "$GEOMETRY" \
    --codebook_dir "$A14_DISC/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY" --group_topk 4 \
    --spatial_group_posterior_dir "$SPATIAL" \
    --group_readout "$readout" --group_query_temperature 0.05 \
    --global_group_temperature 0.05 --global_group_semantic_weight 0.75 \
    --global_group_ring_contrast_strength 0.50 \
    --global_group_maximum_penalty 0.08 --global_group_entropy_relaxation 0.50 \
    --spatial_posterior_core_membership 0.30 \
    --spatial_posterior_geodesic_delta 0.05 \
    --spatial_posterior_recovery_factor 0.20 \
    --label_dir "$LABEL_DIR" --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds 0.55 --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${SCENE}_${name}_eval.log" 2>&1
}

evaluate global_sparsemax equal_query_global_sparsemax
evaluate global_entmax15 equal_query_global_entmax15
evaluate global_entmax15_geodesic equal_query_global_entmax15_geodesic

"$PYTHON_BIN" - "$RUN_ROOT" "$A33_RUN" "$A51_RUN" "$A52_RUN" <<'PY'
import json, os, sys
root, a33, a51, a52 = sys.argv[1:]

def row(path):
    x = json.load(open(path))["threshold_summary"][0]
    return {k: x[k] for k in ("mIoU", "mAcc@0.25", "mAcc@0.5", "per_category")}

paths = {
    "a33": os.path.join(a33, "ramen", "eval_equal_query_max", "metrics.json"),
    "a51_composite_refine": os.path.join(a51, "ramen", "eval_composite_refine", "metrics.json"),
    "a52_local_posterior": os.path.join(a52, "ramen", "eval_spatial_posterior", "metrics.json"),
    "a53_global_sparsemax": os.path.join(root, "ramen", "eval_global_sparsemax", "metrics.json"),
    "a53_global_entmax15": os.path.join(root, "ramen", "eval_global_entmax15", "metrics.json"),
    "a53_global_entmax15_geodesic": os.path.join(root, "ramen", "eval_global_entmax15_geodesic", "metrics.json"),
}
metrics = {name: row(path) for name, path in paths.items()}
reference = metrics["a33"]
variants = ("a53_global_sparsemax", "a53_global_entmax15", "a53_global_entmax15_geodesic")
for name in variants:
    metrics[name]["delta_from_a33"] = {
        key: metrics[name][key] - reference[key]
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
    metrics[name]["bowl_delta_from_a33"] = (
        metrics[name]["per_category"]["bowl"] - reference["per_category"]["bowl"]
    )
winner = max(variants, key=lambda name: metrics[name]["mIoU"])
summary = {
    "experiment": "A53_global_sparse_group_retrieval",
    "fixed_seed": 20260719,
    "evaluation": "TopK45, selection=0.55, occupancy=0.7",
    "metrics": metrics,
    "best_a53_variant": winner,
    "checks": {
        "best_beats_a33_miou": metrics[winner]["mIoU"] > reference["mIoU"],
        "best_preserves_strict_accuracy": metrics[winner]["mAcc@0.5"] >= reference["mAcc@0.5"],
        "best_bowl_at_least_0.45": metrics[winner]["per_category"]["bowl"] >= 0.45,
    },
}
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A53 global sparse Group retrieval complete: $RUN_ROOT"
