#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A31 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
E83_ROOT=${E83_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/runs/paper_selection_20260714}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A31}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A31}
SCENE=${SCENE:-teatime}
GPU=${GPU:-1}
SEED=${SEED:-20260717}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p "$RUN_ROOT" "$LOG_DIR"

prepare_level_cache() {
  local level=$1
  local cache=$RUN_ROOT/$SCENE/sam_l${level}_split2
  if [[ -f "$cache/manifest.json" && -f "$cache/consensus.pt" ]]; then
    return
  fi
  "$PYTHON_BIN" -u "$SOURCE_DIR/prepare_semantic_field.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$cache" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
    --feature_dir "$ROOT/drsplat_data/lerf_ovs/$SCENE/language_features_multiscale" \
    --feature_level "$level" --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --signed_segment_ownership --consensus_only --consensus_splits 2 \
    --seed "$SEED" > "$LOG_DIR/${SCENE}_sam_l${level}_cache.log" 2>&1
}

evaluate_readout() {
  local readout=$1
  local output=$RUN_ROOT/$SCENE/eval_${readout}
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$SCENE" -m "$ROOT/runs/3dgs/$SCENE" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$RUN_ROOT/$SCENE/equal_four_token_memory" \
    --group_topk 4 --group_readout "$readout" \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output"
}

run_scene() {
  local level
  for level in 0 1 2 3; do
    prepare_level_cache "$level"
  done
  local memory=$RUN_ROOT/$SCENE/equal_four_token_memory
  if [[ ! -f "$memory/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
      --old_consensus "$A14_CONT_ROOT/$SCENE/old_split2/consensus.pt" \
      --sam_l0_consensus "$RUN_ROOT/$SCENE/sam_l0_split2/consensus.pt" \
      --sam_l1_consensus "$RUN_ROOT/$SCENE/sam_l1_split2/consensus.pt" \
      --sam_l2_consensus "$RUN_ROOT/$SCENE/sam_l2_split2/consensus.pt" \
      --sam_l3_consensus "$RUN_ROOT/$SCENE/sam_l3_split2/consensus.pt" \
      --output_dir "$memory" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
      --stability_floor 0.50 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --fallback_reliability 0.05 --faiss_gpu \
      > "$LOG_DIR/${SCENE}_four_codebooks_train.log" 2>&1
  fi

  "$PYTHON_BIN" - "$memory" <<'PY'
import json
import os
import sys
import numpy as np

root = sys.argv[1]
manifest = json.load(open(os.path.join(root, "manifest.json")))
ids = np.load(os.path.join(root, manifest["point_group_ids"]), mmap_mode="r")
weights = np.load(os.path.join(root, manifest["point_group_weights"]), mmap_mode="r")
parents = np.load(os.path.join(root, manifest["group_parent_ids"]), mmap_mode="r")
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [2048, 4096, 8192, 16384]
assert ids.shape == weights.shape == (manifest["num_gaussians"], 4)
assert np.all(ids != manifest["invalid_id"])
assert np.all(weights == 255) and np.all(parents == -1)
assert manifest["reproducibility"]["seed"] == 20260717
print(
    "A31_TEATIME_FOUR_TOKEN_CONTRACT_OK",
    manifest["usable_slot_fraction"],
    manifest["usable_covered_fraction"],
)
PY

  evaluate_readout equal_query_softmax \
    > "$LOG_DIR/${SCENE}_equal_query_softmax_eval.log" 2>&1
  evaluate_readout equal_query_max \
    > "$LOG_DIR/${SCENE}_equal_query_max_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/prepare_semantic_field.py" \
  "$ROOT/scripts/gpu_guard.py" \
  "$ROOT/drsplat_data/lerf_ovs/$SCENE/language_features_multiscale" \
  "$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth" \
  "$A14_CONT_ROOT/$SCENE/old_split2/consensus.pt" \
  "$A14_DISC_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids/manifest.json" \
  "$E83_ROOT/$SCENE/eval/metrics.json" \
  "$BASELINE_ROOT/$SCENE/baseline/metrics.json"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

if [[ "${RUN_WITHOUT_GUARD:-0}" == "1" ]]; then
  run_scene
else
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$GPU" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "${BASH_SOURCE[0]}" --worker \
    > "$LOG_DIR/worker_${SCENE}_gpu_${GPU}.log" 2>&1
fi

"$PYTHON_BIN" - "$RUN_ROOT" "$E83_ROOT" "$A14_DISC_ROOT" "$BASELINE_ROOT" \
  "$SCENE" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" <<'PY'
import json
import os
import sys

root, e83, a14, baseline, scene, raw_t, raw_bt = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metrics}

def route(path):
    payload = json.load(open(path))
    dominant = {f"level_{level}": 0 for level in range(4)}
    entropies = []
    for item in payload.get("route_diagnostics", {}).values():
        for name, count in item.get("dominant_level_counts", {}).items():
            dominant[name] += int(count)
        entropies.append(float(item.get("mean_normalized_token_entropy", 0.0)))
    total = sum(dominant.values())
    return {
        "dominant_level_fraction": {
            name: count / max(1, total) for name, count in dominant.items()
        },
        "mean_normalized_token_entropy": sum(entropies) / max(1, len(entropies)),
    }

scene_root = os.path.join(root, scene)
softmax_path = os.path.join(scene_root, "eval_equal_query_softmax", "metrics.json")
max_path = os.path.join(scene_root, "eval_equal_query_max", "metrics.json")
memory = json.load(open(os.path.join(scene_root, "equal_four_token_memory", "manifest.json")))
summary = {
    "method": "A31 teatime validation of A30 equal four-token query fusion",
    "scene": scene,
    "seed": int(os.environ["PYTHONHASHSEED"]),
    "selection_threshold": threshold,
    "paper_baseline_local": row(
        os.path.join(baseline, scene, "baseline", "metrics.json"), baseline_threshold
    ),
    "e8_3": row(os.path.join(e83, scene, "eval", "metrics.json"), threshold),
    "a14_e8_joint32k": row(os.path.join(a14, scene, "eval", "metrics.json"), threshold),
    "a31_equal_query_softmax": row(softmax_path, threshold),
    "a31_equal_query_max": row(max_path, threshold),
    "softmax_route": route(softmax_path),
    "max_route": route(max_path),
    "usable_slot_fraction": memory["usable_slot_fraction"],
    "usable_covered_fraction": memory["usable_covered_fraction"],
    "storage": memory["storage"],
}
for method in ("a31_equal_query_softmax", "a31_equal_query_max"):
    summary[method + "_minus_e8_3"] = {
        name: summary[method][name] - summary["e8_3"][name] for name in metrics
    }
summary["decision"] = {
    "max_beats_e8_3_miou": summary["a31_equal_query_max_minus_e8_3"]["mIoU"] > 0.0,
    "max_preserves_e8_3_strict_accuracy": summary["a31_equal_query_max_minus_e8_3"]["mAcc@0.5"] >= 0.0,
}
with open(os.path.join(root, "teatime_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A31 teatime validation complete: $RUN_ROOT"
