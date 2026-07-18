#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must point to the isolated A30 source snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A27_CACHE_ROOT=${A27_CACHE_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717_193243}
A28_ROOT=${A28_ROOT:-$ROOT/runs/a28_complementary_semantic_moe_20260717_223843}
A29_ROOT=${A29_ROOT:-$ROOT/runs/a29_sparse_l3_residual_20260718_085729}
E83_ROOT=${E83_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
BASELINE_ROOT=${BASELINE_ROOT:-$ROOT/runs/paper_selection_20260714}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be unique for A30}
LOG_DIR=${LOG_DIR:?LOG_DIR must be unique for A30}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
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

old_consensus() {
  printf '%s\n' "$A14_CONT_ROOT/$1/old_split2/consensus.pt"
}

level_consensus() {
  printf '%s\n' "$A27_CACHE_ROOT/$1/sam_l$2_split2/consensus.pt"
}

evaluate_scene() {
  local scene=$1
  local memory=$2
  local readout=$3
  local output=$4
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory" --group_topk 4 \
    --group_readout "$readout" \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output"
}

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local memory=$scene_root/equal_four_token_memory
  mkdir -p "$scene_root"

  if [[ ! -f "$memory/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --old_consensus "$(old_consensus "$scene")" \
      --sam_l0_consensus "$(level_consensus "$scene" 0)" \
      --sam_l1_consensus "$(level_consensus "$scene" 1)" \
      --sam_l2_consensus "$(level_consensus "$scene" 2)" \
      --sam_l3_consensus "$(level_consensus "$scene" 3)" \
      --output_dir "$memory" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
      --stability_floor 0.50 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --fallback_reliability 0.05 \
      --faiss_gpu > "$LOG_DIR/${scene}_four_codebooks_train.log" 2>&1
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
reliability = np.load(
    os.path.join(root, manifest["point_group_reliability"]), mmap_mode="r"
)
parents = np.load(os.path.join(root, manifest["group_parent_ids"]), mmap_mode="r")
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert manifest["resident_slots_required"] == 4 and manifest["top_m"] == 4
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [2048, 4096, 8192, 16384]
assert ids.shape == weights.shape == reliability.shape == (manifest["num_gaussians"], 4)
assert np.all(ids != manifest["invalid_id"])
assert np.all(weights == 255)
assert np.any(reliability > 0.0)
assert np.all(parents == -1)
assert "no parent preference" in manifest["codebook"]["query_readout"]
expected_seed = int(os.environ["PYTHONHASHSEED"])
assert manifest["reproducibility"]["seed"] == expected_seed
print(
    "A30_FOUR_EQUAL_TOKEN_CONTRACT_OK",
    manifest["usable_slot_fraction"],
    manifest["usable_covered_fraction"],
)
PY

  evaluate_scene "$scene" "$memory" equal_query_softmax \
    "$scene_root/eval_equal_query_softmax" \
    > "$LOG_DIR/${scene}_equal_query_softmax_eval.log" 2>&1
  evaluate_scene "$scene" "$memory" equal_query_max \
    "$scene_root/eval_equal_query_max" \
    > "$LOG_DIR/${scene}_equal_query_max_eval.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$ROOT/build_gaussian_multilevel_codebook.py" \
  "$ROOT/scripts/gpu_guard.py"; do
  [[ -f "$required" ]] || { echo "Missing required source: $required" >&2; exit 2; }
done

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || {
  echo "SCENES and GPU_LIST must have equal lengths" >&2
  exit 2
}
for scene in "${scenes[@]}"; do
  for level in 0 1 2 3; do
    required=$(level_consensus "$scene" "$level")
    [[ -f "$required" ]] || { echo "Missing level cache: $required" >&2; exit 2; }
  done
  for required in \
    "$(old_consensus "$scene")" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$E83_ROOT/$scene/eval/metrics.json" \
    "$A20_ROOT/$scene/eval_fine_part/metrics.json" \
    "$A28_ROOT/$scene/diagnostics/readout_ablation/eval_raw_top2_codebook/metrics.json" \
    "$A29_ROOT/$scene/eval_global_weak_l3/metrics.json" \
    "$BASELINE_ROOT/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

script_path=${BASH_SOURCE[0]}
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$script_path" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$E83_ROOT" "$A20_ROOT" "$A28_ROOT" \
  "$A29_ROOT" "$BASELINE_ROOT" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" \
  "${scenes[@]}" <<'PY'
import json
import os
import sys

root, e83, a20, a28, a29, baseline, raw_t, raw_bt, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metric_names}

def route_summary(path):
    diagnostics = json.load(open(path)).get("route_diagnostics", {})
    dominant = {f"level_{level}": 0 for level in range(4)}
    covered = []
    dominant_weight = []
    entropy = []
    fallback = []
    for item in diagnostics.values():
        for name, count in item.get("dominant_level_counts", {}).items():
            dominant[name] += int(count)
        covered.append(int(item.get("covered_points", 0)))
        fallback.append(int(item.get("fallback_points", 0)))
        dominant_weight.append(float(item.get("mean_dominant_token_weight", 0.0)))
        entropy.append(float(item.get("mean_normalized_token_entropy", 0.0)))
    total = sum(dominant.values())
    return {
        "dominant_level_counts": dominant,
        "dominant_level_fraction": {
            name: count / max(1, total) for name, count in dominant.items()
        },
        "mean_covered_points_per_query": sum(covered) / max(1, len(covered)),
        "mean_fallback_points_per_query": sum(fallback) / max(1, len(fallback)),
        "mean_dominant_token_weight": sum(dominant_weight) / max(1, len(dominant_weight)),
        "mean_normalized_token_entropy": sum(entropy) / max(1, len(entropy)),
    }

summary = {
    "method": "A30 four equal resident tokens with query-aware score fusion",
    "seed": int(os.environ["PYTHONHASHSEED"]),
    "query_temperature": 0.05,
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    memory_path = os.path.join(root, scene, "equal_four_token_memory", "manifest.json")
    memory = json.load(open(memory_path))
    softmax_metrics = os.path.join(root, scene, "eval_equal_query_softmax", "metrics.json")
    max_metrics = os.path.join(root, scene, "eval_equal_query_max", "metrics.json")
    summary["scenes"][scene] = {
        "paper_baseline_local": row(
            os.path.join(baseline, scene, "baseline", "metrics.json"), baseline_threshold
        ),
        "e8_3": row(os.path.join(e83, scene, "eval", "metrics.json"), threshold),
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), threshold),
        "a28_raw_top2": row(
            os.path.join(a28, scene, "diagnostics", "readout_ablation", "eval_raw_top2_codebook", "metrics.json"),
            threshold,
        ),
        "a29_global_weak_l3": row(
            os.path.join(a29, scene, "eval_global_weak_l3", "metrics.json"), threshold
        ),
        "a30_equal_query_softmax": row(softmax_metrics, threshold),
        "a30_equal_query_max": row(max_metrics, threshold),
        "softmax_route": route_summary(softmax_metrics),
        "max_route": route_summary(max_metrics),
        "vocabulary": {
            item["name"]: item["num_codes"] for item in memory["level_codebooks"]
        },
        "usable_slot_fraction": memory["usable_slot_fraction"],
        "usable_covered_fraction": memory["usable_covered_fraction"],
        "resident_slot_fraction": memory["covered_fraction"],
        "storage": memory["storage"],
    }
methods = (
    "paper_baseline_local",
    "e8_3",
    "a20",
    "a28_raw_top2",
    "a29_global_weak_l3",
    "a30_equal_query_softmax",
    "a30_equal_query_max",
)
for method in methods:
    summary[method + "_mean"] = {
        metric: sum(summary["scenes"][scene][method][metric] for scene in scenes) / len(scenes)
        for metric in metric_names
    }
for method in ("a30_equal_query_softmax", "a30_equal_query_max"):
    summary[method + "_minus_a20"] = {
        metric: summary[method + "_mean"][metric] - summary["a20_mean"][metric]
        for metric in metric_names
    }
summary["decision"] = {
    "softmax_beats_a20_miou": summary["a30_equal_query_softmax_minus_a20"]["mIoU"] > 0.0,
    "softmax_preserves_a20_strict_accuracy": summary["a30_equal_query_softmax_minus_a20"]["mAcc@0.5"] >= 0.0,
    "max_beats_a20_miou": summary["a30_equal_query_max_minus_a20"]["mIoU"] > 0.0,
    "max_preserves_a20_strict_accuracy": summary["a30_equal_query_max_minus_a20"]["mAcc@0.5"] >= 0.0,
    "all_scenes_store_four_resident_ids": all(
        summary["scenes"][scene]["resident_slot_fraction"] == 1.0 for scene in scenes
    ),
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A30 equal four-token query fusion complete: $RUN_ROOT"
