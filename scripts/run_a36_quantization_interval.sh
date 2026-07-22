#!/usr/bin/env bash
set -euo pipefail

# A36: retrain four seeded peer codebooks, retain per-slot quantization error,
# and resolve close query matches with quantization-adaptive score intervals.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
GPU_GUARD=${GPU_GUARD:?GPU_GUARD is required}
DATA_ROOT=${DATA_ROOT:-$ROOT}
GEOMETRY_ROOT=${GEOMETRY_ROOT:?GEOMETRY_ROOT is required}
OLD_ROOT=${OLD_ROOT:?OLD_ROOT is required}
SAM_ROOT=${SAM_ROOT:?SAM_ROOT is required}
A14_DISC_ROOT=${A14_DISC_ROOT:?A14_DISC_ROOT is required}
A20_ROOT=${A20_ROOT:?A20_ROOT is required}
BASELINE_ROOT=${BASELINE_ROOT:?BASELINE_ROOT is required}
CONTROL_SUMMARY=${CONTROL_SUMMARY:?CONTROL_SUMMARY is required}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
SCALES=${SCALES:-"0.025 0.05 0.10"}
SEED=${SEED:-20260719}
GPU=${GPU:-1}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

SITE=${SITE:-$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

old_consensus() {
  printf '%s\n' "$OLD_ROOT/$1/old_split2/consensus.pt"
}

sam_consensus() {
  printf '%s\n' "$SAM_ROOT/$1/sam_l$2_split2/consensus.pt"
}

evaluate_scene() {
  local scene=$1 memory=$2 mode=$3 output=$4 scale=${5:-0.0}
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  local extra=()
  if [[ "$mode" == "equal_query_quantization_lcb" ]]; then
    extra+=(--group_quantization_uncertainty_scale "$scale")
  fi
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" -m "$GEOMETRY_ROOT/$scene" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$memory" --group_topk 4 \
    --group_readout "$mode" --group_query_temperature "$QUERY_TEMPERATURE" \
    "${extra[@]}" \
    --label_dir "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output"
}

validate_memory() {
  local memory=$1
  "$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
    --artifact_dir "$memory" --required base sam_l0 sam_l1 sam_l2 sam_l3
  "$PYTHON_BIN" - "$memory" "$SEED" <<'PY'
import json
import os
import sys

import numpy as np

root, raw_seed = sys.argv[1:]
manifest = json.load(open(os.path.join(root, "manifest.json")))
ids = np.load(os.path.join(root, manifest["point_group_ids"]), mmap_mode="r")
errors = np.load(
    os.path.join(root, manifest["point_group_quantization_error"]), mmap_mode="r"
)
parents = np.load(os.path.join(root, manifest["group_parent_ids"]), mmap_mode="r")
assert manifest["format_version"] == 3
assert manifest["representation"] == "hierarchical_independent_group_codebooks"
assert manifest["resident_slots_required"] == manifest["top_m"] == 4
assert manifest["reproducibility"]["seed"] == int(raw_seed)
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [
    2048, 4096, 8192, 16384
]
assert ids.shape == errors.shape == (manifest["num_gaussians"], 4)
assert errors.dtype == np.uint8
assert manifest["point_group_quantization_error_quantizer"] == "ceil_upper_bound"
assert np.all(ids != manifest["invalid_id"])
assert np.all(parents == -1)
assert np.any(errors > 0)
print(
    "A36_QUANTIZATION_INTERVAL_CONTRACT_OK",
    errors.shape,
    float(errors.mean()) * manifest["point_group_quantization_error_scale"],
)
PY
}

run_scene() {
  local scene=$1 scene_root=$RUN_ROOT/$scene
  local memory=$scene_root/equal_four_token_memory
  mkdir -p "$scene_root"
  if [[ ! -f "$memory/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
      --old_consensus "$(old_consensus "$scene")" \
      --sam_l0_consensus "$(sam_consensus "$scene" 0)" \
      --sam_l1_consensus "$(sam_consensus "$scene" 1)" \
      --sam_l2_consensus "$(sam_consensus "$scene" 2)" \
      --sam_l3_consensus "$(sam_consensus "$scene" 3)" \
      --output_dir "$memory" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 \
      --assignment_chunk_size 2000000 \
      --stability_floor 0.50 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --fallback_reliability 0.05 --faiss_gpu \
      > "$LOG_DIR/${scene}_codebook_retrain.log" 2>&1
  fi
  validate_memory "$memory" > "$LOG_DIR/${scene}_memory_contract.log" 2>&1

  evaluate_scene "$scene" "$memory" equal_query_max \
    "$scene_root/eval_equal_query_max" \
    > "$LOG_DIR/${scene}_raw_max_eval.log" 2>&1
  local scale tag
  for scale in $SCALES; do
    tag=${scale/./p}
    evaluate_scene "$scene" "$memory" equal_query_quantization_lcb \
      "$scene_root/eval_quantization_lcb_${tag}" "$scale" \
      > "$LOG_DIR/${scene}_quantization_lcb_${tag}_eval.log" 2>&1
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  for scene in $SCENES; do run_scene "$scene"; done
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/build_hierarchical_semantic_memory.py" \
  "$SOURCE_DIR/build_hierarchical_group_semantic_codebook.py" \
  "$SOURCE_DIR/build_gaussian_multilevel_codebook.py" \
  "$SOURCE_DIR/build_gaussian_superpoint_support.py" \
  "$SOURCE_DIR/train_joint_query_preserving_vocabulary.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/eval_lerf_ovs_miou.py" \
  "$SOURCE_DIR/evaluation/openclip_encoder.py" \
  "$SOURCE_DIR/lerf_ovs_paper_protocol.py" \
  "$SOURCE_DIR/semantic_field_utils.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  "$GPU_GUARD" "$CONTROL_SUMMARY"; do
  [[ -f "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done
for scene in $SCENES; do
  for level in 0 1 2 3; do
    required=$(sam_consensus "$scene" "$level")
    [[ -f "$required" ]] || { echo "Missing SAM cache: $required" >&2; exit 2; }
  done
  for required in \
    "$(old_consensus "$scene")" \
    "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$A20_ROOT/$scene/eval_fine_part/metrics.json" \
    "$BASELINE_ROOT/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
  done
done

(cd "$SOURCE_DIR" && "$PYTHON_BIN" - <<'PY'
import numpy as np
import torch

from build_seeded_hierarchical_resident_memory import (
    CHORD_ERROR_SCALE,
    quantize_chord_error_upper_bound,
)
from semantic_hypothesis_routing import fuse_quantization_aware_equal_query_tokens

error = np.array([0.0, 0.001, 0.125, 1.0, 2.0], dtype=np.float32)
packed = quantize_chord_error_upper_bound(error)
assert packed.dtype == np.uint8
assert np.all(packed.astype(np.float32) * CHORD_ERROR_SCALE + 1e-7 >= error)
base = torch.tensor([[0.2], [0.2], [0.3]])
candidate = torch.tensor([[0.80, 0.79], [0.80, 0.70], [0.90, 0.85]])
errors = torch.tensor([[0.50, 0.05], [0.10, 0.10], [0.10, 0.10]])
ones = torch.ones_like(candidate)
valid = torch.tensor([[True, True], [True, True], [False, False]])
output, stats = fuse_quantization_aware_equal_query_tokens(
    base, candidate, ones, ones, errors, valid, 0.05, 0.10
)
assert torch.allclose(output, torch.tensor([[0.79], [0.80], [0.30]]))
assert stats["ambiguous_points"] == stats["selection_changed_points"] == 1
print("A36_UNIT_CONTRACT_OK")
PY
) > "$LOG_DIR/unit_contract.log" 2>&1

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$CONTROL_SUMMARY" "$A20_ROOT" "$BASELINE_ROOT" \
  "$SEED" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" "$SCALES" $SCENES <<'PY'
import json
import os
import statistics
import sys

root, control_path, a20_root, baseline_root, seed, raw_t, raw_bt, raw_scales, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
scales = raw_scales.split()
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metric_names}, payload


def mean(rows):
    return {
        name: statistics.mean(item[name] for item in rows.values())
        for name in metric_names
    }


def route_stats(payload):
    diagnostics = payload.get("route_diagnostics", {})
    total_covered = sum(int(item.get("covered_points", 0)) for item in diagnostics.values())
    return {
        "ambiguous_fraction_covered": sum(
            int(item.get("ambiguous_points", 0)) for item in diagnostics.values()
        ) / max(total_covered, 1),
        "selection_changed_fraction_covered": sum(
            int(item.get("selection_changed_points", 0)) for item in diagnostics.values()
        ) / max(total_covered, 1),
        "mean_selected_quantization_error": statistics.mean(
            float(item.get("mean_selected_quantization_error", 0.0))
            for item in diagnostics.values()
        ) if diagnostics else 0.0,
    }


control = json.load(open(control_path))["per_seed"][seed]["mean"]
variants = {}
modes = [("equal_query_max", "eval_equal_query_max", None)] + [
    (f"quantization_lcb_{scale}", f"eval_quantization_lcb_{scale.replace('.', 'p')}", scale)
    for scale in scales
]
for mode, directory, scale in modes:
    scene_rows, diagnostics = {}, {}
    for scene in scenes:
        scene_rows[scene], payload = row(
            os.path.join(root, scene, directory, "metrics.json"), threshold
        )
        if scale is not None:
            diagnostics[scene] = route_stats(payload)
    variants[mode] = {
        "scale": float(scale) if scale is not None else None,
        "mean": mean(scene_rows),
        "scenes": scene_rows,
        "route_diagnostics": diagnostics,
    }

raw = variants["equal_query_max"]["mean"]
for name, variant in variants.items():
    variant["delta_from_retrained_raw_max"] = {
        metric: variant["mean"][metric] - raw[metric] for metric in metric_names
    }
best_name = max(
    (name for name in variants if name != "equal_query_max"),
    key=lambda name: variants[name]["mean"]["mIoU"],
)
a20_rows = {
    scene: row(os.path.join(a20_root, scene, "eval_fine_part", "metrics.json"), threshold)[0]
    for scene in scenes
}
baseline_rows = {
    scene: row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), baseline_threshold)[0]
    for scene in scenes
}
summary = {
    "method": "A36 retrained four-token codebooks with quantization-adaptive LCB routing",
    "seed": int(seed),
    "scenes": scenes,
    "codes_per_level": [2048, 4096, 8192, 16384],
    "selection_threshold": threshold,
    "same_hardware_a33_control": control,
    "a20_mean": mean(a20_rows),
    "baseline_mean": mean(baseline_rows),
    "variants": variants,
    "best_variant": best_name,
    "decision": {
        "beats_retrained_raw_miou": variants[best_name]["mean"]["mIoU"] > raw["mIoU"],
        "acc50_not_worse_than_retrained_raw": (
            variants[best_name]["mean"]["mAcc@0.5"] >= raw["mAcc@0.5"]
        ),
        "advance_to_multiseed": (
            variants[best_name]["mean"]["mIoU"] > raw["mIoU"]
            and variants[best_name]["mean"]["mAcc@0.5"] >= raw["mAcc@0.5"]
        ),
    },
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as target:
    json.dump(summary, target, indent=2)
open(os.path.join(root, "PROBE_COMPLETE"), "w").close()
print(json.dumps(summary, indent=2))
PY
