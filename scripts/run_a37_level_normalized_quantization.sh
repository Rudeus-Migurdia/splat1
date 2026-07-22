#!/usr/bin/env bash
set -euo pipefail

# A37: remove per-level codebook-capacity bias by converting each resident
# token's quantization error to a within-level empirical midrank percentile.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
GPU_GUARD=${GPU_GUARD:?GPU_GUARD is required}
MEMORY_ROOT=${MEMORY_ROOT:?MEMORY_ROOT is required}
DATA_ROOT=${DATA_ROOT:-$ROOT}
GEOMETRY_ROOT=${GEOMETRY_ROOT:?GEOMETRY_ROOT is required}
A14_DISC_ROOT=${A14_DISC_ROOT:?A14_DISC_ROOT is required}
A20_ROOT=${A20_ROOT:?A20_ROOT is required}
BASELINE_ROOT=${BASELINE_ROOT:?BASELINE_ROOT is required}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
SCALES=${SCALES:-"0.0025 0.005 0.01"}
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
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:?OPENCLIP_PRETRAINED is required}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

evaluate_scene() {
  local scene=$1 scale=$2 tag=${2/./p}
  local output=$RUN_ROOT/$scene/eval_level_normalized_lcb_${tag}
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/drsplat_data/lerf_ovs/$scene" -m "$GEOMETRY_ROOT/$scene" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY_ROOT/$scene/equal_four_token_memory" \
    --group_topk 4 --group_readout equal_query_quantization_percentile_lcb \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --group_quantization_uncertainty_scale "$scale" \
    --label_dir "$DATA_ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${scene}_scale_${tag}.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  local_scene=
  for local_scene in $SCENES; do
    local_scale=
    for local_scale in $SCALES; do evaluate_scene "$local_scene" "$local_scale"; done
  done
  exit 0
fi

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/eval_lerf_ovs_miou.py" \
  "$SOURCE_DIR/evaluation/openclip_encoder.py" \
  "$SOURCE_DIR/lerf_ovs_paper_protocol.py" \
  "$SOURCE_DIR/semantic_field_utils.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED"; do
  [[ -f "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done
for scene in $SCENES; do
  manifest=$MEMORY_ROOT/$scene/equal_four_token_memory/manifest.json
  [[ -f "$manifest" ]] || { echo "Missing A36 memory: $manifest" >&2; exit 2; }
  "$PYTHON_BIN" - "$manifest" "$SEED" <<'PY'
import json
import os
import sys

import numpy as np

manifest = json.load(open(sys.argv[1]))
root = os.path.dirname(sys.argv[1])
assert manifest["reproducibility"]["seed"] == int(sys.argv[2])
assert manifest["resident_slots_required"] == 4
assert [item["num_codes"] for item in manifest["level_codebooks"]] == [
    2048, 4096, 8192, 16384
]
error = np.load(os.path.join(root, manifest["point_group_quantization_error"]))
assert error.dtype == np.uint8 and error.shape[1] == 4
PY
  for required in \
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
import torch
from semantic_hypothesis_routing import fuse_quantization_aware_equal_query_tokens

base = torch.tensor([[0.2]])
candidate = torch.tensor([[0.80, 0.79]])
percentile = torch.tensor([[0.95, 0.05]])
ones = torch.ones_like(candidate)
output, stats = fuse_quantization_aware_equal_query_tokens(
    base, candidate, ones, ones, percentile, ones.bool(), 0.05, 0.02,
    uncertainty_measure_name="within_level_error_midrank_percentile",
)
assert output.item() == candidate[0, 1].item()
assert stats["quantization_uncertainty_measure"] == "within_level_error_midrank_percentile"
print("A37_UNIT_CONTRACT_OK")
PY
) > "$LOG_DIR/unit_contract.log" 2>&1

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$MEMORY_ROOT" "$A20_ROOT" "$BASELINE_ROOT" \
  "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" "$SCALES" $SCENES <<'PY'
import json
import os
import statistics
import sys

root, memory_root, a20_root, baseline_root, raw_t, raw_bt, raw_scales, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
scales = raw_scales.split()
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def row(path, selected_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selected_threshold) < 1e-8
    )
    return {name: float(item[name]) for name in metrics}, payload


def mean(rows):
    return {name: statistics.mean(item[name] for item in rows.values()) for name in metrics}


raw_rows = {
    scene: row(os.path.join(memory_root, scene, "eval_equal_query_max", "metrics.json"), threshold)[0]
    for scene in scenes
}
raw_mean = mean(raw_rows)
variants = {}
for scale in scales:
    tag = scale.replace(".", "p")
    scene_rows, route = {}, {}
    for scene in scenes:
        scene_rows[scene], payload = row(
            os.path.join(root, scene, f"eval_level_normalized_lcb_{tag}", "metrics.json"),
            threshold,
        )
        diagnostics = payload.get("route_diagnostics", {})
        covered = sum(int(item.get("covered_points", 0)) for item in diagnostics.values())
        route[scene] = {
            "ambiguous_fraction_covered": sum(
                int(item.get("ambiguous_points", 0)) for item in diagnostics.values()
            ) / max(covered, 1),
            "selection_changed_fraction_covered": sum(
                int(item.get("selection_changed_points", 0)) for item in diagnostics.values()
            ) / max(covered, 1),
            "mean_selected_within_level_error_percentile": statistics.mean(
                float(item.get("mean_selected_uncertainty_measure", 0.0))
                for item in diagnostics.values()
            ),
        }
    variant_mean = mean(scene_rows)
    variants[scale] = {
        "mean": variant_mean,
        "scenes": scene_rows,
        "route_diagnostics": route,
        "delta_from_raw_max": {
            name: variant_mean[name] - raw_mean[name] for name in metrics
        },
    }
best_scale = max(scales, key=lambda scale: variants[scale]["mean"]["mIoU"])
a20 = mean({
    scene: row(os.path.join(a20_root, scene, "eval_fine_part", "metrics.json"), threshold)[0]
    for scene in scenes
})
baseline = mean({
    scene: row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), baseline_threshold)[0]
    for scene in scenes
})
summary = {
    "method": "A37 within-level quantization-error percentile LCB routing",
    "seed": 20260719,
    "scenes": scenes,
    "raw_max": raw_mean,
    "a20_mean": a20,
    "baseline_mean": baseline,
    "variants": variants,
    "best_scale": best_scale,
    "decision": {
        "beats_raw_miou": variants[best_scale]["mean"]["mIoU"] > raw_mean["mIoU"],
        "acc50_not_worse_than_raw": variants[best_scale]["mean"]["mAcc@0.5"] >= raw_mean["mAcc@0.5"],
        "advance_quantization_confidence_direction": (
            variants[best_scale]["mean"]["mIoU"] > raw_mean["mIoU"]
            and variants[best_scale]["mean"]["mAcc@0.5"] >= raw_mean["mAcc@0.5"]
        ),
    },
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as target:
    json.dump(summary, target, indent=2)
open(os.path.join(root, "PROBE_COMPLETE"), "w").close()
print(json.dumps(summary, indent=2))
PY
