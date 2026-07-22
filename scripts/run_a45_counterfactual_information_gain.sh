#!/usr/bin/env bash
set -euo pipefail

# A45 stage 1: four peer tokens remain fixed. Route by complete-codebook
# information gain, optionally penalized by same-level counterfactual ambiguity.
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
RUNTIME_ROOT=${RUNTIME_ROOT:-$RUN_ROOT/runtime}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
INPUT_ROOT=${INPUT_ROOT:-$RUN_ROOT/inputs}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/Dr-Splat-envs/drsplat236_py39/bin/python}
GPU_GUARD=${GPU_GUARD:-$SOURCE_DIR/gpu_guard.py}
GPU=${GPU:-1}
SEED=${SEED:-20260719}
SCENE=${SCENE:-ramen}
MEMORY_DIR=${MEMORY_DIR:-$INPUT_ROOT/a36_fixed_seed_20260719/$SCENE/equal_four_token_memory}
A14_ROOT=${A14_ROOT:-$INPUT_ROOT/a14}
DATA_ROOT=${DATA_ROOT:-$INPUT_ROOT/data/lerf_ovs}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$INPUT_ROOT/geometry}
CONTROL_METRICS=${CONTROL_METRICS:-$INPUT_ROOT/control/$SCENE/metrics.json}
OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$INPUT_ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
COUNTERFACTUAL_DIR=${COUNTERFACTUAL_DIR:-$RUN_ROOT/$SCENE/counterfactual_codebook_neighborhoods}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

SITE=$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$RUNTIME_ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

evaluate_variant() {
  local mode=$1
  local output=$RUN_ROOT/$SCENE/eval_$mode
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  local counterfactual_args=()
  if [[ "$mode" == "equal_query_counterfactual_information_gain" ]]; then
    counterfactual_args=(--group_counterfactual_codebook_dir "$COUNTERFACTUAL_DIR")
  fi
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/$SCENE" -m "$GEOMETRY_ROOT/$SCENE" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$A14_ROOT/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_ROOT/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY_DIR" --group_topk 4 \
    --group_readout "$mode" --group_query_temperature "$QUERY_TEMPERATURE" \
    "${counterfactual_args[@]}" \
    --label_dir "$DATA_ROOT/label/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${SCENE}_${mode}.log" 2>&1
}

run_worker() {
  evaluate_variant equal_query_information_gain
  evaluate_variant equal_query_counterfactual_information_gain
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for required in \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/build_counterfactual_codebook_neighborhoods.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" \
  "$MEMORY_DIR/manifest.json" "$COUNTERFACTUAL_DIR/manifest.json" \
  "$A14_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_ROOT/$SCENE/pruned_candidate_ids/manifest.json" \
  "$DATA_ROOT/$SCENE" "$DATA_ROOT/label/$SCENE" \
  "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
  "$GEOMETRY_ROOT/$SCENE/cfg_args" "$CONTROL_METRICS"; do
  [[ -e "$required" ]] || { echo "Missing A45 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$MEMORY_DIR" "$COUNTERFACTUAL_DIR" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import hashlib
import json
import os
import sys

import numpy as np

memory_dir, neighbor_dir, raw_seed = sys.argv[1:]
memory_path = os.path.join(memory_dir, "manifest.json")
memory = json.load(open(memory_path))
neighbors = json.load(open(os.path.join(neighbor_dir, "manifest.json")))
assert memory["representation"] == "hierarchical_independent_group_codebooks"
assert memory["resident_slots_required"] == 4
assert memory["reproducibility"]["seed"] == int(raw_seed)
assert neighbors["representation"] == "hierarchical_codebook_counterfactual_neighborhoods"
assert neighbors["memory_manifest_sha256"] == hashlib.sha256(open(memory_path, "rb").read()).hexdigest()
ids = np.load(os.path.join(memory_dir, memory["point_group_ids"]), mmap_mode="r")
assert ids.shape == (memory["num_gaussians"], 4)
assert [item["level"] for item in neighbors["levels"]] == [0, 1, 2, 3]
print("A45_INPUT_CONTRACT_OK")
PY

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$SCENE" "$CONTROL_METRICS" \
  "$SELECTION_THRESHOLD" "$SEED" <<'PY'
import json
import os
import sys

root, scene, control_path, raw_threshold, raw_seed = sys.argv[1:]
threshold = float(raw_threshold)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")


def row(path):
    payload = json.load(open(path))
    selected = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(selected[name]) for name in metrics}, payload


control, control_payload = row(control_path)
variants = {}
for name in (
    "equal_query_information_gain",
    "equal_query_counterfactual_information_gain",
):
    values, payload = row(os.path.join(root, scene, f"eval_{name}", "metrics.json"))
    route = list(payload.get("route_diagnostics", {}).values())
    variants[name] = {
        "metrics": values,
        "delta_from_fixed_memory_raw_max": {
            metric: values[metric] - control[metric] for metric in metrics
        },
        "mean_selected_information_gain": sum(
            float(item.get("mean_selected_information_gain", 0.0)) for item in route
        ) / max(len(route), 1),
        "mean_selected_counterfactual_penalty": sum(
            float(item.get("mean_selected_counterfactual_penalty", 0.0)) for item in route
        ) / max(len(route), 1),
        "dominant_level_counts": {
            f"level_{level}": sum(
                int(item.get("dominant_level_counts", {}).get(f"level_{level}", 0))
                for item in route
            )
            for level in range(4)
        },
    }

best_name = max(variants, key=lambda name: variants[name]["metrics"]["mIoU"])
best = variants[best_name]
a20_ramen = {
    "mIoU": 0.30721603001157216,
    "mAcc@0.25": 0.4507042253521127,
    "mAcc@0.5": 0.323943661971831,
}
decision = {
    "ramen_miou_gain_at_least_5_points": (
        best["delta_from_fixed_memory_raw_max"]["mIoU"] >= 0.05
    ),
    "ramen_acc025_not_below_a20": best["metrics"]["mAcc@0.25"] >= a20_ramen["mAcc@0.25"],
}
decision["expand_three_scene"] = all(decision.values())
summary = {
    "method": "A45 occupancy-prior information gain with semantic counterfactual identifiability",
    "stage": "Ramen-first fixed-codebook readout falsification",
    "seed": int(raw_seed),
    "selection_threshold": threshold,
    "query_temperature": 0.05,
    "fixed_memory_control": control,
    "a20_ramen_reference": a20_ramen,
    "variants": variants,
    "best_variant": best_name,
    "decision": decision,
    "codebook_update": "not applicable in stage 1; token assignments and codebooks are fixed",
}
with open(os.path.join(root, "ramen_probe_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
marker = "READOUT_GATE_PASSED" if decision["expand_three_scene"] else "PROBE_COMPLETE"
with open(os.path.join(root, marker), "w") as output:
    output.write(best_name + "\n")
PY

echo "A45 Ramen probe complete: $RUN_ROOT"
