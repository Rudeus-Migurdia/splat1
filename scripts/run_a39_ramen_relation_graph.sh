#!/usr/bin/env bash
set -euo pipefail

# A39: keep four peer semantic tokens unchanged, then apply one local signed
# graph step to their query scores using label-free multiview SAM agreement.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
SCENE=${SCENE:-ramen}
GPU=${GPU:-4}
SEED=${SEED:-20260719}
MEMORY_DIR=${MEMORY_DIR:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/equal_four_token_memory}
CONTROL_METRICS=${CONTROL_METRICS:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948/$SCENE/eval_equal_query_max/metrics.json}
CACHE_DIR=${CACHE_DIR:-$ROOT/runs/a6_responsibility_multiscene_20260715/$SCENE/cache_l2_raw}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
LABEL_ROOT=${LABEL_ROOT:-$ROOT/drsplat_data/lerf_ovs/label}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.05}
SELECTION_THRESHOLDS=${SELECTION_THRESHOLDS:-"0.50 0.55"}
PRIMARY_THRESHOLD=${PRIMARY_THRESHOLD:-0.55}
RELATION_DIR=$RUN_ROOT/$SCENE/local_relation_graph
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

evaluate_variant() {
  local name=$1 positive_strength=$2 negative_strength=$3
  local output=$RUN_ROOT/$SCENE/eval_$name
  [[ -f "$output/metrics.json" ]] && return
  mkdir -p "$output"
  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
    -s "$DATA_ROOT/$SCENE" -m "$GEOMETRY_ROOT/$SCENE" \
    --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
    --codebook_dir "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids" \
    --query_route_base_codebook_dir "$A14_DISC_ROOT/$SCENE/base_ids" \
    --codebook_query_route query_positive \
    --group_hierarchy_dir "$MEMORY_DIR" --group_topk 4 \
    --group_readout equal_query_relation_graph \
    --group_query_temperature "$QUERY_TEMPERATURE" \
    --group_relation_graph_dir "$RELATION_DIR" \
    --group_relation_positive_strength "$positive_strength" \
    --group_relation_negative_strength "$negative_strength" \
    --group_relation_maximum_delta 0.05 \
    --label_dir "$LABEL_ROOT/$SCENE" \
    --evaluation_protocol drsplat_3d_selection \
    --selection_thresholds $SELECTION_THRESHOLDS --occupancy_threshold 0.7 \
    --output "$output" > "$LOG_DIR/${SCENE}_${name}_eval.log" 2>&1
}

run_worker() {
  if [[ ! -f "$RELATION_DIR/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_multiview_local_relation_graph.py" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
      --memory_dir "$MEMORY_DIR" --cache_dir "$CACHE_DIR" \
      --output_dir "$RELATION_DIR" --neighbors 8 \
      --spatial_radius_factor 1.5 --minimum_dominant_fraction 0.55 \
      --minimum_split_views 3 --minimum_absolute_relation 0.05 \
      --chunk_size 65536 --knn_workers 6 \
      --expected_memory_seed "$SEED" --faiss_gpu \
      > "$LOG_DIR/${SCENE}_relation_graph_build.log" 2>&1
  fi

  evaluate_variant signed_main 0.20 0.10
  evaluate_variant positive_only 0.20 0.00
  evaluate_variant signed_weak 0.10 0.05
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_multiview_local_relation_graph.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED" \
  "$GEOMETRY_ROOT/$SCENE/chkpnt30000.pth" \
  "$MEMORY_DIR/manifest.json" "$CACHE_DIR/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/base_ids/manifest.json" \
  "$A14_DISC_ROOT/$SCENE/pruned_candidate_ids/manifest.json" \
  "$LABEL_ROOT/$SCENE" "$CONTROL_METRICS"; do
  [[ -e "$required" ]] || { echo "Missing A39 input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - "$MEMORY_DIR" "$CACHE_DIR" "$SEED" <<'PY' \
  > "$LOG_DIR/input_contract.log" 2>&1
import hashlib
import json
import os
import sys

import numpy as np

memory_dir, cache_dir, raw_seed = sys.argv[1:]
seed = int(raw_seed)
memory = json.load(open(os.path.join(memory_dir, "manifest.json")))
cache = json.load(open(os.path.join(cache_dir, "manifest.json")))
assert memory["representation"] == "hierarchical_independent_group_codebooks"
assert memory["resident_slots_required"] == 4
assert memory["reproducibility"]["seed"] == seed
assert [item["num_codes"] for item in memory["level_codebooks"]] == [
    2048, 4096, 8192, 16384
]
ids = np.load(os.path.join(memory_dir, memory["point_group_ids"]), mmap_mode="r")
assert ids.shape == (memory["num_gaussians"], 4)
assert cache["num_gaussians"] == memory["num_gaussians"]
assert cache["raw_contribution_weights"] and cache["topk"] >= 45
assert len(cache["views"]) >= 6
hashes = {}
for item in memory["level_codebooks"]:
    path = os.path.join(memory_dir, item["codebook"])
    hashes[item["name"]] = hashlib.sha256(open(path, "rb").read()).hexdigest()
print("A39_INPUT_CONTRACT_OK", json.dumps(hashes, sort_keys=True))
PY

"$PYTHON_BIN" "$GPU_GUARD" --gpu "$GPU" --hold-mb 384 \
  --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
  bash "$0" --worker > "$LOG_DIR/worker_gpu_${GPU}.log" 2>&1

"$PYTHON_BIN" - "$RUN_ROOT" "$SCENE" "$CONTROL_METRICS" \
  "$PRIMARY_THRESHOLD" "$SEED" <<'PY'
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
for name in ("signed_main", "positive_only", "signed_weak"):
    path = os.path.join(root, scene, f"eval_{name}", "metrics.json")
    values, payload = row(path)
    relation = [
        item.get("relation_graph", {})
        for item in payload.get("route_diagnostics", {}).values()
    ]
    variants[name] = {
        "metrics": values,
        "delta_from_equal_query_max": {
            metric: values[metric] - control[metric] for metric in metrics
        },
        "mean_absolute_score_delta": sum(
            float(item.get("mean_absolute_delta_active", 0.0)) for item in relation
        ) / max(len(relation), 1),
        "mean_clipped_fraction_active": sum(
            float(item.get("clipped_fraction_active", 0.0)) for item in relation
        ) / max(len(relation), 1),
    }

graph = json.load(open(os.path.join(root, scene, "local_relation_graph", "manifest.json")))
primary = variants["signed_main"]
summary = {
    "method": "A39 3D-first multiview local signed relation correction",
    "scene": scene,
    "seed": int(raw_seed),
    "selection_threshold": threshold,
    "codebook_contract": "A33 seed-20260719 four independent codebooks reused byte-for-byte; token source and assignments unchanged",
    "control_equal_query_max": control,
    "variants": variants,
    "relation_graph": graph,
    "decision": {
        "primary_improves_miou_by_half_point": primary["delta_from_equal_query_max"]["mIoU"] >= 0.005,
        "primary_preserves_strict_accuracy": primary["delta_from_equal_query_max"]["mAcc@0.5"] >= 0.0,
        "primary_reaches_ramen_40": primary["metrics"]["mIoU"] >= 0.40,
    },
}
summary["decision"]["expand_three_scene"] = all(summary["decision"].values())
with open(os.path.join(root, "ramen_probe_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A39 Ramen relation-graph probe complete: $RUN_ROOT"
