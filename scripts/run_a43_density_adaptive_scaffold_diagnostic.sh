#!/usr/bin/env bash
set -euo pipefail

# A43: label-free three-scene diagnostic for an SFS-style PCA+HDBSCAN
# post-aggregation scaffold. Dependencies are isolated under this run root.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
SCENES=${SCENES:-"ramen figurines waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SEED=${SEED:-20260719}
HDBSCAN_NOISE_POLICY=${HDBSCAN_NOISE_POLICY:-pooled_background}
HDBSCAN_CLUSTER_SELECTION_METHOD=${HDBSCAN_CLUSTER_SELECTION_METHOD:-eom}
A33_ROOT=${A33_ROOT:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
A40_ROOT=${A40_ROOT:-$ROOT/runs/a40_post_aggregation_filter_20260719_162740}
A42_ROOT=${A42_ROOT:-$ROOT/runs/a42_filtered_l0_multiscene_20260719_165702}
DEPS_DIR=$RUN_ROOT/deps
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$DEPS_DIR:$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$DEPS_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

if [[ ! -f "$DEPS_DIR/DEPS_COMPLETE" ]]; then
  "$PYTHON_BIN" -m pip install --disable-pip-version-check --no-deps \
    --target "$DEPS_DIR" \
    scikit-learn==1.6.1 hdbscan==0.8.40 joblib==1.4.2 threadpoolctl==3.5.0 \
    > "$LOG_DIR/dependency_install.log" 2>&1
  "$PYTHON_BIN" - <<'PY' > "$LOG_DIR/dependency_contract.log" 2>&1
import hdbscan
import joblib
import sklearn
import threadpoolctl

print("hdbscan", getattr(hdbscan, "__version__", "0.8.40"))
print("scikit-learn", sklearn.__version__)
print("joblib", joblib.__version__)
print("threadpoolctl", threadpoolctl.__version__)
PY
  date +%FT%T > "$DEPS_DIR/DEPS_COMPLETE"
fi

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene/hdbscan_l0_diagnostic
  mkdir -p "$RUN_ROOT/$scene"
  if [[ ! -f "$output/DIAGNOSTIC_COMPLETE" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
      -s "$DATA_ROOT/$scene" -m "$RUN_ROOT/$scene/model_context" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
      --feature_dir "$DATA_ROOT/$scene/language_features_multiscale" \
      --provisional_memory_dir "$A33_ROOT/$scene/equal_four_token_memory" \
      --output_dir "$output" --seed "$SEED" --topk 45 \
      --scaffold_method hdbscan_pca --scaffold_pca_dim 50 \
      --hdbscan_min_cluster_size 10 --hdbscan_min_samples 10 \
      --hdbscan_noise_policy "$HDBSCAN_NOISE_POLICY" \
      --hdbscan_cluster_selection_method "$HDBSCAN_CLUSTER_SELECTION_METHOD" \
      --mask_iou_threshold 0.8 --projection_chunk_pixels 8192 \
      --levels 0 --diagnostic_only \
      > "$LOG_DIR/${scene}_hdbscan_diagnostic.log" 2>&1
  fi
  date +%FT%T > "$RUN_ROOT/$scene/SCENE_COMPLETE"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
  "$GPU_GUARD" \
  "$A40_ROOT/ramen/post_aggregation_filter/manifest.json" \
  "$A42_ROOT/figurines/post_aggregation_filter/manifest.json" \
  "$A42_ROOT/waldo_kitchen/post_aggregation_filter/manifest.json"; do
  [[ -e "$required" ]] || { echo "Missing A43 input: $required" >&2; exit 2; }
done

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#scenes[@]}" -eq "${#gpus[@]}" ]] || {
  echo "SCENES and GPU_LIST must have equal lengths" >&2
  exit 2
}
for scene in "${scenes[@]}"; do
  for required in \
    "$A33_ROOT/$scene/equal_four_token_memory/manifest.json" \
    "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    "$DATA_ROOT/$scene/language_features_multiscale"; do
    [[ -e "$required" ]] || { echo "Missing A43 scene input: $required" >&2; exit 2; }
  done
done

pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  "$PYTHON_BIN" "$GPU_GUARD" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 21600 --poll-interval 5 -- \
    bash "$0" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A40_ROOT" "$A42_ROOT" "$SEED" "$HDBSCAN_NOISE_POLICY" "$HDBSCAN_CLUSTER_SELECTION_METHOD" "${scenes[@]}" <<'PY'
import json
import math
import os
import sys

root, a40_root, a42_root, raw_seed, noise_policy, selection_method, *scenes = sys.argv[1:]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def retention(manifest):
    diag = manifest["levels"][0]["filter_diagnostics"]
    views = manifest["views"]
    split = []
    for parity in (0, 1):
        selected = [view["levels"][0] for view in views if view["view_index"] % 2 == parity]
        kept = sum(item["kept_masks"] for item in selected)
        total = sum(item["input_masks"] for item in selected)
        split.append(kept / max(1, total))
    return {
        "kept_mask_fraction": diag["kept_mask_fraction"],
        "kept_pixel_fraction": diag["kept_pixel_fraction"],
        "mask_iou_quantiles": diag["mask_iou_quantiles"],
        "even_mask_retention": split[0],
        "odd_mask_retention": split[1],
        "even_odd_gap": abs(split[0] - split[1]),
    }


def coefficient_of_variation(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / max(mean, 1e-12)


kmeans_paths = {
    "ramen": os.path.join(a40_root, "ramen", "post_aggregation_filter", "manifest.json"),
    "figurines": os.path.join(a42_root, "figurines", "post_aggregation_filter", "manifest.json"),
    "waldo_kitchen": os.path.join(a42_root, "waldo_kitchen", "post_aggregation_filter", "manifest.json"),
}
summary = {
    "method": "A43 label-free PCA50+HDBSCAN L0 scaffold diagnostic",
    "seed": int(raw_seed),
    "leakage_control": "training cameras and SAM masks only; no text query, OVS labels, or evaluation metrics",
    "parameters": {
        "pca_components": 50,
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 10,
        "hdbscan_noise_policy": noise_policy,
        "hdbscan_cluster_selection_method": selection_method,
        "mask_iou_threshold": 0.8,
        "topk": 45,
    },
    "scenes": {},
}
for scene in scenes:
    adaptive = load(os.path.join(root, scene, "hdbscan_l0_diagnostic", "manifest.json"))
    fixed = load(kmeans_paths[scene])
    summary["scenes"][scene] = {
        "hdbscan_scaffold": adaptive["scaffold"]["diagnostics"],
        "hdbscan_filter": retention(adaptive),
        "fixed_kmeans_filter": retention(fixed),
    }

hdb_retention = [
    values["hdbscan_filter"]["kept_mask_fraction"]
    for values in summary["scenes"].values()
]
kmeans_retention = [
    values["fixed_kmeans_filter"]["kept_mask_fraction"]
    for values in summary["scenes"].values()
]
summary["cross_scene"] = {
    "hdbscan_mask_retention_cv": coefficient_of_variation(hdb_retention),
    "fixed_kmeans_mask_retention_cv": coefficient_of_variation(kmeans_retention),
    "hdbscan_pixel_retention_cv": coefficient_of_variation([
        values["hdbscan_filter"]["kept_pixel_fraction"]
        for values in summary["scenes"].values()
    ]),
    "fixed_kmeans_pixel_retention_cv": coefficient_of_variation([
        values["fixed_kmeans_filter"]["kept_pixel_fraction"]
        for values in summary["scenes"].values()
    ]),
}
checks = {
    "all_scenes_have_non_noise_clusters": all(
        values["hdbscan_scaffold"]["semantic_cluster_count"] >= 2
        for values in summary["scenes"].values()
    ),
    "all_gaussian_unassigned_fraction_le_0.5": all(
        values["hdbscan_scaffold"].get(
            "gaussian_unassigned_fraction",
            values["hdbscan_scaffold"]["gaussian_noise_fraction"],
        ) <= 0.5
        for values in summary["scenes"].values()
    ),
    "all_scenes_retain_masks": all(value > 0.0 for value in hdb_retention),
    "all_even_odd_gaps_le_0.02": all(
        values["hdbscan_filter"]["even_odd_gap"] <= 0.02
        for values in summary["scenes"].values()
    ),
    "cross_scene_retention_cv_not_worse": (
        summary["cross_scene"]["hdbscan_mask_retention_cv"]
        <= summary["cross_scene"]["fixed_kmeans_mask_retention_cv"]
    ),
    "all_pixel_retention_le_1.5x_fixed": all(
        values["hdbscan_filter"]["kept_pixel_fraction"]
        <= 1.5 * values["fixed_kmeans_filter"]["kept_pixel_fraction"]
        for values in summary["scenes"].values()
    ),
    "cross_scene_pixel_retention_cv_not_worse": (
        summary["cross_scene"]["hdbscan_pixel_retention_cv"]
        <= summary["cross_scene"]["fixed_kmeans_pixel_retention_cv"]
    ),
}
summary["pre_registered_gate"] = checks
summary["promote_to_codebook_training"] = all(checks.values())
with open(os.path.join(root, "three_scene_scaffold_diagnostics.json"), "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A43 density-adaptive three-scene diagnostic complete: $RUN_ROOT"
