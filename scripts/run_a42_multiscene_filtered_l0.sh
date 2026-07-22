#!/usr/bin/env bash
set -euo pipefail

# A42: expand A41's promoted filtered-L0 memory to Figurines and Waldo.
# Each worker builds fresh filtered consensuses/codebooks, then deploys only the
# new L0 alongside the scene's unchanged A33 L1--L3 peer tokens.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
GPU_GUARD=${GPU_GUARD:-$ROOT/scripts/gpu_guard.py}
SCENES=${SCENES:-"figurines waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"3 4"}
SEED=${SEED:-20260719}
A33_ROOT=${A33_ROOT:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
GEOMETRY_ROOT=${GEOMETRY_ROOT:-$ROOT/runs/3dgs}
DATA_ROOT=${DATA_ROOT:-$ROOT/drsplat_data/lerf_ovs}
LABEL_ROOT=${LABEL_ROOT:-$ROOT/drsplat_data/lerf_ovs/label}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/.cache}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$SOURCE_DIR:$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
export XDG_CACHE_HOME=$CACHE_ROOT/xdg TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface CUDA_CACHE_PATH=$CACHE_ROOT/nv
mkdir -p "$RUN_ROOT" "$LOG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local filter=$scene_root/post_aggregation_filter
  local fresh=$scene_root/fresh_four_token_memory
  local deployed=$scene_root/filtered_l0_memory
  local a33=$A33_ROOT/$scene/equal_four_token_memory
  mkdir -p "$scene_root"

  if [[ ! -f "$filter/FILTER_COMPLETE" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
      -s "$DATA_ROOT/$scene" -m "$scene_root/filter_model_context" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
      --feature_dir "$DATA_ROOT/$scene/language_features_multiscale" \
      --provisional_memory_dir "$a33" --output_dir "$filter" \
      --seed "$SEED" --topk 45 --scaffold_clusters 256 \
      --scaffold_train_samples 200000 --scaffold_iterations 25 \
      --mask_iou_threshold 0.8 --consensus_chunk_pixels 1024 \
      --projection_chunk_pixels 8192 --levels 0 --faiss_gpu \
      > "$LOG_DIR/${scene}_filter.log" 2>&1
  fi

  if [[ ! -f "$fresh/manifest.json" ]]; then
    local source_l1 source_l2 source_l3
    local -a source_consensuses
    readarray -t source_consensuses < <(
      "$PYTHON_BIN" - "$a33" <<'PY'
import json
import os
import sys

manifest = json.load(open(os.path.join(sys.argv[1], "manifest.json")))
for path in manifest["source"]["sam_l0_l3_consensus"]:
    print(path)
PY
    )
    source_l1=${source_consensuses[1]}
    source_l2=${source_consensuses[2]}
    source_l3=${source_consensuses[3]}
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
      --old_consensus "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" \
      --sam_l0_consensus "$filter/sam_l0_split2/consensus.pt" \
      --sam_l1_consensus "$source_l1" \
      --sam_l2_consensus "$source_l2" \
      --sam_l3_consensus "$source_l3" \
      --output_dir "$fresh" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
      --stability_floor 0.50 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --fallback_reliability 0.05 --faiss_gpu \
      > "$LOG_DIR/${scene}_codebooks.log" 2>&1
  fi

  if [[ ! -f "$deployed/manifest.json" ]]; then
    "$PYTHON_BIN" "$SOURCE_DIR/compose_independent_hierarchical_memory.py" \
      --level_memory_dirs "$fresh" "$a33" "$a33" "$a33" \
      --output_dir "$deployed" --seed "$SEED" \
      > "$LOG_DIR/${scene}_compose.log" 2>&1
  fi
  "$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
    --artifact_dir "$deployed" --required base sam_l0 sam_l1 sam_l2 sam_l3 \
    > "$LOG_DIR/${scene}_contract.log" 2>&1

  local output=$scene_root/eval_equal_query_max
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$DATA_ROOT/$scene" -m "$GEOMETRY_ROOT/$scene" \
      --geometry_checkpoint "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$deployed" --group_topk 4 \
      --group_readout equal_query_max --group_query_temperature 0.05 \
      --label_dir "$LABEL_ROOT/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds 0.55 --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi
  date +%FT%T > "$scene_root/SCENE_COMPLETE"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_scene "$2"
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_post_aggregation_filtered_consensus.py" \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/compose_independent_hierarchical_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
  "$GPU_GUARD" "$OPENCLIP_PRETRAINED"; do
  [[ -e "$required" ]] || { echo "Missing A42 source: $required" >&2; exit 2; }
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
    "$A33_ROOT/$scene/eval_equal_query_max/metrics.json" \
    "$A14_CONT_ROOT/$scene/old_split2/consensus.pt" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$GEOMETRY_ROOT/$scene/chkpnt30000.pth" \
    "$DATA_ROOT/$scene/language_features_multiscale" "$LABEL_ROOT/$scene"; do
    [[ -e "$required" ]] || { echo "Missing A42 input: $required" >&2; exit 2; }
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

"$PYTHON_BIN" - "$RUN_ROOT" "$A33_ROOT" "$SEED" "${scenes[@]}" <<'PY'
import hashlib
import json
import os
import sys

root, a33_root, raw_seed, *scenes = sys.argv[1:]
ramen_a41 = "/home/anlanfan/Dr-Splat/runs/a41_level_selective_filter_20260719_165158"


def row(path):
    payload = json.load(open(path))
    return next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - 0.55) < 1e-8
    )


summary = {
    "method": "A42 SFS-filtered L0 plus unfiltered A33 L1--L3",
    "seed": int(raw_seed),
    "selection_threshold": 0.55,
    "scenes": {},
}
ramen_a33 = row(os.path.join(a33_root, "ramen", "eval_equal_query_max", "metrics.json"))
ramen_new = row(os.path.join(ramen_a41, "ramen", "filtered_l0", "eval", "metrics.json"))
summary["scenes"]["ramen"] = {
    "a33": {key: ramen_a33[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
    "filtered_l0": {key: ramen_new[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
}
for scene in scenes:
    old = row(os.path.join(a33_root, scene, "eval_equal_query_max", "metrics.json"))
    new = row(os.path.join(root, scene, "eval_equal_query_max", "metrics.json"))
    filter_manifest = json.load(
        open(os.path.join(root, scene, "post_aggregation_filter", "manifest.json"))
    )
    fresh_dir = os.path.join(root, scene, "fresh_four_token_memory")
    old_dir = os.path.join(a33_root, scene, "equal_four_token_memory")
    fresh = json.load(open(os.path.join(fresh_dir, "manifest.json")))
    old_memory = json.load(open(os.path.join(old_dir, "manifest.json")))
    fresh_l0 = fresh["level_codebooks"][0]["codebook"]
    old_l0 = old_memory["level_codebooks"][0]["codebook"]
    summary["scenes"][scene] = {
        "a33": {key: old[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
        "filtered_l0": {key: new[key] for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")},
        "filter_l0": filter_manifest["levels"][0]["filter_diagnostics"],
        "fresh_l0_codebook": {
            "num_codes": fresh["level_codebooks"][0]["num_codes"],
            "sha256": hashlib.sha256(open(os.path.join(fresh_dir, fresh_l0), "rb").read()).hexdigest(),
            "different_from_a33": hashlib.sha256(open(os.path.join(fresh_dir, fresh_l0), "rb").read()).digest()
            != hashlib.sha256(open(os.path.join(old_dir, old_l0), "rb").read()).digest(),
        },
    }
for scene, values in summary["scenes"].items():
    values["delta"] = {
        key: values["filtered_l0"][key] - values["a33"][key]
        for key in values["a33"]
    }
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")
summary["a33_mean"] = {
    key: sum(values["a33"][key] for values in summary["scenes"].values()) / 3
    for key in metrics
}
summary["filtered_l0_mean"] = {
    key: sum(values["filtered_l0"][key] for values in summary["scenes"].values()) / 3
    for key in metrics
}
summary["mean_delta"] = {
    key: summary["filtered_l0_mean"][key] - summary["a33_mean"][key]
    for key in metrics
}
summary["decision"] = {
    "beats_a33_mean_miou": summary["mean_delta"]["mIoU"] > 0.0,
    "preserves_mean_strict_accuracy": summary["mean_delta"]["mAcc@0.5"] >= 0.0,
    "fresh_l0_codebooks": all(
        values.get("fresh_l0_codebook", {}).get("different_from_a33", True)
        for values in summary["scenes"].values()
    ),
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A42 filtered-L0 three-scene expansion complete: $RUN_ROOT"
