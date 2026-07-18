#!/usr/bin/env bash
set -euo pipefail

# A27.1: fixed four resident slots, seeded per-level K-means codebooks,
# Old/SAM agreement gating, and calibrated query-aware peer-token fusion.
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SOURCE_DIR=${SOURCE_DIR:-$ROOT}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_CONT_ROOT=${A14_CONT_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A14_DISC_ROOT=${A14_DISC_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
A24_ROOT=${A24_ROOT:-$ROOT/runs/a24_multiscale_micro_identity_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a27_seeded_four_slot_memory_20260717}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a27_seeded_four_slot_memory_20260717}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SEED=${SEED:-20260717}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}
QUERY_TEMPERATURE=${QUERY_TEMPERATURE:-0.10}
LEVEL_MARGIN_THRESHOLD=${LEVEL_MARGIN_THRESHOLD:-0.25}
LEVEL_MARGIN_TEMPERATURE=${LEVEL_MARGIN_TEMPERATURE:-0.10}

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

prepare_level_cache() {
  local scene=$1 level=$2 cache=$3
  [[ -f "$cache/manifest.json" && -f "$cache/consensus.pt" ]] && return
  "$PYTHON_BIN" -u "$ROOT/prepare_semantic_field.py" \
    -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$cache" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --feature_dir "$ROOT/drsplat_data/lerf_ovs/$scene/language_features_multiscale" \
    --feature_level "$level" --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --signed_segment_ownership --consensus_only --consensus_splits 2 \
    --seed "$SEED" \
    > "$LOG_DIR/${scene}_sam_l${level}_cache.log" 2>&1
}

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local old memory output
  old=$(old_consensus "$scene")
  memory=$scene_root/hierarchical_memory
  output=$scene_root/eval
  mkdir -p "$scene_root"

  local level
  for level in 0 1 2 3; do
    prepare_level_cache "$scene" "$level" "$scene_root/sam_l${level}_split2"
  done

  if [[ ! -f "$memory/manifest.json" ]]; then
    "$PYTHON_BIN" -u "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --old_consensus "$old" \
      --sam_l0_consensus "$scene_root/sam_l0_split2/consensus.pt" \
      --sam_l1_consensus "$scene_root/sam_l1_split2/consensus.pt" \
      --sam_l2_consensus "$scene_root/sam_l2_split2/consensus.pt" \
      --sam_l3_consensus "$scene_root/sam_l3_split2/consensus.pt" \
      --output_dir "$memory" --device cuda --seed "$SEED" --neighbors 8 \
      --semantic_thresholds 0.76 0.82 0.87 0.91 \
      --maximum_group_sizes 2048 512 128 32 \
      --minimum_group_sizes 16 8 4 2 \
      --codes_per_level 2048 4096 8192 16384 \
      --train_samples 200000 --kmeans_iterations 25 --assignment_chunk_size 8192 \
      --stability_floor 0.5 --minimum_reliability 0.25 \
      --source_agreement_floor 0.80 --source_margin 0.0 \
      --faiss_gpu > "$LOG_DIR/${scene}_memory_build.log" 2>&1
  fi
  "$PYTHON_BIN" "$SOURCE_DIR/validate_semantic_vocabulary_contract.py" \
    --artifact_dir "$memory" --required base sam_l0 sam_l1 sam_l2 sam_l3 \
    > "$LOG_DIR/${scene}_memory_contract.log" 2>&1

  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_DISC_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive --group_hierarchy_dir "$memory" \
      --group_topk 4 --group_readout calibrated_hierarchical_memory \
      --group_query_temperature "$QUERY_TEMPERATURE" \
      --group_level_margin_threshold "$LEVEL_MARGIN_THRESHOLD" \
      --group_level_margin_temperature "$LEVEL_MARGIN_TEMPERATURE" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_eval.log" 2>&1
  fi

  "$PYTHON_BIN" -u "$SOURCE_DIR/eval_gaussian_split_query_consistency.py" \
    --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
    --pq_index "$ROOT/ckpts/pq_index.faiss" \
    --a14_base_dir "$A14_DISC_ROOT/$scene/base_ids" \
    --a14_candidate_dir "$A14_DISC_ROOT/$scene/pruned_candidate_ids" \
    --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
    --a21_group_dir "$A21_ROOT/$scene/atom_codebook" \
    --a24_group_dir "$A24_ROOT/$scene/micro_codebook" \
    --a27_group_dir "$memory" --samples 100000 --seed "$SEED" \
    --a27_group_query_temperature "$QUERY_TEMPERATURE" \
    --a27_level_margin_threshold "$LEVEL_MARGIN_THRESHOLD" \
    --a27_level_margin_temperature "$LEVEL_MARGIN_TEMPERATURE" \
    --output "$scene_root/query_consistency.json" \
    > "$LOG_DIR/${scene}_consistency.log" 2>&1
}

if [[ "${1:-}" == --worker ]]; then
  shift
  run_scene "$1"
  exit 0
fi

for required in \
  "$SOURCE_DIR/build_seeded_hierarchical_resident_memory.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/eval_gaussian_split_query_consistency.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/validate_semantic_vocabulary_contract.py"; do
  [[ -f "$required" ]] || { echo "Missing isolated source: $required" >&2; exit 2; }
done
read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST cannot be empty" >&2; exit 2; }
for scene in "${scenes[@]}"; do
  for required in \
    "$(old_consensus "$scene")" \
    "$ROOT/drsplat_data/lerf_ovs/$scene/language_features_multiscale" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$A14_DISC_ROOT/$scene/base_ids/manifest.json" \
    "$A14_DISC_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$A20_ROOT/$scene/fine_part_codebook/manifest.json" \
    "$A21_ROOT/$scene/atom_codebook/manifest.json" \
    "$A24_ROOT/$scene/micro_codebook/manifest.json" \
    "$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

script_path=${BASH_SOURCE[0]}
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$script_path" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

for scene in "${scenes[@]}"; do
  "$PYTHON_BIN" "$SOURCE_DIR/analyze_small_object_metrics.py" \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --metrics \
      "baseline@${BASELINE_THRESHOLD}=$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json" \
      "A14@${SELECTION_THRESHOLD}=$A14_DISC_ROOT/$scene/eval/metrics.json" \
      "A20@${SELECTION_THRESHOLD}=$A20_ROOT/$scene/eval_fine_part/metrics.json" \
      "A24@${SELECTION_THRESHOLD}=$A24_ROOT/$scene/eval/metrics.json" \
      "A27@${SELECTION_THRESHOLD}=$RUN_ROOT/$scene/eval/metrics.json" \
    --output "$RUN_ROOT/$scene/small_object_analysis.json" \
    > "$LOG_DIR/${scene}_small_object.log" 2>&1
done

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_DISC_ROOT" "$A20_ROOT" "$A24_ROOT" \
  "$ROOT/runs/paper_selection_20260714" "$SELECTION_THRESHOLD" \
  "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, a14, a20, a24, baseline_root, raw_t, raw_bt, *scenes = sys.argv[1:]
threshold, baseline_threshold = float(raw_t), float(raw_bt)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path, selection_threshold):
    payload = json.load(open(path))
    item = next(
        value for value in payload["threshold_summary"]
        if abs(float(value["selection_threshold"]) - selection_threshold) < 1e-8
    )
    return {metric: float(item[metric]) for metric in metrics}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "seed": int(os.environ["PYTHONHASHSEED"]),
    "selection_threshold": threshold,
    "baseline_threshold": baseline_threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
for scene in scenes:
    memory = json.load(open(os.path.join(root, scene, "hierarchical_memory", "manifest.json")))
    consistency = json.load(open(os.path.join(root, scene, "query_consistency.json")))["representations"]
    small = json.load(open(os.path.join(root, scene, "small_object_analysis.json")))["small_category_mean_iou"]
    summary["scenes"][scene] = {
        "drsplat_pq_baseline": row(os.path.join(baseline_root, scene, "baseline", "metrics.json"), baseline_threshold),
        "a14": row(os.path.join(a14, scene, "eval", "metrics.json"), threshold),
        "a20": row(os.path.join(a20, scene, "eval_fine_part", "metrics.json"), threshold),
        "a24": row(os.path.join(a24, scene, "eval", "metrics.json"), threshold),
        "a27": row(os.path.join(root, scene, "eval", "metrics.json"), threshold),
        "consistency": consistency,
        "small_object": small,
        "hierarchy": memory["hierarchy"],
        "vocabulary": memory["modality_token_counts"],
        "resident_slot_fraction": memory["covered_fraction"],
        "usable_slot_fraction": memory["usable_slot_fraction"],
    }
for method in ("drsplat_pq_baseline", "a14", "a20", "a24", "a27"):
    summary[method + "_mean"] = {
        metric: sum(summary["scenes"][scene][method][metric] for scene in scenes) / len(scenes)
        for metric in metrics
    }
summary["a27_minus_a20_mean"] = {
    metric: summary["a27_mean"][metric] - summary["a20_mean"][metric]
    for metric in metrics
}
for method, key in {
    "drsplat_pq_baseline": "drsplat_pq_baseline",
    "a20": "a20",
    "a24": "a24_multiscale_micro",
    "a27": "a27_seeded_four_slot",
}.items():
    summary[method + "_consistency_mean"] = {
        metric: sum(summary["scenes"][scene]["consistency"][key][metric] for scene in scenes) / len(scenes)
        for metric in ("canonical_split_symmetric_kl", "canonical_split_top1_flip_rate")
    }
summary["small_object_mean"] = {
    method: sum(summary["scenes"][scene]["small_object"][key] for scene in scenes) / len(scenes)
    for method, key in {
        "drsplat_pq_baseline": "baseline",
        "a14": "A14",
        "a20": "A20",
        "a24": "A24",
        "a27": "A27",
    }.items()
}
summary["decision"] = {
    "beats_a20_mean_miou": summary["a27_minus_a20_mean"]["mIoU"] > 0.0,
    "small_object_at_least_a24": summary["small_object_mean"]["a27"] >= summary["small_object_mean"]["a24"],
    "ramen_macc_at_least_a20": (
        "ramen" not in scenes
        or summary["scenes"]["ramen"]["a27"]["mAcc@0.5"] >= summary["scenes"]["ramen"]["a20"]["mAcc@0.5"]
    ),
    "consistency_not_worse_than_a20": (
        summary["a27_consistency_mean"]["canonical_split_symmetric_kl"] <= summary["a20_consistency_mean"]["canonical_split_symmetric_kl"]
        and summary["a27_consistency_mean"]["canonical_split_top1_flip_rate"] <= summary["a20_consistency_mean"]["canonical_split_top1_flip_rate"]
    ),
    "all_gaussians_have_four_resident_ids": all(
        summary["scenes"][scene]["resident_slot_fraction"] == 1.0 for scene in scenes
    ),
}
with open(os.path.join(root, "three_scene_summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
