#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
E8_ROOT=${E8_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a15_segment_view_importance_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a15_segment_view_importance_20260716}
SCENES=${SCENES:-"figurines waldo_kitchen"}
VARIANTS=${VARIANTS:-"a15_1_agreement a15_2_information"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$RUN_ROOT" "$LOG_DIR"

e8_candidate_consensus() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt"
  else
    printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$scene/consensus_alpha050.pt"
  fi
}

prepare_source() {
  local scene=$1
  local variant=$2
  local source=$3
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local feature_dir feature_level reference information_weight
  case "$source" in
    old)
      feature_dir=$dataset/language_features
      feature_level=1
      reference=$A14_ROOT/$scene/old_split2/consensus.pt
      ;;
    l2)
      feature_dir=$dataset/language_features_multiscale
      feature_level=2
      reference=$A14_ROOT/$scene/l2_split2/consensus.pt
      ;;
    *)
      echo "Unsupported source: $source" >&2
      return 2
      ;;
  esac
  case "$variant" in
    a15_1_agreement) information_weight=0 ;;
    a15_2_information) information_weight=1 ;;
    *) echo "Unsupported variant: $variant" >&2; return 2 ;;
  esac
  local output=$RUN_ROOT/$scene/$variant/${source}_split2
  if [[ -f "$output/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene variant=$variant source=$source reuse"
    return
  fi
  echo "[$(date +%FT%T)] scene=$scene variant=$variant source=$source start"
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$output" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --feature_dir "$feature_dir" --feature_level "$feature_level" \
    --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --consensus_only --consensus_chunk_pixels 1024 --consensus_splits 2 \
    --signed_segment_ownership \
    --segment_view_importance_reference "$reference" \
    --segment_importance_temperature 1 \
    --segment_importance_max_kl 0.02 \
    --segment_importance_ratio_clip 5 \
    --segment_information_weight "$information_weight" \
    > "$LOG_DIR/${scene}_${variant}_${source}_prepare.log" 2>&1
  echo "[$(date +%FT%T)] scene=$scene variant=$variant source=$source done"
}

prepare_worker() {
  local worker=$1
  shift
  echo "[$(date +%FT%T)] worker=$worker tasks=$*"
  while [[ "$#" -gt 0 ]]; do
    prepare_source "$1" "$2" "$3"
    shift 3
  done
}

evaluate_variant() {
  local scene=$1
  local variant=$2
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local variant_root=$RUN_ROOT/$scene/$variant
  local fused=$variant_root/fused_w1p5_t005.pt
  local e8_candidate
  e8_candidate=$(e8_candidate_consensus "$scene")
  if [[ ! -f "$fused" ]]; then
    "$PYTHON_BIN" -u build_split_consistency_fusion.py \
      --base_consensus "$variant_root/old_split2/consensus.pt" \
      --aux_consensus "$variant_root/l2_split2/consensus.pt" \
      --output "$fused" --max_aux_weight 1.5 --temperature 0.05 \
      > "$LOG_DIR/${scene}_${variant}_fusion.log" 2>&1
  fi

  local output=$variant_root/eval_direct
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$fused" --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 --output "$output" \
      > "$LOG_DIR/${scene}_${variant}_direct_eval.log" 2>&1
  fi

  output=$variant_root/eval_e8_candidate
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$e8_candidate" \
      --consensus_blend_base "$fused" \
      --consensus_candidate_weight 1 \
      --consensus_query_route query_positive \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 --output "$output" \
      > "$LOG_DIR/${scene}_${variant}_hybrid_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--prepare-worker" ]]; then
  shift
  prepare_worker "$@"
  exit 0
fi

if [[ "${1:-}" == "--evaluate-worker" ]]; then
  shift
  for scene in "$@"; do
    read -r -a worker_variants <<< "$VARIANTS"
    for variant in "${worker_variants[@]}"; do
      evaluate_variant "$scene" "$variant"
    done
  done
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a variants <<< "$VARIANTS"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST cannot be empty" >&2; exit 2; }
[[ "${#variants[@]}" -gt 0 ]] || { echo "VARIANTS cannot be empty" >&2; exit 2; }

for scene in "${scenes[@]}"; do
  for required in \
    "$A14_ROOT/$scene/old_split2/consensus.pt" \
    "$A14_ROOT/$scene/l2_split2/consensus.pt" \
    "$A14_ROOT/$scene/eval_a14_e8_candidate/metrics.json" \
    "$E8_ROOT/$scene/eval/metrics.json" \
    "$(e8_candidate_consensus "$scene")" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for gpu_index in "${!gpus[@]}"; do
  tasks=()
  task_index=0
  for scene in "${scenes[@]}"; do
    for variant in "${variants[@]}"; do
      for source in old l2; do
        if (( task_index % ${#gpus[@]} == gpu_index )); then
          tasks+=("$scene" "$variant" "$source")
        fi
        ((task_index += 1))
      done
    done
  done
  [[ "${#tasks[@]}" -gt 0 ]] || continue
  gpu=${gpus[$gpu_index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a15_segment_view_importance_probe.sh" \
      --prepare-worker "gpu-$gpu" "${tasks[@]}" \
    > "$LOG_DIR/prepare_worker_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

pids=()
for scene_index in "${!scenes[@]}"; do
  scene=${scenes[$scene_index]}
  gpu=${gpus[$((scene_index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a15_segment_view_importance_probe.sh" \
      --evaluate-worker "$scene" \
    > "$LOG_DIR/eval_worker_${scene}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$E8_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, a14_root, e8_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)

def row_at(path):
    with open(path) as source:
        metrics = json.load(source)
    row = next(
        item for item in metrics["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {key: float(row[key]) for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "registered_gate": {
        "minimum_mIoU_delta_vs_a14_hybrid": 0.0015,
        "minimum_mIoU_delta_vs_e8_3": 0.003,
        "requires_no_accuracy_regression_vs_a14_hybrid": True,
    },
    "scenes": {},
}
variants = tuple(os.environ["VARIANTS"].split())
for scene in scenes:
    rows = {
        "e8_3": row_at(os.path.join(e8_root, scene, "eval", "metrics.json")),
        "a14_hybrid": row_at(
            os.path.join(a14_root, scene, "eval_a14_e8_candidate", "metrics.json")
        ),
    }
    for variant in variants:
        rows[variant + "_direct"] = row_at(
            os.path.join(run_root, scene, variant, "eval_direct", "metrics.json")
        )
        rows[variant + "_hybrid"] = row_at(
            os.path.join(run_root, scene, variant, "eval_e8_candidate", "metrics.json")
        )
        rows[variant + "_delta_vs_a14"] = {
            key: rows[variant + "_hybrid"][key] - rows["a14_hybrid"][key]
            for key in rows["a14_hybrid"]
        }
    summary["scenes"][scene] = rows

mean_names = ["e8_3", "a14_hybrid"]
for variant in variants:
    mean_names.extend(
        (variant + "_direct", variant + "_hybrid", variant + "_delta_vs_a14")
    )
for name in mean_names:
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }

best = max(variants, key=lambda name: summary[name + "_hybrid_mean"]["mIoU"])
best_row = summary[best + "_hybrid_mean"]
a14 = summary["a14_hybrid_mean"]
e8 = summary["e8_3_mean"]
summary["selection"] = {
    "best_variant": best,
    "passed": (
        best_row["mIoU"] - a14["mIoU"] >= 0.0015
        and best_row["mIoU"] - e8["mIoU"] >= 0.003
        and best_row["mAcc@0.25"] >= a14["mAcc@0.25"]
        and best_row["mAcc@0.5"] >= a14["mAcc@0.5"]
    ),
}
output_name = "four_scene_probe.json" if len(scenes) == 4 else "two_scene_probe.json"
with open(os.path.join(run_root, output_name), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A15 segment-view importance probe complete: $RUN_ROOT"
