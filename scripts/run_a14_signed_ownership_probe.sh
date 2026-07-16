#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a14_signed_ownership_20260716}
REFERENCE_ROOT=${REFERENCE_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
SCENES=${SCENES:-"figurines waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"0 1"}
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

a6_consensus() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/fused_w1p5_t005.pt"
  else
    printf '%s\n' "$ROOT/runs/multiscale_split_consistency/$scene/fused_w1p5_t005.pt"
  fi
}

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
  local source=$2
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local feature_dir feature_level
  case "$source" in
    old)
      feature_dir=$dataset/language_features
      feature_level=1
      ;;
    l2)
      feature_dir=$dataset/language_features_multiscale
      feature_level=2
      ;;
    *)
      echo "Unsupported semantic source: $source" >&2
      return 2
      ;;
  esac
  local output=$RUN_ROOT/$scene/${source}_split2
  if [[ -f "$output/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene source=$source reuse"
    return
  fi
  echo "[$(date +%FT%T)] scene=$scene source=$source prepare start"
  "$PYTHON_BIN" -u prepare_semantic_field.py \
    -s "$dataset" -m "$output" \
    --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    --feature_dir "$feature_dir" --feature_level "$feature_level" \
    --semantic_dim 512 --identity_codec \
    --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
    --consensus_only --consensus_chunk_pixels 1024 --consensus_splits 2 \
    --signed_segment_ownership \
    > "$LOG_DIR/${scene}_${source}_prepare.log" 2>&1
  echo "[$(date +%FT%T)] scene=$scene source=$source prepare done"
}

prepare_worker() {
  local worker=$1
  shift
  echo "[$(date +%FT%T)] worker=$worker tasks=$*"
  while [[ "$#" -gt 0 ]]; do
    prepare_source "$1" "$2"
    shift 2
  done
}

evaluate_scene() {
  local scene=$1
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local fused=$RUN_ROOT/$scene/fused_w1p5_t005.pt
  local a6
  a6=$(a6_consensus "$scene")
  local e8_candidate
  e8_candidate=$(e8_candidate_consensus "$scene")

  if [[ ! -f "$fused" ]]; then
    "$PYTHON_BIN" -u build_split_consistency_fusion.py \
      --base_consensus "$RUN_ROOT/$scene/old_split2/consensus.pt" \
      --aux_consensus "$RUN_ROOT/$scene/l2_split2/consensus.pt" \
      --output "$fused" --max_aux_weight 1.5 --temperature 0.05 \
      > "$LOG_DIR/${scene}_fusion.log" 2>&1
  fi

  local name consensus output
  local -a route_args
  for name in a6_continuous a14_signed a14_query_positive a14_e8_candidate; do
    route_args=()
    if [[ "$name" == "a6_continuous" ]]; then
      consensus=$a6
    elif [[ "$name" == "a14_e8_candidate" ]]; then
      consensus=$e8_candidate
    else
      consensus=$fused
    fi
    if [[ "$name" == "a14_query_positive" ]]; then
      route_args=(
        --consensus_blend_base "$a6"
        --consensus_candidate_weight 1
        --consensus_query_route query_positive
      )
    fi
    if [[ "$name" == "a14_e8_candidate" ]]; then
      route_args=(
        --consensus_blend_base "$fused"
        --consensus_candidate_weight 1
        --consensus_query_route query_positive
      )
    fi
    output=$RUN_ROOT/$scene/eval_$name
    if [[ ! -f "$output/metrics.json" ]]; then
      mkdir -p "$output"
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
        --geometry_checkpoint "$geometry" \
        --consensus_path "$consensus" --label_dir "$labels" \
        "${route_args[@]}" \
        --evaluation_protocol drsplat_3d_selection \
        --selection_thresholds "$SELECTION_THRESHOLD" \
        --occupancy_threshold 0.7 --output "$output" \
        > "$LOG_DIR/${scene}_${name}_eval.log" 2>&1
    fi
  done
}

if [[ "${1:-}" == "--prepare-worker" ]]; then
  shift
  prepare_worker "$@"
  exit 0
fi

if [[ "${1:-}" == "--evaluate-worker" ]]; then
  shift
  for scene in "$@"; do
    evaluate_scene "$scene"
  done
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST cannot be empty" >&2; exit 2; }

for scene in "${scenes[@]}"; do
  for required in \
    "$ROOT/drsplat_data/lerf_ovs/$scene/language_features" \
    "$ROOT/drsplat_data/lerf_ovs/$scene/language_features_multiscale" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$REFERENCE_ROOT/$scene/eval/metrics.json" \
    "$(a6_consensus "$scene")" \
    "$(e8_candidate_consensus "$scene")"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

prepare_pids=()
for gpu_index in "${!gpus[@]}"; do
  tasks=()
  task_index=0
  for scene in "${scenes[@]}"; do
    for source in old l2; do
      if (( task_index % ${#gpus[@]} == gpu_index )); then
        tasks+=("$scene" "$source")
      fi
      ((task_index += 1))
    done
  done
  [[ "${#tasks[@]}" -gt 0 ]] || continue
  gpu=${gpus[$gpu_index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 0 -- \
    bash "$ROOT/scripts/run_a14_signed_ownership_probe.sh" \
      --prepare-worker "gpu-$gpu" "${tasks[@]}" \
    > "$LOG_DIR/prepare_worker_gpu_${gpu}.log" 2>&1 &
  prepare_pids+=("$!")
done

status=0
for pid in "${prepare_pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

eval_pids=()
for scene_index in "${!scenes[@]}"; do
  gpu=${gpus[$((scene_index % ${#gpus[@]}))]}
  scene=${scenes[$scene_index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a14_signed_ownership_probe.sh" \
      --evaluate-worker "$scene" \
    > "$LOG_DIR/eval_worker_${scene}.log" 2>&1 &
  eval_pids+=("$!")
done

status=0
for pid in "${eval_pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$REFERENCE_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, reference_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)
canonical_scenes = {"figurines", "ramen", "teatime", "waldo_kitchen"}
paper_complete = set(scenes) == canonical_scenes
summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "paper_complete": paper_complete,
    "selection_threshold": threshold,
    "scenes": {},
}

def row_at(path):
    with open(path) as source:
        metrics = json.load(source)
    row = next(
        item for item in metrics["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {
        key: float(row[key])
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }

for scene in scenes:
    rows = {
        "e8_3": row_at(os.path.join(reference_root, scene, "eval", "metrics.json")),
        "a6_continuous": row_at(
            os.path.join(run_root, scene, "eval_a6_continuous", "metrics.json")
        ),
        "a14_signed": row_at(
            os.path.join(run_root, scene, "eval_a14_signed", "metrics.json")
        ),
        "a14_query_positive": row_at(
            os.path.join(run_root, scene, "eval_a14_query_positive", "metrics.json")
        ),
        "a14_e8_candidate": row_at(
            os.path.join(run_root, scene, "eval_a14_e8_candidate", "metrics.json")
        ),
    }
    for method in ("a14_signed", "a14_query_positive", "a14_e8_candidate"):
        for reference in ("e8_3", "a6_continuous"):
            rows[method + "_minus_" + reference] = {
                key: rows[method][key] - rows[reference][key]
                for key in rows[method]
            }
    summary["scenes"][scene] = rows

for name in (
    "e8_3",
    "a6_continuous",
    "a14_signed",
    "a14_query_positive",
    "a14_e8_candidate",
    "a14_signed_minus_e8_3",
    "a14_signed_minus_a6_continuous",
    "a14_query_positive_minus_e8_3",
    "a14_query_positive_minus_a6_continuous",
    "a14_e8_candidate_minus_e8_3",
    "a14_e8_candidate_minus_a6_continuous",
):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }

delta = summary["a14_signed_minus_e8_3_mean"]
summary["go_no_go"] = {
    "required_mIoU_delta": 0.003,
    "requires_no_accuracy_regression": True,
    "passed": (
        delta["mIoU"] >= 0.003
        and delta["mAcc@0.25"] >= 0.0
        and delta["mAcc@0.5"] >= 0.0
    ),
}
hybrid_delta = summary["a14_e8_candidate_minus_e8_3_mean"]
summary["hybrid_go_no_go"] = {
    "required_mIoU_delta": 0.003,
    "requires_no_accuracy_regression": True,
    "passed": (
        hybrid_delta["mIoU"] >= 0.003
        and hybrid_delta["mAcc@0.25"] >= 0.0
        and hybrid_delta["mAcc@0.5"] >= 0.0
    ),
}
path = os.path.join(
    run_root,
    "four_scene_probe.json" if paper_complete else "two_scene_probe.json",
)
with open(path, "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A14 signed-ownership probe complete: $RUN_ROOT"
