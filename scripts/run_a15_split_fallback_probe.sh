#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
A15_ROOT=${A15_ROOT:-$ROOT/runs/a15_segment_view_importance_20260716}
E8_ROOT=${E8_ROOT:-$ROOT/runs/a6_novelty_joint32k_20260715}
RUN_ROOT=${RUN_ROOT:-$A15_ROOT/a15_3_split_fallback}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a15_split_fallback_20260716}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
mkdir -p "$RUN_ROOT" "$LOG_DIR"

e8_candidate_consensus() {
  local scene=$1
  if [[ "$scene" == "waldo_kitchen" ]]; then
    printf '%s\n' "$ROOT/runs/a6_semantic_residual_waldo_20260715/discrete_alpha050_k16384x2/consensus_alpha050.pt"
  else
    printf '%s\n' "$ROOT/runs/a6_responsibility_multiscene_20260715/$scene/consensus_alpha050.pt"
  fi
}

run_scene() {
  local scene=$1
  local output=$RUN_ROOT/$scene
  local fallback=$output/fused_reliability_fallback.pt
  mkdir -p "$output"
  if [[ ! -f "$fallback" ]]; then
    "$PYTHON_BIN" -u build_split_reliability_fallback.py \
      --baseline "$A14_ROOT/$scene/fused_w1p5_t005.pt" \
      --candidate "$A15_ROOT/$scene/a15_1_agreement/fused_w1p5_t005.pt" \
      --old_split_consensus "$A14_ROOT/$scene/old_split2/consensus.pt" \
      --l2_split_consensus "$A14_ROOT/$scene/l2_split2/consensus.pt" \
      --output "$fallback" \
      > "$LOG_DIR/${scene}_build.log" 2>&1
  fi

  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local direct=$output/eval_direct
  if [[ ! -f "$direct/metrics.json" ]]; then
    mkdir -p "$direct"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$fallback" --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 --output "$direct" \
      > "$LOG_DIR/${scene}_direct_eval.log" 2>&1
  fi

  local hybrid=$output/eval_e8_candidate
  if [[ ! -f "$hybrid/metrics.json" ]]; then
    mkdir -p "$hybrid"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$(e8_candidate_consensus "$scene")" \
      --consensus_blend_base "$fallback" \
      --consensus_candidate_weight 1 \
      --consensus_query_route query_positive \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 --output "$hybrid" \
      > "$LOG_DIR/${scene}_hybrid_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  for scene in "$@"; do
    run_scene "$scene"
  done
  exit 0
fi

read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
pids=()
for gpu_index in "${!gpus[@]}"; do
  worker_scenes=()
  for scene_index in "${!scenes[@]}"; do
    if (( scene_index % ${#gpus[@]} == gpu_index )); then
      worker_scenes+=("${scenes[$scene_index]}")
    fi
  done
  [[ "${#worker_scenes[@]}" -gt 0 ]] || continue
  gpu=${gpus[$gpu_index]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a15_split_fallback_probe.sh" \
      --worker "${worker_scenes[@]}" \
    > "$LOG_DIR/worker_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A15_ROOT" "$E8_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

root, a14_root, a15_root, e8_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)

def row_at(path):
    with open(path) as source:
        metrics = json.load(source)
    row = next(x for x in metrics["threshold_summary"] if abs(x["selection_threshold"] - threshold) < 1e-8)
    return {key: float(row[key]) for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")}

summary = {"selection_threshold": threshold, "scenes": {}}
for scene in scenes:
    rows = {
        "e8_3": row_at(os.path.join(e8_root, scene, "eval", "metrics.json")),
        "a14": row_at(os.path.join(a14_root, scene, "eval_a14_e8_candidate", "metrics.json")),
        "a15_1": row_at(os.path.join(a15_root, scene, "a15_1_agreement", "eval_e8_candidate", "metrics.json")),
        "a15_3": row_at(os.path.join(root, scene, "eval_e8_candidate", "metrics.json")),
    }
    rows["delta_vs_a14"] = {key: rows["a15_3"][key] - rows["a14"][key] for key in rows["a14"]}
    summary["scenes"][scene] = rows
for name in ("e8_3", "a14", "a15_1", "a15_3", "delta_vs_a14"):
    summary[name + "_mean"] = {
        key: sum(summary["scenes"][scene][name][key] for scene in scenes) / len(scenes)
        for key in ("mIoU", "mAcc@0.25", "mAcc@0.5")
    }
a14 = summary["a14_mean"]
a15 = summary["a15_3_mean"]
summary["passed"] = (
    a15["mIoU"] - a14["mIoU"] >= 0.0015
    and a15["mAcc@0.25"] >= a14["mAcc@0.25"]
    and a15["mAcc@0.5"] >= a14["mAcc@0.5"]
)
with open(os.path.join(root, "four_scene_probe.json"), "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A15.3 split-reliability fallback probe complete: $RUN_ROOT"
