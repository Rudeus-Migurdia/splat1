#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_signed_ownership_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a16_sparse_view_modes_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a16_sparse_view_modes_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
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

run_scene() {
  local scene=$1
  local dataset=$ROOT/drsplat_data/lerf_ovs/$scene
  local labels=$ROOT/drsplat_data/lerf_ovs/label/$scene
  local geometry=$ROOT/runs/3dgs/$scene/chkpnt30000.pth
  local base=$A14_ROOT/$scene/fused_w1p5_t005.pt
  local e8_candidate
  e8_candidate=$(e8_candidate_consensus "$scene")
  local scene_root=$RUN_ROOT/$scene
  local cache=$scene_root/l2_signed_discordant_view_cache
  local hypothesis=$scene_root/hypothesis
  mkdir -p "$scene_root"

  if [[ ! -f "$cache/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene signed view cache start"
    "$PYTHON_BIN" -u prepare_semantic_field.py \
      -s "$dataset" -m "$cache" \
      --geometry_checkpoint "$geometry" \
      --feature_dir "$dataset/language_features_multiscale" --feature_level 2 \
      --semantic_dim 512 --identity_codec \
      --max_pixels_per_view 0 --topk 45 --raw_contribution_weights \
      --signed_segment_ownership --compact_view_cache \
      --view_cache_reference "$base" \
      --view_cache_deviation_cosine_max 0.75 \
      > "$LOG_DIR/${scene}_cache.log" 2>&1
    echo "[$(date +%FT%T)] scene=$scene signed view cache done"
  fi

  if [[ ! -f "$hypothesis/manifest.json" ]]; then
    echo "[$(date +%FT%T)] scene=$scene reproducible mode start"
    "$PYTHON_BIN" -u build_split_reproducible_semantic_modes.py \
      --base_consensus "$base" \
      --view_cache_dir "$cache" \
      --output_dir "$hypothesis" \
      --device cuda \
      --deviation_cosine_max 0.75 \
      --observation_weight_power 0.5 \
      --min_views_per_split 3 \
      --support_saturation 6 \
      --min_compactness 0.88 \
      --min_cross_split_cosine 0.90 \
      --max_base_cosine 0.90 \
      > "$LOG_DIR/${scene}_build.log" 2>&1
    echo "[$(date +%FT%T)] scene=$scene reproducible mode done"
  fi

  local variant output
  local -a readout
  for variant in reliability_blend_margin switch_margin; do
    output=$scene_root/eval_$variant
    [[ -f "$output/metrics.json" ]] && continue
    readout=(--hypothesis_readout reliability_blend --hypothesis_query_margin)
    if [[ "$variant" == "switch_margin" ]]; then
      readout=(--hypothesis_readout switch --hypothesis_query_margin)
    fi
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$dataset" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$geometry" \
      --consensus_path "$e8_candidate" \
      --consensus_blend_base "$base" \
      --consensus_candidate_weight 1 \
      --consensus_query_route query_positive \
      --hypothesis_dir "$hypothesis" \
      "${readout[@]}" \
      --label_dir "$labels" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" \
      --occupancy_threshold 0.7 --output "$output" \
      > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
  done
  echo "[$(date +%FT%T)] scene=$scene complete"
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
[[ "${#gpus[@]}" -gt 0 ]] || { echo "GPU_LIST cannot be empty" >&2; exit 2; }

for scene in "${scenes[@]}"; do
  for required in \
    "$ROOT/drsplat_data/lerf_ovs/$scene/language_features_multiscale" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$A14_ROOT/$scene/fused_w1p5_t005.pt" \
    "$A14_ROOT/$scene/eval_a14_e8_candidate/metrics.json" \
    "$(e8_candidate_consensus "$scene")"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for scene_index in "${!scenes[@]}"; do
  scene=${scenes[$scene_index]}
  gpu=${gpus[$((scene_index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a16_sparse_view_modes_probe.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json
import os
import sys

run_root, a14_root, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold)
metrics = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row_at(path):
    with open(path) as source:
        payload = json.load(source)
    row = next(
        item for item in payload["threshold_summary"]
        if abs(float(item["selection_threshold"]) - threshold) < 1e-8
    )
    return {name: float(row[name]) for name in metrics}

summary = {
    "evaluation_protocol": "drsplat_3d_selection",
    "selection_threshold": threshold,
    "occupancy_threshold": 0.7,
    "scenes": {},
}
variants = ("reliability_blend_margin", "switch_margin")
for scene in scenes:
    baseline = row_at(
        os.path.join(a14_root, scene, "eval_a14_e8_candidate", "metrics.json")
    )
    with open(os.path.join(run_root, scene, "hypothesis", "manifest.json")) as source:
        hypothesis = json.load(source)
    scene_rows = {
        "a14_e8_candidate": baseline,
        "hypothesis": {
            "selected_fraction": hypothesis["selected_fraction"],
            "num_hypotheses": hypothesis["num_hypotheses"],
            "mean_reliability": hypothesis["mean_reliability"],
            "selected_diagnostics": hypothesis["selected_diagnostics"],
        },
    }
    for variant in variants:
        row = row_at(os.path.join(run_root, scene, "eval_" + variant, "metrics.json"))
        scene_rows[variant] = row
        scene_rows[variant + "_minus_a14"] = {
            name: row[name] - baseline[name] for name in metrics
        }
    summary["scenes"][scene] = scene_rows

for variant in variants:
    summary[variant + "_mean"] = {
        name: sum(summary["scenes"][scene][variant][name] for scene in scenes) / len(scenes)
        for name in metrics
    }
    summary[variant + "_minus_a14_mean"] = {
        name: sum(
            summary["scenes"][scene][variant + "_minus_a14"][name]
            for scene in scenes
        ) / len(scenes)
        for name in metrics
    }
    summary[variant + "_go_no_go"] = {
        "required_mean_mIoU_delta": 0.0015,
        "requires_no_per_scene_metric_regression": True,
        "passed": (
            summary[variant + "_minus_a14_mean"]["mIoU"] >= 0.0015
            and all(
                summary["scenes"][scene][variant + "_minus_a14"][name] >= -1e-8
                for scene in scenes
                for name in metrics
            )
        ),
    }

path = os.path.join(run_root, "three_scene_probe.json")
with open(path, "w") as output:
    json.dump(summary, output, indent=2)
print(json.dumps(summary, indent=2))
PY

date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
echo "A16 sparse view-mode probe complete: $RUN_ROOT"
