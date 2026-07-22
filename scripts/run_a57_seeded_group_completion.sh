#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A57 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A57 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A52_RUN=${A52_RUN:-$ROOT/runs/a52_query_conditioned_spatial_posterior_20260721_213802}
A56_RUN=${A56_RUN:-$ROOT/runs/a56_cross_scene_mass_conserving_validation_20260721_223620}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A31_RUN=${A31_RUN:-$ROOT/runs/a31_teatime_equal_four_token_validation_20260718_104609}

mkdir -p "$RUN_ROOT" "$LOG_DIR"

memory_dir() {
  case "$1" in
    ramen) printf '%s\n' "$A52_RUN/ramen/fresh_equal_four_token_memory" ;;
    figurines|waldo_kitchen) printf '%s\n' "$A33_RUN/$1/equal_four_token_memory" ;;
    teatime) printf '%s\n' "$A31_RUN/teatime/equal_four_token_memory" ;;
    *) return 2 ;;
  esac
}

spatial_dir() {
  case "$1" in
    ramen) printf '%s\n' "$A52_RUN/ramen/query_conditioned_spatial_posterior" ;;
    figurines|teatime|waldo_kitchen) printf '%s\n' "$A56_RUN/$1/query_conditioned_spatial_posterior" ;;
    *) return 2 ;;
  esac
}

for required in \
  "$A52_RUN/PROBE_COMPLETE" "$A56_RUN/PROBE_COMPLETE" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/scripts/run_a57_seeded_group_completion_worker.sh" \
  "$ROOT/scripts/gpu_guard.py"; do
  [[ -e "$required" ]] || { echo "Missing A57 input: $required" >&2; exit 2; }
done

for scene in ramen figurines teatime waldo_kitchen; do
  for required in \
    "$(memory_dir "$scene")/manifest.json" \
    "$(spatial_dir "$scene")/manifest.json" \
    "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
    "$ROOT/drsplat_data/lerf_ovs/label/$scene"; do
    [[ -e "$required" ]] || { echo "Missing $scene input: $required" >&2; exit 2; }
  done
done

SCENE_GPU_SPECS=${SCENE_GPU_SPECS:-"ramen:0 figurines:1 teatime:2 waldo_kitchen:3"}
SEQUENTIAL=${SEQUENTIAL:-0}
workers=()
for spec in $SCENE_GPU_SPECS; do
  scene=${spec%%:*}
  gpu=${spec##*:}
  if [[ -f "$RUN_ROOT/$scene/eval_seeded_group_completion/metrics.json" ]]; then
    echo "Reuse completed A57.1 scene: $scene"
    continue
  fi
  command=("$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$gpu" --hold-mb 384
    --max-used-mb 256 --max-utilization 5 --wait-timeout 43200 --poll-interval 10 -- \
    env ROOT="$ROOT" SOURCE_DIR="$SOURCE_DIR" RUN_ROOT="$RUN_ROOT" LOG_DIR="$LOG_DIR" \
      SCENE="$scene" GPU="$gpu" MEMORY="$(memory_dir "$scene")" \
      SPATIAL="$(spatial_dir "$scene")" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" \
      bash "$SOURCE_DIR/scripts/run_a57_seeded_group_completion_worker.sh")
  if [[ "$SEQUENTIAL" == 1 ]]; then
    "${command[@]}" > "$LOG_DIR/${scene}_driver.log" 2>&1
  else
    "${command[@]}" > "$LOG_DIR/${scene}_driver.log" 2>&1 &
    workers+=("$!")
  fi
done

status=0
for pid in "${workers[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "A57 worker failed" >&2; exit 1; }

"$PYTHON_BIN" - "$RUN_ROOT" "$A56_RUN" "$SEED" <<'PY'
import json, os, sys

root, a56, seed = sys.argv[1:]
metric_names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path):
    item = json.load(open(path))["threshold_summary"][0]
    return {key: item[key] for key in (*metric_names, "per_category")}

reference_summary = json.load(open(os.path.join(a56, "summary.json")))
scenes = {}
for scene in ("ramen", "figurines", "teatime", "waldo_kitchen"):
    prior = reference_summary["scenes"][scene]
    candidate = row(os.path.join(root, scene, "eval_seeded_group_completion", "metrics.json"))
    scenes[scene] = {
        "reference": prior["reference"],
        "a55": prior["a55"],
        "a57": candidate,
        "delta_from_reference": {
            key: candidate[key] - prior["reference"][key] for key in metric_names
        },
        "delta_from_a55": {
            key: candidate[key] - prior["a55"][key] for key in metric_names
        },
    }

summary = {
    "experiment": "A57.1_decision_aware_seeded_group_completion",
    "fixed_seed": int(seed),
    "evaluation": "TopK45, selection=0.55, occupancy=0.7",
    "codebook_contract": "read-only reuse of freshly retrained independent L0-L3 codebooks",
    "completion_contract": {
        "query_score_only": True,
        "resident_tokens_unchanged": True,
        "same_group_and_3d_connected_component": True,
        "exact_a55_fallback_without_certificate": True,
        "minimum_seed_score": 0.55,
        "maximum_expansion_ratio": 2.0,
    },
    "scenes": scenes,
    "mean_reference_miou": sum(v["reference"]["mIoU"] for v in scenes.values()) / 4,
    "mean_a55_miou": sum(v["a55"]["mIoU"] for v in scenes.values()) / 4,
    "mean_a57_miou": sum(v["a57"]["mIoU"] for v in scenes.values()) / 4,
    "mean_reference_macc50": sum(v["reference"]["mAcc@0.5"] for v in scenes.values()) / 4,
    "mean_a57_macc50": sum(v["a57"]["mAcc@0.5"] for v in scenes.values()) / 4,
}
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A57 Seeded Group Completion complete: $RUN_ROOT"
