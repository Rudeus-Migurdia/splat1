#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must name an isolated A58 run directory}
LOG_DIR=${LOG_DIR:?LOG_DIR must name an isolated A58 log directory}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}
A52_RUN=${A52_RUN:-$ROOT/runs/a52_query_conditioned_spatial_posterior_20260721_213802}
A56_RUN=${A56_RUN:-$ROOT/runs/a56_cross_scene_mass_conserving_validation_20260721_223620}
A57_RUN=${A57_RUN:-/mnt/zju105100248/home/anlanfan/a57_seeded_group_completion_20260722_005108}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A31_RUN=${A31_RUN:-$ROOT/runs/a31_teatime_equal_four_token_validation_20260718_104609}
GPU=${GPU:-1}

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
  "$A52_RUN/PROBE_COMPLETE" "$A56_RUN/PROBE_COMPLETE" "$A57_RUN/PROBE_COMPLETE" \
  "$SOURCE_DIR/build_group_anisotropic_geometry.py" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" \
  "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/scripts/run_a58_anisotropic_group_completion_worker.sh" \
  "$ROOT/scripts/gpu_guard.py"; do
  [[ -e "$required" ]] || { echo "Missing A58 input: $required" >&2; exit 2; }
done

for scene in ramen figurines teatime waldo_kitchen; do
  shape=$RUN_ROOT/$scene/anisotropic_geometry
  if [[ ! -f "$shape/manifest.json" ]]; then
    mkdir -p "$RUN_ROOT/$scene"
    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -u \
      "$SOURCE_DIR/build_group_anisotropic_geometry.py" \
      --spatial_posterior_dir "$(spatial_dir "$scene")" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --output_dir "$shape" --minimum_membership 0.05 --seed "$SEED" \
      > "$LOG_DIR/${scene}_shape_build.log" 2>&1
  fi
done

for scene in ramen figurines teatime waldo_kitchen; do
  if [[ -f "$RUN_ROOT/$scene/eval_anisotropic_group_completion/metrics.json" ]]; then
    echo "Reuse completed A58 scene: $scene"
    continue
  fi
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$GPU" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 43200 --poll-interval 10 -- \
    env ROOT="$ROOT" SOURCE_DIR="$SOURCE_DIR" RUN_ROOT="$RUN_ROOT" LOG_DIR="$LOG_DIR" \
      SCENE="$scene" GPU="$GPU" MEMORY="$(memory_dir "$scene")" \
      SPATIAL="$(spatial_dir "$scene")" SHAPE="$RUN_ROOT/$scene/anisotropic_geometry" \
      PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" \
      bash "$SOURCE_DIR/scripts/run_a58_anisotropic_group_completion_worker.sh" \
    > "$LOG_DIR/${scene}_driver.log" 2>&1
done

"$PYTHON_BIN" - "$RUN_ROOT" "$A57_RUN" "$SEED" <<'PY'
import json, os, sys

root, a57_root, seed = sys.argv[1:]
names = ("mIoU", "mAcc@0.25", "mAcc@0.5")

def row(path):
    item = json.load(open(path))["threshold_summary"][0]
    return {key: item[key] for key in (*names, "per_category")}

a57_summary = json.load(open(os.path.join(a57_root, "summary.json")))
scenes = {}
for scene in ("ramen", "figurines", "teatime", "waldo_kitchen"):
    prior = a57_summary["scenes"][scene]
    candidate = row(os.path.join(root, scene, "eval_anisotropic_group_completion", "metrics.json"))
    scenes[scene] = {
        "reference": prior["reference"],
        "a55": prior["a55"],
        "a57": prior["a57"],
        "a58": candidate,
        "delta_from_reference": {key: candidate[key] - prior["reference"][key] for key in names},
        "delta_from_a55": {key: candidate[key] - prior["a55"][key] for key in names},
        "delta_from_a57": {key: candidate[key] - prior["a57"][key] for key in names},
    }

summary = {
    "experiment": "A58_anisotropic_group_completion",
    "fixed_seed": int(seed),
    "evaluation": "TopK45, selection=0.55, occupancy=0.7",
    "codebook_contract": "read-only reuse of freshly retrained independent L0-L3 codebooks",
    "shape_contract": {
        "query_independent_weighted_group_covariance": True,
        "directional_atom_conductance": True,
        "shape_adaptive_budget_and_semantic_radius": True,
        "evaluation_queries_or_labels_used": False,
    },
    "scenes": scenes,
}
for method in ("reference", "a55", "a57", "a58"):
    for metric in names:
        summary[f"mean_{method}_{metric}"] = sum(
            item[method][metric] for item in scenes.values()
        ) / len(scenes)
with open(os.path.join(root, "summary.json"), "w") as output:
    json.dump(summary, output, indent=2)
with open(os.path.join(root, "PROBE_COMPLETE"), "w") as output:
    output.write("PROBE_COMPLETE\n")
print(json.dumps(summary, indent=2))
PY

echo "A58 anisotropic Group completion complete: $RUN_ROOT"
