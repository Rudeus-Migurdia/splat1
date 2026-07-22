#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}; RUN_ROOT=${RUN_ROOT:?}; LOG_DIR=${LOG_DIR:?}
SOURCE_DIR=${SOURCE_DIR:-$RUN_ROOT/source_snapshot}; PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
SEED=${SEED:-20260719}; GPU=${GPU:-1}
A52_RUN=${A52_RUN:-$ROOT/runs/a52_query_conditioned_spatial_posterior_20260721_213802}
A56_RUN=${A56_RUN:-$ROOT/runs/a56_cross_scene_mass_conserving_validation_20260721_223620}
A57_RUN=${A57_RUN:-/mnt/zju105100248/home/anlanfan/a57_seeded_group_completion_20260722_005108}
A58_RUN=${A58_RUN:-/mnt/zju105100248/home/anlanfan/a58_anisotropic_group_completion_20260722_012258}
A33_RUN=${A33_RUN:-$ROOT/runs/a33_equal_four_token_seed_replication_20260718_113948}
A31_RUN=${A31_RUN:-$ROOT/runs/a31_teatime_equal_four_token_validation_20260718_104609}
mkdir -p "$RUN_ROOT" "$LOG_DIR"

memory_dir() { case "$1" in ramen) echo "$A52_RUN/ramen/fresh_equal_four_token_memory";; figurines|waldo_kitchen) echo "$A33_RUN/$1/equal_four_token_memory";; teatime) echo "$A31_RUN/teatime/equal_four_token_memory";; esac; }
spatial_dir() { case "$1" in ramen) echo "$A52_RUN/ramen/query_conditioned_spatial_posterior";; figurines|teatime|waldo_kitchen) echo "$A56_RUN/$1/query_conditioned_spatial_posterior";; esac; }

for required in "$A57_RUN/PROBE_COMPLETE" "$A58_RUN/PROBE_COMPLETE" \
  "$SOURCE_DIR/eval_lerf_ovs_gaussian_codebook_miou.py" "$SOURCE_DIR/semantic_hypothesis_routing.py" \
  "$SOURCE_DIR/query_conditioned_spatial_posterior.py" \
  "$SOURCE_DIR/scripts/run_a59_seed_conditioned_anisotropic_worker.sh" "$ROOT/scripts/gpu_guard.py"; do
  [[ -e "$required" ]] || { echo "Missing A59 input: $required" >&2; exit 2; }
done

for scene in ramen figurines teatime waldo_kitchen; do
  shape=$A58_RUN/$scene/anisotropic_geometry
  [[ -f "$shape/manifest.json" ]] || { echo "Missing A58 shape: $scene" >&2; exit 2; }
  if [[ -f "$RUN_ROOT/$scene/eval_seed_conditioned_anisotropic/metrics.json" ]]; then continue; fi
  "$PYTHON_BIN" "$ROOT/scripts/gpu_guard.py" --gpu "$GPU" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 43200 --poll-interval 10 -- \
    env ROOT="$ROOT" SOURCE_DIR="$SOURCE_DIR" RUN_ROOT="$RUN_ROOT" LOG_DIR="$LOG_DIR" \
      SCENE="$scene" GPU="$GPU" MEMORY="$(memory_dir "$scene")" SPATIAL="$(spatial_dir "$scene")" \
      SHAPE="$shape" PYTHON_BIN="$PYTHON_BIN" SEED="$SEED" \
      bash "$SOURCE_DIR/scripts/run_a59_seed_conditioned_anisotropic_worker.sh" \
    > "$LOG_DIR/${scene}_driver.log" 2>&1
done

"$PYTHON_BIN" - "$RUN_ROOT" "$A57_RUN" "$A58_RUN" "$SEED" <<'PY'
import json, os, sys
root, a57_root, a58_root, seed = sys.argv[1:]
names = ("mIoU", "mAcc@0.25", "mAcc@0.5")
def row(path):
    x=json.load(open(path))["threshold_summary"][0]
    return {k:x[k] for k in (*names,"per_category")}
a57=json.load(open(os.path.join(a57_root,"summary.json")))
a58=json.load(open(os.path.join(a58_root,"summary.json")))
scenes={}
for scene in ("ramen","figurines","teatime","waldo_kitchen"):
    prior=a57["scenes"][scene]
    candidate=row(os.path.join(root,scene,"eval_seed_conditioned_anisotropic","metrics.json"))
    scenes[scene]={"reference":prior["reference"],"a55":prior["a55"],"a57":prior["a57"],"a58":a58["scenes"][scene]["a58"],"a59":candidate}
    for method in ("reference","a55","a57","a58"):
        scenes[scene]["delta_from_"+method]={k:candidate[k]-scenes[scene][method][k] for k in names}
summary={"experiment":"A59_query_seed_conditioned_anisotropic_propagation","fixed_seed":int(seed),"evaluation":"TopK45, selection=0.55, occupancy=0.7","codebook_contract":"read-only reuse of freshly retrained independent L0-L3 codebooks","scenes":scenes}
for method in ("reference","a55","a57","a58","a59"):
    for metric in names: summary[f"mean_{method}_{metric}"]=sum(x[method][metric] for x in scenes.values())/4
json.dump(summary,open(os.path.join(root,"summary.json"),"w"),indent=2)
open(os.path.join(root,"PROBE_COMPLETE"),"w").write("PROBE_COMPLETE\n")
print(json.dumps(summary,indent=2))
PY
echo "A59 seed-conditioned anisotropic propagation complete: $RUN_ROOT"
