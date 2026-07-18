#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a20_fine_identity_codebook_20260716}
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

run_scene() {
  local scene=$1
  local scene_root=$A20_ROOT/$scene
  local artifact=$scene_root/fine_part_codebook_strict
  local output=$scene_root/eval_fine_part_strict
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_fine_part_shared_codebook.py \
      --part_artifact_dir "$A18_ROOT/$scene/interior/soft" \
      --fine_consensus "$scene_root/l1_signed_split2/consensus.pt" \
      --output_dir "$artifact" --device cuda \
      --stability_floor 0.5 --min_group_size 3 --max_group_size 16 \
      --min_reliability 0.6 --min_disagreement 0.10 \
      > "$LOG_DIR/${scene}_fine_part_strict_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine \
    > "$LOG_DIR/${scene}_vocabulary_contract_strict.log" 2>&1
  if [[ ! -f "$output/metrics.json" ]]; then
    mkdir -p "$output"
    "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
      -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
      --codebook_query_route query_positive \
      --group_hierarchy_dir "$artifact" --group_topk 2 \
      --group_readout hypothesis_blend --group_route_priority reliability_gain \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --evaluation_protocol drsplat_3d_selection \
      --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
      --output "$output" > "$LOG_DIR/${scene}_fine_part_strict_eval.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}; gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a20_fine_identity_strict.sh" --worker "$scene" \
    > "$LOG_DIR/worker_strict_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$A20_ROOT" "$A14_ROOT" "$A18_ROOT" "$SELECTION_THRESHOLD" "${scenes[@]}" <<'PY'
import json, os, sys
root, a14, a18, raw_threshold, *scenes = sys.argv[1:]
threshold = float(raw_threshold); names = ("mIoU", "mAcc@0.25", "mAcc@0.5")
def row(path):
    payload=json.load(open(path))
    value=next(x for x in payload["threshold_summary"] if abs(float(x["selection_threshold"])-threshold)<1e-8)
    return {name:float(value[name]) for name in names}
summary={"evaluation_protocol":"drsplat_3d_selection","selection_threshold":threshold,"occupancy_threshold":0.7,"scenes":{}}
for scene in scenes:
    rows={
        "a14":row(os.path.join(a14,scene,"eval","metrics.json")),
        "a18_soft":row(os.path.join(a18,scene,"eval_part_interior_soft","metrics.json")),
        "a20_broad":row(os.path.join(root,scene,"eval_fine_part","metrics.json")),
        "a20_strict":row(os.path.join(root,scene,"eval_fine_part_strict","metrics.json")),
    }
    manifest=json.load(open(os.path.join(root,scene,"fine_part_codebook_strict","manifest.json")))
    rows["fine_selection"]=manifest["fine_selection"]
    rows["vocabulary"]=manifest["modality_token_counts"]
    for reference in ("a14","a18_soft","a20_broad"):
        rows["delta_vs_"+reference]={name:rows["a20_strict"][name]-rows[reference][name] for name in names}
    summary["scenes"][scene]=rows
for method in ("a14","a18_soft","a20_broad","a20_strict"):
    summary[method+"_mean"]={name:sum(summary["scenes"][s][method][name] for s in scenes)/len(scenes) for name in names}
summary["selection"]={
    "best_a20_variant":max(("a20_broad","a20_strict"),key=lambda method:summary[method+"_mean"]["mIoU"]),
    "all_vocabulary_contracts_updated":True,
}
json.dump(summary,open(os.path.join(root,"strict_summary.json"),"w"),indent=2)
print(json.dumps(summary,indent=2))
PY

for scene in "${scenes[@]}"; do
  "$PYTHON_BIN" analyze_small_object_metrics.py \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --metrics \
      "A14=$A14_ROOT/$scene/eval/metrics.json" \
      "A18=$A18_ROOT/$scene/eval_part_interior_soft/metrics.json" \
      "A20b=$A20_ROOT/$scene/eval_fine_part/metrics.json" \
      "A20s=$A20_ROOT/$scene/eval_fine_part_strict/metrics.json" \
    --selection_threshold "$SELECTION_THRESHOLD" \
    --output "$A20_ROOT/$scene/small_object_analysis_strict.json" \
    > "$LOG_DIR/${scene}_small_object_analysis_strict.log" 2>&1
done
date +%FT%T > "$A20_ROOT/STRICT_COMPLETE"
