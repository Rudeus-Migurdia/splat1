#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a21_view_invariant_atoms_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"
source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1
  local scene_root=$RUN_ROOT/$scene
  local artifact=$scene_root/atom_codebook
  mkdir -p "$scene_root"
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u train_view_invariant_semantic_atoms.py \
      --a20_artifact_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
      --output_dir "$artifact" --device cuda --neighbors 16 \
      --steps 100 --learning_rate 0.03 --contrastive_margin 0.10 \
      --push_weight 0.25 --anchor_weight 1.0 --faiss_gpu \
      > "$LOG_DIR/${scene}_atom_train.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine competitor \
    > "$LOG_DIR/${scene}_vocabulary_contract.log" 2>&1

  local variant readout output
  for variant in atom_only contrastive; do
    readout=hypothesis_blend
    [[ "$variant" == contrastive ]] && readout=contrastive_blend
    output=$scene_root/eval_$variant
    if [[ ! -f "$output/metrics.json" ]]; then
      mkdir -p "$output"
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
        --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
        --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
        --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
        --codebook_query_route query_positive \
        --group_hierarchy_dir "$artifact" --group_topk 2 \
        --group_readout "$readout" --group_competitor_weight 1 \
        --group_route_priority reliability_gain \
        --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
        --evaluation_protocol drsplat_3d_selection \
        --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
        --output "$output" > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    fi
  done

  if [[ ! -f "$scene_root/query_consistency_common.json" ]]; then
    "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --a14_base_dir "$A14_ROOT/$scene/base_ids" \
      --a14_candidate_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --a21_group_dir "$artifact" --samples 100000 \
      --output "$scene_root/query_consistency_common.json" \
      > "$LOG_DIR/${scene}_query_consistency.log" 2>&1
  fi
}

if [[ "${1:-}" == "--worker" ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"
read -r -a gpus <<< "$GPU_LIST"
for scene in "${scenes[@]}"; do
  for required in \
    "$A20_ROOT/$scene/fine_part_codebook/manifest.json" \
    "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
    "$A14_ROOT/$scene/base_ids/manifest.json" \
    "$A14_ROOT/$scene/pruned_candidate_ids/manifest.json" \
    "$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json" \
    "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"; do
    [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
  done
done

pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}; gpu=${gpus[$((index % ${#gpus[@]}))]}
  "$PYTHON_BIN" scripts/gpu_guard.py \
    --gpu "$gpu" --hold-mb 384 --max-used-mb 256 --max-utilization 5 \
    --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a21_view_invariant_atoms.sh" --worker "$scene" \
    > "$LOG_DIR/worker_${scene}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A20_ROOT" "$ROOT/runs/paper_selection_20260714" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json, os, sys
run_root,a14_root,a20_root,baseline_root,raw_threshold,raw_baseline,*scenes=sys.argv[1:]
threshold=float(raw_threshold); baseline_threshold=float(raw_baseline)
names=("mIoU","mAcc@0.25","mAcc@0.5")
def row(path,value):
    d=json.load(open(path)); x=next(r for r in d["threshold_summary"] if abs(float(r["selection_threshold"])-value)<1e-8)
    return {name:float(x[name]) for name in names}
summary={"evaluation_protocol":"drsplat_3d_selection","selection_threshold":threshold,"baseline_threshold":baseline_threshold,"occupancy_threshold":0.7,"scenes":{}}
for scene in scenes:
    rows={
        "drsplat_pq_baseline":row(os.path.join(baseline_root,scene,"baseline","metrics.json"),baseline_threshold),
        "a14":row(os.path.join(a14_root,scene,"eval","metrics.json"),threshold),
        "a20":row(os.path.join(a20_root,scene,"eval_fine_part","metrics.json"),threshold),
        "a21_atom_only":row(os.path.join(run_root,scene,"eval_atom_only","metrics.json"),threshold),
        "a21_contrastive":row(os.path.join(run_root,scene,"eval_contrastive","metrics.json"),threshold),
    }
    atom=json.load(open(os.path.join(run_root,scene,"atom_codebook","manifest.json")))
    rows["atom_training"]=atom["atom_training"]
    rows["vocabulary"]=atom["modality_token_counts"]
    rows["query_consistency"]=json.load(open(os.path.join(run_root,scene,"query_consistency_common.json")))["representations"]
    summary["scenes"][scene]=rows
for method in ("drsplat_pq_baseline","a14","a20","a21_atom_only","a21_contrastive"):
    summary[method+"_mean"]={name:sum(summary["scenes"][s][method][name] for s in scenes)/len(scenes) for name in names}
    summary[method+"_consistency_mean"]={
        key:sum(summary["scenes"][s]["query_consistency"][method][key] for s in scenes)/len(scenes)
        for key in ("canonical_split_symmetric_kl","canonical_split_top1_flip_rate")
    }
summary["selection"]={
    "best_a21_variant":max(("a21_atom_only","a21_contrastive"),key=lambda m:summary[m+"_mean"]["mIoU"]),
    "all_vocabulary_contracts_updated":True,
}
json.dump(summary,open(os.path.join(run_root,"three_scene_summary.json"),"w"),indent=2)
print(json.dumps(summary,indent=2))
PY
date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
