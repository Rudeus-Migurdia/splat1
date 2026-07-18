#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/a22_dual_code_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a22_dual_code_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
SELECTION_THRESHOLD=${SELECTION_THRESHOLD:-0.55}
BASELINE_THRESHOLD=${BASELINE_THRESHOLD:-0.50}

cd "$ROOT"; source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$RUN_ROOT" "$LOG_DIR"

run_scene() {
  local scene=$1 root=$RUN_ROOT/$1 artifact=$RUN_ROOT/$1/dual_codebook
  mkdir -p "$root"
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_dual_semantic_identity_codebook.py \
      --a20_artifact_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --a21_artifact_dir "$A21_ROOT/$scene/atom_codebook" \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --output_dir "$artifact" --target_occupancy 4 --iterations 10 \
      --device cuda > "$LOG_DIR/${scene}_dual_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" \
    --required base part fine competitor semantic_atom \
    > "$LOG_DIR/${scene}_contract.log" 2>&1

  local variant readout output
  for variant in agreement contrastive; do
    readout=dual_agreement; [[ "$variant" == contrastive ]] && readout=dual_contrastive
    output=$root/eval_$variant
    if [[ ! -f "$output/metrics.json" ]]; then
      mkdir -p "$output"
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
        --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
        --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
        --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
        --codebook_query_route query_positive --group_hierarchy_dir "$artifact" \
        --group_topk 2 --group_readout "$readout" \
        --group_route_priority reliability_gain --group_competitor_weight 1 \
        --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
        --evaluation_protocol drsplat_3d_selection \
        --selection_thresholds "$SELECTION_THRESHOLD" --occupancy_threshold 0.7 \
        --output "$output" > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    fi
  done

  if [[ ! -f "$root/query_consistency.json" ]]; then
    "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --a14_base_dir "$A14_ROOT/$scene/base_ids" \
      --a14_candidate_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --a21_group_dir "$A21_ROOT/$scene/atom_codebook" \
      --a22_group_dir "$artifact" --samples 100000 \
      --output "$root/query_consistency.json" \
      > "$LOG_DIR/${scene}_consistency.log" 2>&1
  fi
}

if [[ "${1:-}" == --worker ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"; read -r -a gpus <<< "$GPU_LIST"
for scene in "${scenes[@]}"; do
  for required in \
    "$A20_ROOT/$scene/fine_part_codebook/manifest.json" \
    "$A21_ROOT/$scene/atom_codebook/manifest.json" \
    "$A14_ROOT/$scene/base_ids/manifest.json" \
    "$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json"; do
    [[ -e "$required" ]] || { echo "Missing input: $required" >&2; exit 2; }
  done
done
pids=()
for i in "${!scenes[@]}"; do s=${scenes[$i]}; g=${gpus[$((i%${#gpus[@]}))]};
  "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$g" --hold-mb 384 \
    --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a22_dual_code.sh" --worker "$s" \
    > "$LOG_DIR/worker_${s}_gpu_${g}.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=$?; done; [[ $status -eq 0 ]] || exit "$status"

for scene in "${scenes[@]}"; do
  "$PYTHON_BIN" analyze_small_object_metrics.py \
    --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
    --metrics \
      "baseline@${BASELINE_THRESHOLD}=$ROOT/runs/paper_selection_20260714/$scene/baseline/metrics.json" \
      "A14@${SELECTION_THRESHOLD}=$A14_ROOT/$scene/eval/metrics.json" \
      "A20@${SELECTION_THRESHOLD}=$A20_ROOT/$scene/eval_fine_part/metrics.json" \
      "A22_agreement@${SELECTION_THRESHOLD}=$RUN_ROOT/$scene/eval_agreement/metrics.json" \
      "A22_contrastive@${SELECTION_THRESHOLD}=$RUN_ROOT/$scene/eval_contrastive/metrics.json" \
    --output "$RUN_ROOT/$scene/small_object_analysis.json" >/dev/null
done

"$PYTHON_BIN" - "$RUN_ROOT" "$A14_ROOT" "$A20_ROOT" "$A21_ROOT" "$ROOT/runs/paper_selection_20260714" "$SELECTION_THRESHOLD" "$BASELINE_THRESHOLD" "${scenes[@]}" <<'PY'
import json,os,sys
root,a14,a20,a21,base_root,raw_t,raw_bt,*scenes=sys.argv[1:]
t=float(raw_t); bt=float(raw_bt); names=("mIoU","mAcc@0.25","mAcc@0.5")
def row(path,threshold):
 d=json.load(open(path)); x=next(v for v in d["threshold_summary"] if abs(v["selection_threshold"]-threshold)<1e-8); return {k:float(x[k]) for k in names}
d={"evaluation_protocol":"drsplat_3d_selection","selection_threshold":t,"baseline_threshold":bt,"occupancy_threshold":.7,"scenes":{}}
for s in scenes:
 consistency=json.load(open(os.path.join(root,s,"query_consistency.json")))["representations"]
 manifest=json.load(open(os.path.join(root,s,"dual_codebook","manifest.json")))
 small=json.load(open(os.path.join(root,s,"small_object_analysis.json")))
 d["scenes"][s]={
  "drsplat_pq_baseline":row(os.path.join(base_root,s,"baseline","metrics.json"),bt),
  "a14":row(os.path.join(a14,s,"eval","metrics.json"),t),
  "a20":row(os.path.join(a20,s,"eval_fine_part","metrics.json"),t),
  "a21_atom":row(os.path.join(a21,s,"eval_atom_only","metrics.json"),t),
  "a22_agreement":row(os.path.join(root,s,"eval_agreement","metrics.json"),t),
  "a22_contrastive":row(os.path.join(root,s,"eval_contrastive","metrics.json"),t),
  "consistency":consistency,"small_object":small["small_category_mean_iou"],
  "dual_code_training":manifest["dual_code_training"],"vocabulary":manifest["modality_token_counts"]}
methods=("drsplat_pq_baseline","a14","a20","a21_atom","a22_agreement","a22_contrastive")
for m in methods:
 d[m+"_mean"]={k:sum(d["scenes"][s][m][k] for s in scenes)/len(scenes) for k in names}
consistency_names={"drsplat_pq_baseline":"drsplat_pq_baseline","a14":"a14","a20":"a20","a21_atom":"a21_atom_only","a22_agreement":"a22_dual_agreement","a22_contrastive":"a22_dual_contrastive"}
for m,key in consistency_names.items():
 d[m+"_consistency_mean"]={k:sum(d["scenes"][s]["consistency"][key][k] for s in scenes)/len(scenes) for k in ("canonical_split_symmetric_kl","canonical_split_top1_flip_rate")}
small_names={"drsplat_pq_baseline":"baseline","a14":"A14","a20":"A20","a22_agreement":"A22_agreement","a22_contrastive":"A22_contrastive"}
d["small_object_mean"]={m:sum(d["scenes"][s]["small_object"][key] for s in scenes)/len(scenes) for m,key in small_names.items()}
d["selection"]={"best_a22":max(("a22_agreement","a22_contrastive"),key=lambda m:d[m+"_mean"]["mIoU"]),"all_vocabulary_contracts_updated":True}
json.dump(d,open(os.path.join(root,"three_scene_summary.json"),"w"),indent=2); print(json.dumps(d,indent=2))
PY
date +%FT%T > "$RUN_ROOT/PROBE_COMPLETE"
