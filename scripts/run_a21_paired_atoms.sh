#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}
A20_ROOT=${A20_ROOT:-$ROOT/runs/a20_fine_identity_codebook_20260716}
A21_ROOT=${A21_ROOT:-$ROOT/runs/a21_view_invariant_atoms_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a21_view_invariant_atoms_20260716}
SCENES=${SCENES:-"figurines ramen waldo_kitchen"}
GPU_LIST=${GPU_LIST:-"1 2 3"}
cd "$ROOT"; source scripts/drsplat_env.sh
SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline

run_scene() {
  local scene=$1 root=$A21_ROOT/$1 artifact=$A21_ROOT/$1/paired_atom_codebook
  if [[ ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_paired_shared_semantic_atoms.py \
      --a21_artifact_dir "$root/atom_codebook" \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --output_dir "$artifact" --target_occupancy 1.5 --iterations 25 \
      --assignment_topk 4 --device cuda \
      > "$LOG_DIR/${scene}_paired_atom_build.log" 2>&1
  fi
  "$PYTHON_BIN" validate_semantic_vocabulary_contract.py \
    --artifact_dir "$artifact" --required base part fine competitor \
    > "$LOG_DIR/${scene}_paired_contract.log" 2>&1
  local variant readout output
  for variant in paired_atom paired_contrastive; do
    readout=hypothesis_blend; [[ "$variant" == paired_contrastive ]] && readout=contrastive_blend
    output=$root/eval_$variant
    if [[ ! -f "$output/metrics.json" ]]; then
      mkdir -p "$output"
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$ROOT/drsplat_data/lerf_ovs/$scene" -m "$ROOT/runs/3dgs/$scene" \
        --geometry_checkpoint "$ROOT/runs/3dgs/$scene/chkpnt30000.pth" \
        --codebook_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
        --query_route_base_codebook_dir "$A14_ROOT/$scene/base_ids" \
        --codebook_query_route query_positive --group_hierarchy_dir "$artifact" \
        --group_topk 2 --group_readout "$readout" --group_competitor_weight 1 \
        --group_route_priority reliability_gain \
        --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
        --evaluation_protocol drsplat_3d_selection --selection_thresholds 0.55 \
        --occupancy_threshold 0.7 --output "$output" \
        > "$LOG_DIR/${scene}_${variant}_eval.log" 2>&1
    fi
  done
  if [[ ! -f "$root/paired_query_consistency.json" ]]; then
    "$PYTHON_BIN" -u eval_gaussian_split_query_consistency.py \
      --fine_consensus "$A20_ROOT/$scene/l1_signed_split2/consensus.pt" \
      --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$scene" \
      --pq_checkpoint "$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth" \
      --pq_index "$ROOT/ckpts/pq_index.faiss" \
      --a14_base_dir "$A14_ROOT/$scene/base_ids" \
      --a14_candidate_dir "$A14_ROOT/$scene/pruned_candidate_ids" \
      --a20_group_dir "$A20_ROOT/$scene/fine_part_codebook" \
      --a21_group_dir "$artifact" --samples 100000 \
      --output "$root/paired_query_consistency.json" \
      > "$LOG_DIR/${scene}_paired_query_consistency.log" 2>&1
  fi
}
if [[ "${1:-}" == --worker ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a scenes <<< "$SCENES"; read -r -a gpus <<< "$GPU_LIST"; pids=()
for i in "${!scenes[@]}"; do s=${scenes[$i]}; g=${gpus[$((i%${#gpus[@]}))]};
  "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$g" --hold-mb 384 --max-used-mb 256 \
    --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- \
    bash "$ROOT/scripts/run_a21_paired_atoms.sh" --worker "$s" \
    > "$LOG_DIR/worker_paired_${s}_gpu_${g}.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=$?; done; [[ $status -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$A21_ROOT" "${scenes[@]}" <<'PY'
import json,os,sys
root,*scenes=sys.argv[1:]; names=("mIoU","mAcc@0.25","mAcc@0.5")
def row(path):
 d=json.load(open(path)); x=next(v for v in d["threshold_summary"] if abs(v["selection_threshold"]-.55)<1e-8); return {k:float(x[k]) for k in names}
d={"scenes":{}}
for s in scenes:
 c=json.load(open(os.path.join(root,s,"paired_query_consistency.json")))["representations"]
 m=json.load(open(os.path.join(root,s,"paired_atom_codebook","manifest.json")))
 d["scenes"][s]={"a21_unique":row(os.path.join(root,s,"eval_atom_only","metrics.json")),"paired_atom":row(os.path.join(root,s,"eval_paired_atom","metrics.json")),"paired_contrastive":row(os.path.join(root,s,"eval_paired_contrastive","metrics.json")),"paired_atom_training":m["paired_atom_training"],"consistency":{"baseline":c["drsplat_pq_baseline"],"a20":c["a20"],"paired_atom":c["a21_atom_only"],"paired_contrastive":c["a21_contrastive"]}}
for method in ("a21_unique","paired_atom","paired_contrastive"):
 d[method+"_mean"]={k:sum(d["scenes"][s][method][k] for s in scenes)/len(scenes) for k in names}
for method in ("baseline","a20","paired_atom","paired_contrastive"):
 d[method+"_consistency_mean"]={k:sum(d["scenes"][s]["consistency"][method][k] for s in scenes)/len(scenes) for k in ("canonical_split_symmetric_kl","canonical_split_top1_flip_rate")}
json.dump(d,open(os.path.join(root,"paired_summary.json"),"w"),indent=2); print(json.dumps(d,indent=2))
PY
date +%FT%T > "$A21_ROOT/PAIRED_COMPLETE"
