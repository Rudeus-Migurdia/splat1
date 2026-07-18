#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/anlanfan/Dr-Splat}; PYTHON_BIN=${PYTHON_BIN:-$ROOT/.venv/bin/python}
A14_ROOT=${A14_ROOT:-$ROOT/runs/a14_e8_joint32k_20260716}; A18_ROOT=${A18_ROOT:-$ROOT/runs/a18_hierarchical_group_codebook_20260716}
LOG_DIR=${LOG_DIR:-$ROOT/logs/a18_hierarchical_group_codebook_20260716}; SCENES=${SCENES:-"figurines ramen waldo_kitchen"}; GPU_LIST=${GPU_LIST:-"1 2 3"}
cd "$ROOT"; source scripts/drsplat_env.sh; SITE=$ROOT/.venv/lib/python3.9/site-packages
export PATH="$(dirname "$PYTHON_BIN"):$PATH"; export PYTHONPATH="$ROOT:$SITE:$SITE/setuptools/_vendor:${PYTHONPATH:-}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}; export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
run_scene() {
  local s=$1 art=$A18_ROOT/$1/interior/power2 out=$A18_ROOT/$1/eval_part_interior_power2
  if [[ ! -f "$art/manifest.json" ]]; then "$PYTHON_BIN" build_group_membership_power.py --source_artifact "$A18_ROOT/$s/exact_part_extension" --support "$A18_ROOT/$s/interior/part_interior_support.npy" --power 2 --output_dir "$art" > "$LOG_DIR/${s}_interior_power2_build.log" 2>&1; fi
  if [[ ! -f "$out/metrics.json" ]]; then mkdir -p "$out"; "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py -s "$ROOT/drsplat_data/lerf_ovs/$s" -m "$ROOT/runs/3dgs/$s" --geometry_checkpoint "$ROOT/runs/3dgs/$s/chkpnt30000.pth" --codebook_dir "$A14_ROOT/$s/pruned_candidate_ids" --query_route_base_codebook_dir "$A14_ROOT/$s/base_ids" --codebook_query_route query_positive --group_hierarchy_dir "$art" --group_topk 1 --group_readout hypothesis_blend --group_route_priority reliability_gain --label_dir "$ROOT/drsplat_data/lerf_ovs/label/$s" --evaluation_protocol drsplat_3d_selection --selection_thresholds 0.55 --occupancy_threshold 0.7 --output "$out" > "$LOG_DIR/${s}_interior_power2_eval.log" 2>&1; fi
}
if [[ "${1:-}" == "--worker" ]]; then shift; run_scene "$1"; exit 0; fi
read -r -a ss <<< "$SCENES"; read -r -a gs <<< "$GPU_LIST"; ps=(); for i in "${!ss[@]}"; do s=${ss[$i]}; g=${gs[$((i%${#gs[@]}))]}; "$PYTHON_BIN" scripts/gpu_guard.py --gpu "$g" --hold-mb 384 --max-used-mb 256 --max-utilization 5 --wait-timeout 120 --poll-interval 5 -- bash "$ROOT/scripts/run_a18_part_interior_power2.sh" --worker "$s" > "$LOG_DIR/worker_power2_${s}_gpu_${g}.log" 2>&1 & ps+=("$!"); done; st=0; for p in "${ps[@]}"; do wait "$p" || st=$?; done; exit "$st"
