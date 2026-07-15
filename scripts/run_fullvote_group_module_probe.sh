#!/usr/bin/env bash
set -euo pipefail

# Evaluate an existing self-trained group hierarchy on the full-view discrete
# codebook.  This is intentionally evaluation-only: no PQ codes or labels are
# consumed by the semantic representation.
ROOT=${ROOT:-/mnt/zju105100171/home/anlanfan/Dr-Splat}
PYTHON_BIN=${PYTHON_BIN:-/home/anlanfan/.local/python3.9-171/bin/python3.9}
SCENE=${SCENE:-waldo_kitchen}
GROUP_HIERARCHY=${GROUP_HIERARCHY:-$ROOT/runs/group_view_importance/waldo_kitchen_stage_b_information_kl_kl0p02_top1/group_hierarchy}
RGR_ALPHA=${RGR_ALPHA:-0.5}
AGREEMENT_FLOOR=${AGREEMENT_FLOOR:--1.0}
AGREEMENT_POWER=${AGREEMENT_POWER:-1.0}
TAG=${TAG:-a${RGR_ALPHA}_agree${AGREEMENT_FLOOR}}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}

codebook="$ROOT/runs/baseline_voting_consensus/waldo_kitchen_topk45_fullraw/initial_codebook"
dataset="$ROOT/drsplat_data/lerf_ovs/$SCENE"
labels="$ROOT/drsplat_data/lerf_ovs/label/$SCENE"
geometry="$ROOT/runs/3dgs/$SCENE/chkpnt30000.pth"
run_root="$ROOT/runs/fullvote_group_modules/$SCENE"
output="$run_root/$TAG"

for path in "$codebook/manifest.json" "$GROUP_HIERARCHY/manifest.json" "$dataset" "$labels" "$geometry"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done

mkdir -p "$run_root"
"$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
  -s "$dataset" -m "$run_root" \
  --geometry_checkpoint "$geometry" \
  --codebook_dir "$codebook" \
  --label_dir "$labels" \
  --group_hierarchy_dir "$GROUP_HIERARCHY" \
  --group_topk 1 \
  --group_aggregation weighted \
  --rgr_alpha "$RGR_ALPHA" \
  --rgr_mode positive \
  --group_feature_agreement_floor "$AGREEMENT_FLOOR" \
  --group_feature_agreement_power "$AGREEMENT_POWER" \
  --score_calibration category_percentile \
  --calibration_low 1 --calibration_high 99 \
  --thresholds $THRESHOLDS \
  --output "$output"

echo "full-view group module probe complete: $output"
