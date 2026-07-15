#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
SCENES=("$@")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=(figurines ramen teatime waldo_kitchen)
fi

for scene in "${SCENES[@]}"; do
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  labels="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  gs="$ROOT/runs/3dgs/$scene/chkpnt30000.pth"
  drs="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"

  echo "scene=$scene"
  [[ -d "$dataset/images" ]] && echo "  dataset_images=ok" || echo "  dataset_images=missing:$dataset/images"
  [[ -d "$dataset/sparse/0" ]] && echo "  colmap_sparse=ok" || echo "  colmap_sparse=missing:$dataset/sparse/0"
  [[ -d "$dataset/language_features" ]] && echo "  language_features=ok" || echo "  language_features=missing:$dataset/language_features"
  [[ -d "$labels" ]] && echo "  labels=ok" || echo "  labels=missing:$labels"
  [[ -f "$gs" ]] && echo "  3dgs_checkpoint=ok:$gs" || echo "  3dgs_checkpoint=missing:$gs"
  [[ -f "$drs" ]] && echo "  drsplat_checkpoint=ok:$drs" || echo "  drsplat_checkpoint=missing:$drs"
done
