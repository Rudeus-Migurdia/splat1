#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/anlanfan/Dr-Splat}
VENV_PATH=${VENV_PATH:-$ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-$VENV_PATH/bin/python}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}
CODES_PER_LEVEL=${CODES_PER_LEVEL:-"8192 8192"}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-262144}
KMEANS_ITERATIONS=${KMEANS_ITERATIONS:-25}
ASSIGNMENT_CHUNK=${ASSIGNMENT_CHUNK:-16384}
FAISS_GPU=${FAISS_GPU:-1}
RGR_ALPHA=${RGR_ALPHA:-0.75}
RGR_VARIANT_TAG=${RGR_VARIANT_TAG:-codebook_rgr}
GROUP_TOP_M=${GROUP_TOP_M:-3}
RUN_BUILD=${RUN_BUILD:-1}
RUN_EVAL=${RUN_EVAL:-1}
LOG_DIR=${LOG_DIR:-$ROOT/logs/large_gaussian_codebook_probe}
THRESHOLDS=${THRESHOLDS:-"0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"}

cd "$ROOT"
source scripts/drsplat_env.sh
export PATH="$VENV_PATH/bin:$PATH"
export OPENCLIP_PRETRAINED=${OPENCLIP_PRETRAINED:-$ROOT/ckpts/open_clip_vit_b16_laion2b_s34b_b88k.bin}
export WANDB_MODE=offline
mkdir -p "$LOG_DIR"

scene_group_dir() {
  local scene=$1
  if [[ "$scene" == "figurines" ]]; then
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/group_soft_topk10_tokens"
  else
    printf "%s\n" "$ROOT/runs/prototypes/mask_group_lift/${scene}_teacher_codebook_k256"
  fi
}

scene_calibration() {
  local scene=$1
  if [[ "$scene" == "ramen" || "$scene" == "waldo_kitchen" ]]; then
    printf "%s\n" "category_percentile:1:99"
  else
    printf "%s\n" "frame_minmax:0:100"
  fi
}

level_tag=${CODES_PER_LEVEL// /x}
faiss_args=()
if [[ "$FAISS_GPU" == "1" ]]; then
  faiss_args+=(--faiss_gpu)
fi

for scene in $SCENES; do
  dataset="$ROOT/drsplat_data/lerf_ovs/$scene"
  labels="$ROOT/drsplat_data/lerf_ovs/label/$scene"
  checkpoint="$ROOT/runs/drsplat/${scene}_1_pq_openclip_topk45_weight_128/chkpnt0.pth"
  pq_index="$ROOT/ckpts/pq_index.faiss"
  group_dir=$(scene_group_dir "$scene")
  group_codebook="$group_dir/group_features.npy"
  group_assignments="$group_dir/point_group_assignments.npz"
  run_root="$ROOT/runs/gaussian_multilevel_codebook/${scene}_k${level_tag}_seed0"
  artifact="$run_root/artifact"
  group_hierarchy="$run_root/group_hierarchy_top${GROUP_TOP_M}"
  IFS=: read -r calibration calibration_low calibration_high <<< "$(scene_calibration "$scene")"

  for path in "$dataset" "$labels" "$checkpoint" "$pq_index" "$group_codebook" "$group_assignments"; do
    [[ -e "$path" ]] || { echo "Missing input for $scene: $path" >&2; exit 1; }
  done
  mkdir -p "$run_root"

  if [[ ! -f "$group_hierarchy/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_compact_group_hierarchy.py \
      --group_features "$group_codebook" \
      --assignments "$group_assignments" \
      --top_m "$GROUP_TOP_M" \
      --output_dir "$group_hierarchy" \
      > "$LOG_DIR/${scene}_group_hierarchy_top${GROUP_TOP_M}.log" 2>&1
  fi

  if [[ "$RUN_BUILD" == "1" && ! -f "$artifact/manifest.json" ]]; then
    "$PYTHON_BIN" -u build_gaussian_multilevel_codebook.py \
      --drsplat_checkpoint "$checkpoint" \
      --pq_index "$pq_index" \
      --codes_per_level $CODES_PER_LEVEL \
      --train_samples "$TRAIN_SAMPLES" \
      --iterations "$KMEANS_ITERATIONS" \
      --assignment_chunk "$ASSIGNMENT_CHUNK" \
      "${faiss_args[@]}" \
      --output_dir "$artifact" \
      > "$LOG_DIR/${scene}_k${level_tag}_build.log" 2>&1
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    for variant in codebook_only "$RGR_VARIANT_TAG"; do
      rgr_alpha=0.0
      group_args=()
      if [[ "$variant" != "codebook_only" ]]; then
        rgr_alpha=$RGR_ALPHA
        group_args+=(
          --group_hierarchy_dir "$group_hierarchy"
          --group_aggregation weighted
          --group_score_power 1.0
          --point_gate_floor 0.1
          --point_gate_power 1.0
        )
      fi
      output="$run_root/eval_${variant}_${calibration}_l${calibration_low}_h${calibration_high}"
      if [[ -f "$output/metrics.json" ]]; then
        echo "reuse scene=$scene variant=$variant output=$output"
        continue
      fi
      "$PYTHON_BIN" -u eval_lerf_ovs_gaussian_codebook_miou.py \
        -s "$dataset" -m "$run_root" \
        --geometry_checkpoint "$checkpoint" \
        --codebook_dir "$artifact" \
        --label_dir "$labels" \
        --rgr_alpha "$rgr_alpha" \
        "${group_args[@]}" \
        --score_calibration "$calibration" \
        --calibration_low "$calibration_low" \
        --calibration_high "$calibration_high" \
        --thresholds $THRESHOLDS \
        --output "$output" \
        > "$LOG_DIR/${scene}_k${level_tag}_${variant}_${calibration}.log" 2>&1
    done
  fi
done

echo "large Gaussian codebook probe complete: scenes=$SCENES levels=$CODES_PER_LEVEL"
