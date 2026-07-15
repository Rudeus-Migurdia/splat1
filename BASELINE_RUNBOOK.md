# Dr.Splat Baseline Runbook

This repo expects a prepared scene. The full baseline run has three stages:

1. `train_3dgs.py`: train vanilla 3DGS and save `chkpnt30000.pth`.
2. `preprocessing.py`: generate per-image SAM masks and OpenCLIP features into `<DATASET_PATH>/language_features`.
3. `train.py`: vote/register those features onto the 3DGS checkpoint and write a Dr.Splat point cloud/checkpoint.

## Server Setup

```bash
cd /path/to/Dr-Splat
git submodule update --init --recursive
CONDA_ENV_PREFIX="$PWD/.conda/envs/drsplat" \
DRSPLAT_CACHE_DIR="$PWD/.cache/drsplat" \
bash scripts/setup_drsplat_env.sh
conda activate "$PWD/.conda/envs/drsplat"
source scripts/drsplat_env.sh
```

The setup script installs into an isolated conda prefix, uses project-local caches for pip/Torch/HuggingFace/CUDA, builds the CUDA submodules, installs `tiny-cuda-nn`, and downloads `ckpts/sam_vit_h_4b8939.pth` if it is missing.

If the server uses a different CUDA/PyTorch stack, override `TORCH_INDEX_URL` or install torch manually before rerunning the script.

On a shared lab server, prefer putting the environment and cache on your own scratch/work directory:

```bash
CONDA_ENV_PREFIX=/scratch/$USER/drsplat_env \
DRSPLAT_CACHE_DIR=/scratch/$USER/drsplat_cache \
bash scripts/setup_drsplat_env.sh
conda activate /scratch/$USER/drsplat_env
export DRSPLAT_CACHE_DIR=/scratch/$USER/drsplat_cache
source scripts/drsplat_env.sh
```

## Required Inputs

Dataset layout:

```text
<DATASET_PATH>/
  images/
  sparse/0/              # COLMAP cameras/images/points files
```

Training input:

```text
ckpts/pq_index.faiss
ckpts/sam_vit_h_4b8939.pth
```

Run the pre-flight check before spending GPU time:

```bash
python scripts/check_drsplat_ready.py \
  --dataset "$DATASET_PATH" \
  --stage 3dgs
```

## Baseline Run

```bash
GPU_ID=0 \
DATASET_PATH=/path/to/scene \
OUTPUT_PATH=output/drsplat_baseline_scene \
bash scripts/run_drsplat_baseline.sh
```

By default, `scripts/run_drsplat_baseline.sh` trains vanilla 3DGS first if `output/3dgs_baseline/chkpnt30000.pth` does not exist. If `GS_ITERATIONS` is overridden, the default checkpoint path follows it, for example `chkpnt7000.pth`.

Useful overrides:

```bash
TRAINED_3DGS_PATH=output/3dgs_scene
START_CHECKPOINT=/path/to/existing/chkpnt30000.pth
TRAIN_3DGS_IF_MISSING=0
GS_ITERATIONS=30000
EXTRA_3DGS_ARGS="--eval"
RUN_PREPROCESSING=auto   # default; skip if language_features already exist
RUN_PREPROCESSING=1      # force regeneration
RUN_TRAIN=0              # only preprocess/check
FEATURE_LEVEL=1
TOPK=45
RESOLUTION=-1
EXTRA_TRAIN_ARGS="--eval"
```

Expected outputs:

```text
<TRAINED_3DGS_PATH>/chkpnt30000.pth
<DATASET_PATH>/language_features/*_f.npy
<DATASET_PATH>/language_features/*_s.npy
<OUTPUT_PATH>.../point_cloud/iteration_0/point_cloud.ply
<OUTPUT_PATH>.../chkpnt0.pth
```

## Notes

- `preprocessing.py` can be memory-heavy because it loads all images into a single tensor before feature extraction. If the target scene is large and OOMs, the first code change should be batching the image list.
- The official README has no evaluation script yet. For a baseline smoke test, confirm that preprocessing completes, `train.py` saves iteration 0, and `render_pca.py` can render from the produced model.
