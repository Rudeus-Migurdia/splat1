#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-drsplat}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${PWD}/.conda/envs/${ENV_NAME}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
SAM_CKPT_PATH="${SAM_CKPT_PATH:-ckpts/sam_vit_h_4b8939.pth}"
SAM_CKPT_URL="${SAM_CKPT_URL:-https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth}"

source "$(dirname "$0")/drsplat_env.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Load miniconda/anaconda on the server first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
  conda create -y -p "${CONDA_ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_PREFIX}"
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements.txt

python -m pip install submodules/langsplat-rasterization
python -m pip install submodules/segment-anything-langsplat
python -m pip install submodules/simple-knn

python -m pip install ninja kmeans_pytorch faiss-cpu
python -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

mkdir -p "$(dirname "${SAM_CKPT_PATH}")"
if [[ ! -f "${SAM_CKPT_PATH}" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "${SAM_CKPT_PATH}" "${SAM_CKPT_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${SAM_CKPT_URL}" -o "${SAM_CKPT_PATH}"
  else
    echo "Neither wget nor curl is available; download SAM manually to ${SAM_CKPT_PATH}." >&2
    exit 1
  fi
fi

python scripts/check_drsplat_ready.py --stage preprocess
echo "Environment is prepared at ${CONDA_ENV_PREFIX}"
echo "Activate with: conda activate ${CONDA_ENV_PREFIX}"
echo "Cache root: ${DRSPLAT_CACHE_DIR}"
