#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="occluded-sitting-smplx"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
git submodule update --init --recursive

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda in WSL2 first." >&2
  exit 1
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libosmesa6 libgl1-mesa-glx libglib2.0-0
else
  apt-get update
  apt-get install -y libosmesa6 libgl1-mesa-glx libglib2.0-0
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file environment.yml --prune
else
  conda env create --file environment.yml
fi
conda run -n "$ENV_NAME" python -m pip install \
  mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
conda run -n "$ENV_NAME" python -m pip install -r third_party/SMPLer-X/requirements.txt
conda run -n "$ENV_NAME" python -m pip install -v -e third_party/SMPLer-X/main/transformer_utils
conda run -n "$ENV_NAME" python -m pip install numpy==1.23.5
conda run -n "$ENV_NAME" python scripts/download_assets.py

echo
echo "Environment and public weights are installed."
echo "Next: place the licensed SMPL/SMPL-X files under models/body_models, then run:"
echo "  conda run -n $ENV_NAME smplx-reconstruct check-assets"
