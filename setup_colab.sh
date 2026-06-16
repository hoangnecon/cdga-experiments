#!/bin/bash
# setup_colab.sh - Colab Environment Setup Script
# Works on Google Colab runtimes to install dependencies and clone required repositories.

set -e

echo "=== [1/3] Installing Pip Dependencies ==="
pip install -q timm==0.9.16 albumentations scipy einops ttach pyyaml

echo "=== [2/3] Checking GeoSeg Repository ==="
# Determine project root (directory where setup_colab.sh is located)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GEOSEG_DIR="${PROJECT_ROOT}/geoseg"

if [ ! -d "${GEOSEG_DIR}" ]; then
    echo "Cloning GeoSeg into ${GEOSEG_DIR}..."
    git clone https://github.com/WangLibo1995/GeoSeg.git "${GEOSEG_DIR}"
else
    echo "GeoSeg directory already exists at ${GEOSEG_DIR}."
fi

echo "=== [3/3] Verification ==="
python -c "import torch; import timm; import albumentations; import einops; print('✓ Basic imports verified. CUDA available:', torch.cuda.is_available())"

echo "=== Colab Setup Complete! ==="
