#!/usr/bin/env bash
# Install Grounded-SAM-2 and its dependencies (SAM2 checkpoints, GroundingDINO
# checkpoints, PyTorch, editable packages).
#
# Usage:
#   pixi run -e gs2 bash script/setup_gs2.sh

set -euo pipefail

GS2_DIR="ext/Grounded-SAM-2"

if [[ ! -d "$GS2_DIR" ]]; then
    echo "Error: $GS2_DIR not found. Run: git submodule update --init --recursive"
    exit 1
fi

# --- SAM2 checkpoints ---
if ls "$GS2_DIR"/checkpoints/*.pt 1>/dev/null 2>&1; then
    echo "SAM2 checkpoints already exist, skipping download..."
else
    echo "Downloading SAM2 checkpoints..."
    (cd "$GS2_DIR/checkpoints" && bash download_ckpts.sh)
fi

# --- GroundingDINO checkpoints ---
if ls "$GS2_DIR"/gdino_checkpoints/*.pth 1>/dev/null 2>&1; then
    echo "GroundingDINO checkpoints already exist, skipping download..."
else
    echo "Downloading GroundingDINO checkpoints..."
    (cd "$GS2_DIR/gdino_checkpoints" && bash download_ckpts.sh)
fi

# --- PyTorch ---
echo "Installing PyTorch..."
uv pip install torch torchvision torchaudio

# --- Grounded-SAM-2 + GroundingDINO (editable) ---
echo "Installing Grounded-SAM-2 packages..."
export CUDA_HOME="/usr/lib/nvidia-cuda-toolkit/"
(cd "$GS2_DIR" && uv pip install -e . && uv pip install --no-build-isolation -e grounding_dino)

echo ""
echo "GS2 setup complete."
