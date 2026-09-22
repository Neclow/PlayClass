#!/usr/bin/env bash
# Download V-JEPA 2.1 checkpoints and patch the torch.hub cache to avoid
# the src/ namespace collision with this project's own src/ package.
#
# Usage:
#   bash scripts/setup_vjepa2.1.sh

set -euo pipefail

CHECKPOINTS=(
    "vjepa2_1_vitb_dist_vitG_384"
    "vjepa2_1_vitl_dist_vitG_384"
)

CKPT_DIR="${HOME}/.cache/torch/hub/checkpoints"
HUB_DIR="${HOME}/.cache/torch/hub/facebookresearch_vjepa2_main"

mkdir -p "$CKPT_DIR"
for CKPT_NAME in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}.pt"
    if [[ -f "$CKPT_PATH" ]]; then
        echo "Checkpoint already exists: $CKPT_PATH"
    else
        echo "Downloading ${CKPT_NAME}.pt ..."
        wget -q --show-progress -O "$CKPT_PATH" "https://dl.fbaipublicfiles.com/vjepa2/${CKPT_NAME}.pt"
        echo "Saved to $CKPT_PATH"
    fi
done

# --- Step 2: Clone hub repo (if not cached) ---
if [[ -d "$HUB_DIR" ]]; then
    echo "Hub repo already cached: $HUB_DIR"
else
    echo "Fetching hub repo via torch.hub ..."
    python -c "import torch; torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')" 2>/dev/null || true
fi

# --- Step 3: Rename src/ -> vjepa2/ to avoid namespace collision ---
if [[ -d "$HUB_DIR/src" ]]; then
    echo "Renaming hub repo src/ -> vjepa2/ ..."
    mv "$HUB_DIR/src" "$HUB_DIR/vjepa2"
    find "$HUB_DIR" -name "*.py" -exec sed -i 's/from src\./from vjepa2./g; s/import src\./import vjepa2./g' {} +
    echo "Done."
elif [[ -d "$HUB_DIR/vjepa2" ]]; then
    echo "Hub repo already patched (vjepa2/ exists)."
else
    echo "Error: neither src/ nor vjepa2/ found in $HUB_DIR"
    exit 1
fi

echo ""
echo "Setup complete"
