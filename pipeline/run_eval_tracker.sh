#!/usr/bin/env bash
# Run all SAM3 tracker-evaluation variants (C-strict, D) across both days.
# Variants A and E use production runs and don't need separate inference.
#
# Usage:
#   pixi run -e tracker bash script/run_eval_tracker.sh

set -euo pipefail

CONFIG_DIR="config/tracker_eval"

CONFIGS=(
    "$CONFIG_DIR/tracker_frame_zero_day_28.yaml"
    "$CONFIG_DIR/tracker_frame_zero_day_29.yaml"
    "$CONFIG_DIR/tracker_rerun_fixed_day_28.yaml"
    "$CONFIG_DIR/tracker_rerun_fixed_day_29.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    python -m script.run_tracker --config "$cfg"
done

echo ""
echo "All SAM3 evaluation variants complete."
