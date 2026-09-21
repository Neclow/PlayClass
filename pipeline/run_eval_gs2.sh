#!/usr/bin/env bash
# Run all Grounded-SAM-2 tracker-evaluation variants (B-strict, B-parity)
# across both days.
#
# Usage:
#   pixi run -e gs2 bash script/run_eval_gs2.sh

set -euo pipefail

CONFIG_DIR="config/tracker_eval"

CONFIGS=(
    "$CONFIG_DIR/gs2_fixed_day_28.yaml"
    "$CONFIG_DIR/gs2_fixed_day_29.yaml"
    "$CONFIG_DIR/gs2_parity_day_28.yaml"
    "$CONFIG_DIR/gs2_parity_day_29.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    python -m src.tracker_eval.run_gs2_tracker --config "$cfg"
done

echo ""
echo "All GS2 evaluation variants complete."
