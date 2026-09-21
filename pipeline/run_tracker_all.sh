#!/usr/bin/env bash
# Run tracker evaluation variants via the unified launcher.
#
# Each variant maps to a production config and a pixi environment.
# See docs/2a_tracker_eval.md for the full ablation table.
#
# Usage:
#   bash pipeline/run_eval_tracker.sh          # all variants
#   bash pipeline/run_eval_tracker.sh sam3     # SAM3 variants only
#   bash pipeline/run_eval_tracker.sh gs2      # GS2 variants only

set -euo pipefail

# Variant C-strict: SAM3 frame-0 grounding, no fallbacks
# Variant D: SAM3 adaptive grounding, fixed 60s chunks
# Variant E: SAM3 full method (adaptive chunking + adaptive grounding)
SAM3_VARIANTS=(
    "tracker:config/sam3_baseline.yaml"
    "tracker:config/sam3_adaptive_grounding.yaml"
    "tracker:config/sam3_best.yaml"
)

# Variant B-strict: GS2 fixed 60s, no recovery
# Variant B-parity: GS2 with recovery to match SAM3 scaffolding
GS2_VARIANTS=(
    "gs2:config/gs2_fixed.yaml"
    "gs2:config/gs2_adaptive_recovery.yaml"
)

filter="${1:-all}"

variants=()
case "$filter" in
    sam3)    variants=("${SAM3_VARIANTS[@]}") ;;
    gs2)     variants=("${GS2_VARIANTS[@]}") ;;
    all)     variants=("${SAM3_VARIANTS[@]}" "${GS2_VARIANTS[@]}") ;;
    *)       echo "Unknown filter: $filter (expected: all, sam3, gs2)" >&2; exit 1 ;;
esac

for entry in "${variants[@]}"; do
    env="${entry%%:*}"
    cfg="${entry#*:}"

    echo ""
    echo "=== [$env] Running: $cfg ==="
    pixi run -e "$env" python -m pipeline.run_tracker --config "$cfg"
done

echo ""
echo "All evaluation variants complete."
