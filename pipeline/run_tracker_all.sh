#!/usr/bin/env bash
# Run all tracker variants via the unified launcher.
#
# Each variant maps to a production config and a pixi environment.
# See docs/2a_tracker_eval.md for the full ablation table.
#
# Usage:
#   bash pipeline/run_tracker_all.sh              # all variants
#   bash pipeline/run_tracker_all.sh yolo         # YOLO variants only
#   bash pipeline/run_tracker_all.sh sam3         # SAM3 variants only
#   bash pipeline/run_tracker_all.sh gs2          # GS2 variants only
#   bash pipeline/run_tracker_all.sh --eval       # all variants, benchmark videos only
#   bash pipeline/run_tracker_all.sh sam3 --eval  # SAM3, benchmark videos only
#   bash pipeline/run_tracker_all.sh --overwrite  # re-track even if output exists

set -euo pipefail

# Variant A: YOLO + BoT-SORT
# Variant A1: YOLO + BoT-SORT with ReID
YOLO_VARIANTS=(
    "tracker:config/yolo_botsort.yaml"
    "tracker:config/yolo_botsort_reid_on.yaml"
)

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

flags=""
filter="all"
for arg in "$@"; do
    case "$arg" in
        --eval|--overwrite) flags="$flags $arg" ;;
        *)                  filter="$arg" ;;
    esac
done

variants=()
case "$filter" in
    yolo)    variants=("${YOLO_VARIANTS[@]}") ;;
    sam3)    variants=("${SAM3_VARIANTS[@]}") ;;
    gs2)     variants=("${GS2_VARIANTS[@]}") ;;
    all)     variants=("${YOLO_VARIANTS[@]}" "${SAM3_VARIANTS[@]}" "${GS2_VARIANTS[@]}") ;;
    *)       echo "Unknown filter: $filter (expected: all, yolo, sam3, gs2)" >&2; exit 1 ;;
esac

for entry in "${variants[@]}"; do
    env="${entry%%:*}"
    cfg="${entry#*:}"

    echo ""
    echo "=== [$env] Running: $cfg ==="
    pixi run -e "$env" python -m pipeline.run_tracker --config "$cfg" $flags
done

echo ""
echo "All tracker variants complete."
