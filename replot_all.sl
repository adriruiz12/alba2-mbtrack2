#!/bin/bash
# Run from project root: bash replot_all.sl

set -e

for study_dir in results/soa_*/; do
    study=$(basename "$study_dir")

    for case_dir in "${study_dir}${study}_"*/; do
        [ -d "$case_dir" ] || continue

        # Strip "{study}_" prefix to recover VALIDATION_CASE
        case_name="${case_dir#${study_dir}${study}_}"
        case_name="${case_name%/}"

        echo "──────────────────────────────────────────"
        echo "  STUDY=$study  VALIDATION_CASE=$case_name"
        echo "  dir: $case_dir"

        STUDY="$study" VALIDATION_CASE="$case_name" \
            python src/simulation/replot.py
    done
done

echo "══════════════════════════════════════════"
echo "  Replot completed."