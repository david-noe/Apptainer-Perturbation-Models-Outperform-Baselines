#!/bin/bash
# SLURM batch script to convert Docker tar archives to Apptainer SIF images.
#
# Usage:
#   sbatch slurm_convert_tars.sh
#
# Expects tar files in $PROJECT_DIR/docker_tars/ named cellsimbench-<model>.tar
# Writes SIF files to $PROJECT_DIR/apptainer_sif/

#SBATCH --job-name=convert_tars
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

MODELS=(
    "sclambda"
    "scgpt"
    "gears"
)

PROJECT_DIR="${PROJECT_DIR:-$HOME/cellsimbench}"
TAR_DIR="$PROJECT_DIR/docker_tars"
SIF_DIR="$PROJECT_DIR/apptainer_sif"

mkdir -p "$SIF_DIR" "$PROJECT_DIR/logs"

# Use scratch for apptainer's temp files to avoid quota issues
export TMPDIR="${SCRATCH:-/tmp}/apptainer_tmp_$$"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Converting Docker tars to Apptainer SIF images"
echo "Input:  $TAR_DIR"
echo "Output: $SIF_DIR"
echo "TMPDIR: $TMPDIR"

for model in "${MODELS[@]}"; do
    TAR_FILE="$TAR_DIR/cellsimbench-${model}.tar"
    SIF_FILE="$SIF_DIR/${model}.sif"

    if [[ ! -f "$TAR_FILE" ]]; then
        echo "WARNING: $TAR_FILE not found, skipping"
        continue
    fi

    if [[ -f "$SIF_FILE" ]]; then
        echo "Removing existing $SIF_FILE"
        rm -f "$SIF_FILE"
    fi

    echo ""
    echo "=== Converting $model ==="
    echo "  $TAR_FILE -> $SIF_FILE"
    apptainer build "$SIF_FILE" "docker-archive://$TAR_FILE"
    echo "  Done: $SIF_FILE ($(du -sh "$SIF_FILE" | cut -f1))"
done

echo ""
echo "All conversions complete. SIF files in $SIF_DIR:"
ls -lh "$SIF_DIR"/*.sif
