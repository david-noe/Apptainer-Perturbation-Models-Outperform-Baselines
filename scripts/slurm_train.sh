#!/bin/bash
# SLURM batch script for CellSimBench training on a cluster.
#
# Usage:
#   sbatch slurm_train.sh <model> <dataset>
#
# Example:
#   sbatch slurm_train.sh sclambda norman19
#   sbatch slurm_train.sh scgpt   norman19
#   sbatch slurm_train.sh gears   norman19
#
# Before running:
#   1. Build SIF images (on a node with internet access):
#        apptainer pull $SIF_DIR/sclambda.sif docker://millerh1/cellsimbench-sclambda:latest
#        apptainer pull $SIF_DIR/scgpt.sif    docker://millerh1/cellsimbench-scgpt:latest
#        apptainer pull $SIF_DIR/gears.sif    docker://millerh1/cellsimbench-gears:latest
#
#   2. Download datasets:
#        cd $PROJECT_DIR && python data/run_all_get_data.py --workers 4
#
#   3. For scLambda only - create a .env file in $PROJECT_DIR:
#        echo "OPENAI_API_KEY=sk-..." > .env

#SBATCH --job-name=cellsimbench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G

set -euo pipefail

export PYTHONUNBUFFERED=1

MODEL="${1:?Usage: sbatch slurm_train.sh <model> <dataset>}"
DATASET="${2:?Usage: sbatch slurm_train.sh <model> <dataset>}"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$HOME/cellsimbench}"
SIF_DIR="${SIF_DIR:-$SCRATCH/cellsimbench/sif}"

# ── Environment ──────────────────────────────────────────────────────────────
module load stack/2024-06 python/3.12.8

cd "$PROJECT_DIR"

# Activate virtual environment (created once with: python -m venv .venv && source .venv/bin/activate && pip install -e .)
source .venv/bin/activate

mkdir -p logs

echo "Training $MODEL on $DATASET"
echo "SIF directory: $SIF_DIR"
echo "Project directory: $PROJECT_DIR"

cellsimbench train \
    model="$MODEL" \
    dataset="$DATASET" \
    execution.container_runtime=apptainer \
    execution.sif_dir="$SIF_DIR" \
    execution.parallel_folds=false
