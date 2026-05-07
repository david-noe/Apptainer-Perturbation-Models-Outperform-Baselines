#!/bin/bash
# SLURM batch script for CellSimBench benchmarking on a cluster.
#
# Usage:
#   sbatch slurm_benchmark.sh <model> <dataset>

#SBATCH --job-name=cellsimbench-bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G

set -euo pipefail

export PYTHONUNBUFFERED=1

MODEL="${1:?Usage: sbatch slurm_benchmark.sh <model> <dataset>}"
DATASET="${2:?Usage: sbatch slurm_benchmark.sh <model> <dataset>}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/cellsimbench}"
SIF_DIR="${SIF_DIR:-$SCRATCH/cellsimbench/sif}"

module load stack/2024-06 python/3.12.8

cd "$PROJECT_DIR"
source .venv/bin/activate

mkdir -p logs

echo "Benchmarking $MODEL on $DATASET"

cellsimbench benchmark \
    model="$MODEL" \
    dataset="$DATASET" \
    execution.container_runtime=apptainer \
    execution.sif_dir="$SIF_DIR" \
    execution.parallel_folds=false
