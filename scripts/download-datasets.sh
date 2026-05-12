#!/bin/bash
#SBATCH --job-name=bench-download-datasets
#SBATCH --output=logs/bench-download-datasets_%j.log
#SBATCH --error=logs/bench-download-datasets_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

# Activate virtual environment
source .venv/bin/activate

# Run download
./scripts/pull_all_datasets.sh