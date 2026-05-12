#!/bin/bash
#SBATCH --job-name=apptainer-pull
#SBATCH --output=logs/apptainer-pull-%j.out
#SBATCH --error=logs/apptainer-pull-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

# --- Modules (Euler) ---
# eth_proxy gives compute nodes outbound HTTPS access for the pull.
module load eth_proxy

# --- Paths ---
SIF_DIR="/cluster/work/bewi/members/dnoe/Perturbation-Models-Outperform-Baselines/apptainer_sif"
mkdir -p "$SIF_DIR"
mkdir -p logs

# --- Route Apptainer caches off $HOME to avoid quota issues ---
export APPTAINER_CACHEDIR="/cluster/work/bewi/members/dnoe/.apptainer/cache"
export APPTAINER_TMPDIR="/cluster/work/bewi/members/dnoe/.apptainer/tmp"

# --- Pulls ---
echo "[$(date)] Pulling sclambda..."
apptainer pull --force "$SIF_DIR/sclambda.sif" docker://millerh1/cellsimbench-sclambda:latest

echo "[$(date)] Pulling scgpt..."
apptainer pull --force "$SIF_DIR/scgpt.sif"    docker://millerh1/cellsimbench-scgpt:latest

echo "[$(date)] Pulling gears..."
apptainer pull --force "$SIF_DIR/gears.sif"    docker://millerh1/cellsimbench-gears:latest

echo "[$(date)] Done. Images in $SIF_DIR:"
ls -lh "$SIF_DIR"

# --- Optional: clean the blob cache after successful build ---
# apptainer cache clean --force