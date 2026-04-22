#!/bin/bash
# Run fMLP size ablation: original, med, and small for all embedding types
# Runs up to MAX_JOBS benchmark jobs in parallel (no GPU pinning).

DATASET="replogle22k562"
MAX_JOBS=6

MODELS=(
  # Med (hidden=256, latent=64)
  fmlp_med_esm2
  fmlp_med_geneformer
  fmlp_med_scgpt
  fmlp_med_genept
  # Small (hidden=128, latent=32)
  fmlp_small_esm2
  fmlp_small_geneformer
  fmlp_small_scgpt
  fmlp_small_genept
  # Original (hidden=512, latent=128)
  fmlp_esm2
  fmlp_geneformer
  fmlp_scgpt
  fmlp_genept
)

# Create a FIFO as a counting semaphore (max MAX_JOBS concurrent)
SEM_FIFO=$(mktemp -u)
mkfifo "$SEM_FIFO"
exec 3<>"$SEM_FIFO"
rm "$SEM_FIFO"

# Seed the semaphore with MAX_JOBS tokens
for ((i=0; i<MAX_JOBS; i++)); do
  echo "x" >&3
done

PIDS=()

for MODEL in "${MODELS[@]}"; do
  # Block until a job slot is available
  read -r _ <&3

  (
    # Always release the slot back on exit (success or failure)
    trap 'echo "x" >&3' EXIT

    echo "============================================"
    echo "Benchmarking: ${MODEL} on ${DATASET}"
    echo "============================================"
    uv run cellsimbench benchmark model=${MODEL} dataset=${DATASET} +run_nir_analysis=true
  ) &

  PIDS+=($!)
done

# Wait for all jobs and track failures
FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "ERROR: Job with PID ${PID} failed."
    FAILED=1
  fi
done

exec 3>&-

if [ "$FAILED" -ne 0 ]; then
  echo "============================================"
  echo "Some fMLP size ablation runs FAILED."
  echo "============================================"
  exit 1
fi

echo "============================================"
echo "All fMLP size ablation runs complete."
echo "============================================"
