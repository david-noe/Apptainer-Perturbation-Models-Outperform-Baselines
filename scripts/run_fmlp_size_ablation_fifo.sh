#!/bin/bash
# Run fMLP size ablation: original, med, and small for all embedding types
# Runs up to NUM_GPUS models in parallel, assigning each a dedicated GPU.

DATASET="norman19"
NUM_GPUS=4

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

# Create a FIFO to act as a GPU queue/semaphore
GPU_FIFO=$(mktemp -u)
mkfifo "$GPU_FIFO"
exec 3<>"$GPU_FIFO"
rm "$GPU_FIFO"

# Seed the queue with GPU IDs
for ((i=0; i<NUM_GPUS; i++)); do
  echo "$i" >&3
done

PIDS=()

for MODEL in "${MODELS[@]}"; do
  # Block until a GPU is available
  read -r GPU_ID <&3

  (
    # Always release the GPU back to the queue on exit (success or failure)
    trap 'echo "$GPU_ID" >&3' EXIT

    echo "============================================"
    echo "[GPU ${GPU_ID}] Training: ${MODEL} on ${DATASET}"
    echo "============================================"
    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run cellsimbench train model=${MODEL} dataset=${DATASET}

    echo "============================================"
    echo "[GPU ${GPU_ID}] Benchmarking: ${MODEL} on ${DATASET}"
    echo "============================================"
    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run cellsimbench benchmark model=${MODEL} dataset=${DATASET} +run_nir_analysis=true
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
