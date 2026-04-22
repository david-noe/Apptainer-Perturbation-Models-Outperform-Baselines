#!/bin/bash
# Run fMLP size ablation: original, med, and small for all embedding types
# Dataset: replogle22k562

set -e

DATASET="norman19"

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

for MODEL in "${MODELS[@]}"; do
  echo "============================================"
  echo "Training: ${MODEL} on ${DATASET}"
  echo "============================================"
  uv run cellsimbench train model=${MODEL} dataset=${DATASET}

  echo "============================================"
  echo "Benchmarking: ${MODEL} on ${DATASET}"
  echo "============================================"
  uv run cellsimbench benchmark model=${MODEL} dataset=${DATASET}
done

echo "============================================"
echo "All fMLP size ablation runs complete."
echo "============================================"
