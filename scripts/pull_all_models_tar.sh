#!/bin/bash
# Pull all cellsimbench model Docker images from DockerHub and save as tar archives
# Usage: ./scripts/save_all_models.sh [output_dir]
# After uploading, convert on server with:
#   singularity build <model>.sif docker-archive://<model>.tar
set -e

DOCKERHUB_USER="millerh1"
OUTPUT_DIR="${1:-./docker_tars}"

# Note: presage is excluded due to Genentech Non-Commercial Software License.
# Users must build presage locally: bash docker/presage/build.sh
MODELS=(
    "sclambda"
    "scgpt"
    "gears"
#    "fmlp"
#    "cellflow"
)

echo "Pulling cellsimbench model images from DockerHub (${DOCKERHUB_USER})..."
mkdir -p "${OUTPUT_DIR}"
echo "Pulling and saving cellsimbench model images to ${OUTPUT_DIR}..."

for model in "${MODELS[@]}"; do
    REMOTE_IMAGE="${DOCKERHUB_USER}/cellsimbench-${model}:latest"
    LOCAL_IMAGE="cellsimbench/${model}:latest"
    TAR_FILE="${OUTPUT_DIR}/cellsimbench-${model}.tar"
    
    echo ""
    echo "=== Processing ${model} ==="
    
    # Pull from DockerHub
    echo "Pulling ${REMOTE_IMAGE}"
    docker pull "${REMOTE_IMAGE}"
    
    # Re-tag for local use with the framework
    echo "Saving to ${TAR_FILE}"
    docker save "${REMOTE_IMAGE}" -o "${TAR_FILE}"
    
    echo "Done with ${model} -> ${TAR_FILE}"
done

echo ""
echo "All images pulled and tared successfully!"
echo "Images are now available as cellsimbench/<model>:latest"
echo ""
echo "NOTE: presage must be built manually due to licensing restrictions:"
echo "  bash docker/presage/build.sh"
