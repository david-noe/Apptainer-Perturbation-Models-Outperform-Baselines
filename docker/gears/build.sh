#!/bin/bash

# Build script for GEARS Docker container

set -e

echo "Building GEARS Docker container..."

# Build the Docker image
docker buildx build --platform linux/amd64 -f docker/gears/Dockerfile -t cellsimbench/gears:latest .

echo "Docker image built successfully!"
echo "Image: cellsimbench/gears:latest"

