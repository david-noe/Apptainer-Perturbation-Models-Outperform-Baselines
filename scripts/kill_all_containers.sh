#!/bin/bash
# Kill all running Docker containers
# Usage: ./scripts/kill_all_containers.sh

set -e

RUNNING=$(docker ps -q)

if [ -z "$RUNNING" ]; then
    echo "No running Docker containers found."
    exit 0
fi

echo "Killing all running Docker containers..."
docker kill $RUNNING
echo "Done. All running containers have been killed."
