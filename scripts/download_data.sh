#!/bin/bash
set -e

REPO_URL="https://git.geomar.de/willi-rath/ship_routing_data.git"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_LARGE_DIR="${PROJECT_ROOT}/data/large"

# Check if data exists
if [ -d "${DATA_LARGE_DIR}" ] && [ -n "$(ls -A "${DATA_LARGE_DIR}"/*.zarr 2>/dev/null)" ]; then
    echo "Data already exists at ${DATA_LARGE_DIR}"
    echo "To re-download, first remove the data/large directory"
    exit 0
fi

# Create temp directory in system temp (HPC-friendly)
TEMP_CLONE_DIR=$(mktemp -d -t ship_routing_data.XXXXXX)
trap "rm -rf ${TEMP_CLONE_DIR}" EXIT  # Cleanup on exit

mkdir -p "${DATA_LARGE_DIR}"

echo "Cloning Git LFS repository from ${REPO_URL}..."
echo "This may take a while (downloading ~23GB)..."
echo "Using temp directory: ${TEMP_CLONE_DIR}"

# Clone and extract
git clone "${REPO_URL}" "${TEMP_CLONE_DIR}"
mv "${TEMP_CLONE_DIR}"/data/*.zarr "${DATA_LARGE_DIR}/"

echo "Data downloaded successfully to ${DATA_LARGE_DIR}"
# Cleanup handled by trap
