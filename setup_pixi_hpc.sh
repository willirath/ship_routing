#!/bin/bash
# Configure Pixi to use HPC scratch storage instead of HOME directory
# This avoids filling the limited HOME quota on HPC systems
#
# Usage:
#   source setup_pixi_hpc.sh <scratch_directory>
#
# Examples:
#   source setup_pixi_hpc.sh /work/myproject/myusername
#   source setup_pixi_hpc.sh $SCRATCH
#   source setup_pixi_hpc.sh $PWD

# Get scratch base from argument or use default
if [ -n "$1" ]; then
    SCRATCH_BASE="$1"
else
    # Fall back to default base path
    SCRATCH_BASE="$PWD"
    echo "No directory specified, using default: ${SCRATCH_BASE}"
    echo "Usage: source setup_pixi_hpc.sh <scratch_directory>"
fi

# Validate path exists
if [ ! -d "$SCRATCH_BASE" ]; then
    echo "ERROR: Directory does not exist: ${SCRATCH_BASE}"
    echo "Please provide a valid scratch directory path"
    echo "Usage: source setup_pixi_hpc.sh <scratch_directory>"
    return 1
fi

# Pixi cache directory (for downloaded packages)
export PIXI_CACHE_DIR="${SCRATCH_BASE}/.pixi/cache"

# Pixi global data directory
export PIXI_HOME="${SCRATCH_BASE}/.pixi/home"

# Rattler cache directory (Pixi's underlying package manager)
export RATTLER_CACHE_DIR="${SCRATCH_BASE}/.cache/rattler"

# Conda package cache directory
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/.conda/pkgs"

# Create directories if they don't exist
mkdir -p "${PIXI_CACHE_DIR}"
mkdir -p "${PIXI_HOME}"
mkdir -p "${RATTLER_CACHE_DIR}"
mkdir -p "${CONDA_PKGS_DIRS}"

echo "Pixi environment configured for HPC:"
echo "  PIXI_CACHE_DIR=${PIXI_CACHE_DIR}"
echo "  PIXI_HOME=${PIXI_HOME}"
echo "  RATTLER_CACHE_DIR=${RATTLER_CACHE_DIR}"
echo "  CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
echo ""
echo "You can now run: pixi install etc."
