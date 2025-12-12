"""Execution configurations for different compute environments.

Separates infrastructure/execution settings from experiment parameters,
following standard Parsl idioms.
"""

from dataclasses import dataclass


# Worker initialization script for SLURM environments
# TODO: This is machine specific (for nesh)  What if we want to be able to run similar experiments on different HPC systems?
SLURM_WORKER_INIT = """
# Load compiler environment
module load gcc12-env

# Configure HTTP proxy for package downloads
export http_proxy=http://10.0.7.235:3128
export https_proxy=http://10.0.7.235:3128
export ftp_proxy=http://10.0.7.235:3128
export HTTP_PROXY=http://10.0.7.235:3128
export HTTPS_PROXY=http://10.0.7.235:3128
export FTP_PROXY=http://10.0.7.235:3128

# Initialize Pixi environment for Python dependencies
eval "$(pixi shell-hook)"
"""


# TODO: This is machine specific (for nesh)  What if we want to be able to run similar experiments on different HPC systems?
# I guess we create a NeshExecutionConfig and a LevanteExecutionConfig etc?
@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for execution environment and resources."""

    # Resource allocation
    max_workers: int  # For local: max threads; for SLURM: workers per node
    nodes_per_block: int = 1  # SLURM only: nodes per job block
    max_blocks: int = 1  # SLURM only: maximum concurrent blocks

    # SLURM-specific settings
    walltime: str = "01:00:00"  # Job time limit
    partition: str = "base"  # SLURM partition
    qos: str = "express"  # Quality of service

    # Timeouts
    task_timeout: int = 300  # Per-task timeout in seconds

    # Environment setup (None for local, script for SLURM)
    worker_init: str | None = None


# Predefined execution configurations
EXECUTION_CONFIGS: dict[str, ExecutionConfig] = {
    "local-small": ExecutionConfig(
        max_workers=2,
        task_timeout=300,
    ),
    "local-large": ExecutionConfig(
        max_workers=8,
        task_timeout=600,
    ),
    "slurm-test": ExecutionConfig(
        max_workers=4,
        nodes_per_block=2,
        max_blocks=5,
        walltime="01:00:00",
        partition="base",
        qos="express",
        task_timeout=300,
        worker_init=SLURM_WORKER_INIT,
    ),
    "slurm-prod": ExecutionConfig(
        max_workers=8,
        nodes_per_block=10,
        max_blocks=100,
        walltime="04:00:00",
        partition="base",
        qos="express",
        task_timeout=1000,
        worker_init=SLURM_WORKER_INIT,
    ),
}
