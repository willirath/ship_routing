"""Execution configurations for different compute environments.

Separates infrastructure/execution settings from experiment parameters,
following standard Parsl idioms.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalExecutionConfig:
    """Execution config for local machine."""

    max_workers: int
    task_timeout: int = 300


@dataclass(frozen=True)
class SlurmExecutionConfig:
    """Generic execution config for SLURM HPC systems."""

    max_workers: int
    nodes_per_block: int
    max_blocks: int
    walltime: str
    partition: str
    qos: str
    task_timeout: int
    mem_per_node_gb: int  # Memory per node in GB (Parsl appends 'g' suffix; guideline: 5GB per worker)
    worker_init: str = ""  # Worker initialization commands (empty = no init)
    exclusive: bool = True  # Request exclusive node allocation (False for shared nodes)
    account: str | None = None  # SLURM account (None = no account required)


@dataclass(frozen=True)
class NeshExecutionConfig(SlurmExecutionConfig):
    """Execution config for nesh HPC system."""

    worker_init: str = (
        # load compiler env
        "module load gcc12-env\n"
        # ensure internet access
        "export http_proxy=http://10.0.7.235:3128\n"
        "export https_proxy=http://10.0.7.235:3128\n"
        "export ftp_proxy=http://10.0.7.235:3128\n"
        "export HTTP_PROXY=http://10.0.7.235:3128\n"
        "export HTTPS_PROXY=http://10.0.7.235:3128\n"
        "export FTP_PROXY=http://10.0.7.235:3128\n"
        # init pixi
        'eval "$(pixi shell-hook)"\n'
    )


# Predefined execution configurations
EXECUTION_CONFIGS = {
    "local-small": LocalExecutionConfig(
        max_workers=2,
        task_timeout=300,
    ),
    "local-large": LocalExecutionConfig(
        max_workers=8,
        task_timeout=600,
    ),
    "nesh-test": NeshExecutionConfig(
        max_workers=2,
        nodes_per_block=1,
        max_blocks=3,  # Reduced from 5 - enough for 40 tasks with 2 workers each
        walltime="01:00:00",
        partition="base",
        qos="express",
        task_timeout=300,
        mem_per_node_gb=10,  # 2 workers × 5GB each
        exclusive=False,  # Shared nodes
    ),
    "nesh-prod": NeshExecutionConfig(
        max_workers=8,
        nodes_per_block=4,
        max_blocks=100,
        walltime="04:00:00",
        partition="base",
        qos="express",
        task_timeout=1000,
        mem_per_node_gb=40,  # 8 workers × 5GB each
        exclusive=False,  # Shared nodes (only using ~25% of node resources)
    ),
}
