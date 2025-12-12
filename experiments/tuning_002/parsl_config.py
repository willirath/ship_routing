"""Parsl configuration for different execution environments.

Provides configurations for:
- Local execution (for testing)
- SLURM HPC clusters (production)
"""

import os

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname

from execution_config import (
    LocalExecutionConfig,
    SlurmExecutionConfig,
    EXECUTION_CONFIGS,
)


def get_local_config(execution: LocalExecutionConfig) -> Config:
    """Get Parsl config for local execution (testing).

    Uses ThreadPoolExecutor for parallel execution on the local machine.
    Good for testing the workflow before submitting to SLURM.

    Parameters
    ----------
    execution : ExecutionConfig
        Execution configuration containing worker settings

    Returns
    -------
    Config
        Parsl configuration for local execution
    """
    return Config(
        executors=[
            ThreadPoolExecutor(
                label="local",
                max_threads=execution.max_workers,
            )
        ],
        strategy="none",  # No scaling for local
    )


def get_slurm_config(execution: SlurmExecutionConfig) -> Config:
    """Get Parsl config for SLURM HPC clusters.

    Uses HighThroughputExecutor with SlurmProvider for scalable
    execution across multiple SLURM nodes.

    Parameters
    ----------
    execution : SlurmExecutionConfig
        Execution configuration containing SLURM and worker settings

    Returns
    -------
    Config
        Parsl configuration for SLURM execution
    """
    return Config(
        executors=[
            HighThroughputExecutor(
                label="slurm",
                address=address_by_hostname(),
                max_workers_per_node=execution.max_workers,
                provider=SlurmProvider(
                    partition=execution.partition,
                    account=os.environ.get("SLURM_ACCOUNT", ""),
                    nodes_per_block=execution.nodes_per_block,
                    min_blocks=1,
                    max_blocks=execution.max_blocks,
                    walltime=execution.walltime,
                    scheduler_options=f"#SBATCH --qos={execution.qos}",
                    worker_init=execution.worker_init or "",
                    launcher=SrunLauncher(),
                ),
            )
        ],
        strategy="htex_auto_scale",  # Auto-scale based on task queue
        max_idletime=120,  # Shutdown idle workers after 2 minutes
    )


def get_parsl_config(execution_name: str) -> Config:
    """Get Parsl configuration for the specified execution environment.

    Parameters
    ----------
    execution_name : str
        Name of the execution config (e.g., "local-small", "nesh-prod")

    Returns
    -------
    Config
        Parsl configuration

    Raises
    ------
    ValueError
        If execution_name is not found in EXECUTION_CONFIGS
    """
    if execution_name not in EXECUTION_CONFIGS:
        raise ValueError(
            f"Unknown execution config: {execution_name}. "
            f"Available: {list(EXECUTION_CONFIGS.keys())}"
        )

    execution = EXECUTION_CONFIGS[execution_name]

    # Determine executor type from config class
    if isinstance(execution, LocalExecutionConfig):
        return get_local_config(execution)
    elif isinstance(execution, SlurmExecutionConfig):
        return get_slurm_config(execution)
    else:
        raise ValueError(
            f"Unknown execution config type: {type(execution).__name__}. "
            f"Expected LocalExecutionConfig or SlurmExecutionConfig."
        )
