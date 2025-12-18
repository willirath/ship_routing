"""Parsl configuration for different execution environments.

Provides configurations for:
- Local execution (for testing)
- SLURM HPC clusters (production)
"""

from pathlib import Path

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname

from ship_routing.htc.execution import (
    LocalExecutionConfig,
    SlurmExecutionConfig,
    EXECUTION_CONFIGS,
)

# Type alias for execution configs
ExecutionConfig = LocalExecutionConfig | SlurmExecutionConfig


def get_local_config(
    execution: LocalExecutionConfig,
    run_dir: str | Path | None = None,
) -> Config:
    """Get Parsl config for local execution (testing).

    Uses ThreadPoolExecutor for parallel execution on the local machine.
    Good for testing the workflow before submitting to SLURM.

    Parameters
    ----------
    execution : ExecutionConfig
        Execution configuration containing worker settings
    run_dir : str | Path | None, optional
        Path to run directory (default: "runinfo")

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
        run_dir=str(run_dir) if run_dir else "runinfo",
    )


def get_slurm_config(
    execution: SlurmExecutionConfig,
    run_dir: str | Path | None = None,
) -> Config:
    """Get Parsl config for SLURM HPC clusters.

    Uses HighThroughputExecutor with SlurmProvider for scalable
    execution across multiple SLURM nodes.

    Parameters
    ----------
    execution : SlurmExecutionConfig
        Execution configuration containing SLURM and worker settings
    run_dir : str | Path | None, optional
        Path to run directory (default: "runinfo")

    Returns
    -------
    Config
        Parsl configuration for SLURM execution
    """
    # Build SlurmProvider kwargs conditionally
    provider_kwargs = {
        "partition": execution.partition,
        "nodes_per_block": execution.nodes_per_block,
        "min_blocks": 1,
        "max_blocks": execution.max_blocks,
        "walltime": execution.walltime,
        "mem_per_node": execution.mem_per_node_gb,
        "exclusive": execution.exclusive,
        "scheduler_options": (
            f"#SBATCH --qos={execution.qos}\n"
            f"#SBATCH --cpus-per-task={execution.max_workers}"
        ),
        "worker_init": execution.worker_init or "",
        "launcher": SrunLauncher(),
    }

    # Only add account if specified
    if execution.account is not None:
        provider_kwargs["account"] = execution.account

    return Config(
        executors=[
            HighThroughputExecutor(
                label="slurm",
                address=address_by_hostname(),
                working_dir="runinfo/worker_files",
                max_workers_per_node=execution.max_workers,
                cores_per_worker=1.0,  # Request 1 CPU per worker
                provider=SlurmProvider(**provider_kwargs),
            )
        ],
        strategy="simple",  # Simple scaling (less aggressive than htex_auto_scale)
        max_idletime=600,  # Shutdown idle workers after 10 minutes (prevent churning)
        run_dir=str(run_dir) if run_dir else "runinfo",
    )


def get_execution_and_parsl_config(
    execution_name: str,
    run_dir: str | Path | None = None,
) -> tuple[ExecutionConfig, Config]:
    """Get both execution and Parsl configs for the specified environment.

    Parameters
    ----------
    execution_name : str
        Name of execution config (e.g., "local-small", "nesh-prod-40")
    run_dir : str | Path | None, optional
        Path to run directory (default: "runinfo")

    Returns
    -------
    tuple[ExecutionConfig, Config]
        Execution config and Parsl config

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

    execution_config = EXECUTION_CONFIGS[execution_name]

    if isinstance(execution_config, LocalExecutionConfig):
        parsl_config = get_local_config(execution_config, run_dir=run_dir)
    elif isinstance(execution_config, SlurmExecutionConfig):
        parsl_config = get_slurm_config(execution_config, run_dir=run_dir)
    else:
        raise ValueError(f"Unknown execution config type: {type(execution_config)}")

    return execution_config, parsl_config


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
