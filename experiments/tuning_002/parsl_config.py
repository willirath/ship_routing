"""Parsl configuration for different execution environments.

Provides configurations for:
- Local execution (for testing)
- DKRZ Levante cluster (production)
"""

import os
from typing import Literal

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname

from profiles import TuningProfile


def get_local_config(profile: TuningProfile) -> Config:
    """Get Parsl config for local execution (testing).

    Uses ThreadPoolExecutor for parallel execution on the local machine.
    Good for testing the workflow before submitting to SLURM.

    Parameters
    ----------
    profile : TuningProfile
        Profile containing worker configuration

    Returns
    -------
    Config
        Parsl configuration for local execution
    """
    return Config(
        executors=[
            ThreadPoolExecutor(
                label="local",
                max_threads=profile.workers_per_node,
            )
        ],
        strategy="none",  # No scaling for local
    )


def get_slurm_config(profile: TuningProfile) -> Config:
    """Get Parsl config for DKRZ Levante cluster.

    Uses HighThroughputExecutor with SlurmProvider for scalable
    execution across multiple SLURM nodes.

    Parameters
    ----------
    profile : TuningProfile
        Profile containing SLURM and worker configuration

    Returns
    -------
    Config
        Parsl configuration for SLURM execution
    """
    # Worker initialization script
    # Loads modules, sets proxies, activates pixi environment
    worker_init = """
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

    return Config(
        executors=[
            HighThroughputExecutor(
                label="levante",
                address=address_by_hostname(),
                max_workers_per_node=profile.workers_per_node,
                provider=SlurmProvider(
                    partition=profile.partition,
                    account=os.environ.get("SLURM_ACCOUNT", ""),
                    nodes_per_block=profile.nodes_per_block,
                    min_blocks=1,
                    max_blocks=profile.max_blocks,
                    walltime=profile.walltime,
                    scheduler_options=f"#SBATCH --qos={profile.qos}",
                    worker_init=worker_init,
                    launcher=SrunLauncher(),
                    # Move to compute partition for workers
                    move_files=False,
                ),
            )
        ],
        strategy="htex_auto_scale",  # Auto-scale based on task queue
        max_idletime=120,  # Shutdown idle workers after 2 minutes
    )


def get_parsl_config(
    profile: TuningProfile,
    executor: Literal["local", "slurm"] = "local",
) -> Config:
    """Get Parsl configuration for the specified executor.

    Parameters
    ----------
    profile : TuningProfile
        Profile containing execution parameters
    executor : {"local", "slurm"}, default="local"
        Which executor to use

    Returns
    -------
    Config
        Parsl configuration
    """
    if executor == "local":
        return get_local_config(profile)
    elif executor == "slurm":
        return get_slurm_config(profile)
    else:
        raise ValueError(f"Unknown executor: {executor}")
