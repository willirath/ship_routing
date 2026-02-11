"""High-Throughput Computing (HTC) infrastructure for ship routing parameter sweeps.

This package provides tools for running large-scale parameter sweeps using Parsl
for distributed execution. It includes:

- Configuration sampling from parameter spaces
- Execution environment profiles (local, SLURM/HPC)
- Ship routing configuration factories
- Orchestration logic for running sweeps and collecting results

Example
-------
>>> from ship_routing.htc import (
...     sample_routing_configs,
...     run_tuning_sweep,
...     EXECUTION_CONFIGS,
... )
>>>
>>> # Define parameter space
>>> param_space = {
...     'journey': {...},
...     'forcing': {...},
...     'hyper': {...},
... }
>>>
>>> # Generate configs
>>> configs = sample_routing_configs(param_space, n_samples=100, seed=42)
>>>
>>> # Run sweep
>>> results = run_tuning_sweep(configs, "local-small", "results.msgpack")
"""

from ship_routing.htc.config_factory import sample_routing_configs
from ship_routing.htc.execution import (
    EXECUTION_CONFIGS,
    LocalExecutionConfig,
    NeshExecutionConfig,
    SlurmExecutionConfig,
)
from ship_routing.htc.orchestration import (
    make_result_key,
    run_tuning_sweep,
    save_results,
)
from ship_routing.htc.parsl_configs import (
    get_parsl_config,
    get_execution_and_parsl_config,
)

__all__ = [
    # Configuration sampling
    "sample_routing_configs",
    # Execution configs
    "EXECUTION_CONFIGS",
    "LocalExecutionConfig",
    "SlurmExecutionConfig",
    "NeshExecutionConfig",
    # Parsl setup
    "get_parsl_config",
    "get_execution_and_parsl_config",
    # Orchestration
    "run_tuning_sweep",
    "make_result_key",
    "save_results",
]
