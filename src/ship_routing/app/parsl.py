"""Parsl apps for distributed ship routing optimization.

This module provides Parsl-decorated functions for running routing optimization
tasks in distributed environments (local thread pools, SLURM clusters, etc.).

Example
-------
>>> import parsl
>>> from parsl.config import Config
>>> from parsl.executors import ThreadPoolExecutor
>>> from ship_routing.app.parsl import run_single_experiment
>>> from ship_routing.app.config import RoutingConfig
>>>
>>> parsl.load(Config(executors=[ThreadPoolExecutor()]))
>>> config = RoutingConfig(...)  # Configure journey, forcing, hyperparams
>>> future = run_single_experiment(config)
>>> result = future.result()  # Returns RoutingResult directly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from parsl import python_app

if TYPE_CHECKING:
    from ship_routing.app.config import RoutingConfig
    from ship_routing.app.routing import RoutingResult


@python_app
def run_single_experiment(config: "RoutingConfig", walltime=None) -> "RoutingResult":
    """Run one routing optimization.

    This function executes in a Parsl worker process. The RoutingConfig
    object is pickled by Parsl and unpickled in the worker.

    Parameters
    ----------
    config : RoutingConfig
        Complete routing configuration including journey, forcing data paths,
        and hyperparameters.
    walltime : int, optional
        Maximum execution time in seconds. Parsl special keyword argument.

    Returns
    -------
    RoutingResult
        Optimization result containing seed member, elite population, and logs.
    """
    from ship_routing.app.routing import RoutingApp

    return RoutingApp(config).run()
