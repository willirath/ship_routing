"""Parsl apps for distributed ship routing optimization.

This module provides Parsl-decorated functions for running routing optimization
tasks in distributed environments (local thread pools, SLURM clusters, etc.).

Example
-------
>>> import parsl
>>> from parsl.config import Config
>>> from parsl.executors import ThreadPoolExecutor
>>> from ship_routing.app.parsl import run_single_experiment
>>>
>>> parsl.load(Config(executors=[ThreadPoolExecutor()]))
>>> future = run_single_experiment(
...     journey_config={"lon_waypoints": (-80.5, -11.0), ...},
...     forcing_config={"currents_path": "data/currents.zarr", ...},
...     hyper_params={"population_size": 4, ...},
... )
>>> result_bytes = future.result()
"""

from __future__ import annotations

from typing import Any

from parsl import python_app


@python_app
def run_single_experiment(
    journey_config: dict[str, Any],
    forcing_config: dict[str, Any],
    hyper_params: dict[str, Any],
) -> bytes:
    """Run one routing optimization, return msgpack bytes.

    This function executes in a Parsl worker process. All imports are inside
    the function body to ensure they're available in the worker environment.

    Parameters
    ----------
    journey_config : dict
        Journey configuration dict (passed to JourneyConfig constructor).
        Keys: name, lon_waypoints, lat_waypoints, time_start, speed_knots,
        time_resolution_hours, etc.
    forcing_config : dict
        Forcing data configuration dict (passed to ForcingConfig constructor).
        Keys: currents_path, waves_path, winds_path, engine, etc.
    hyper_params : dict
        Hyperparameters dict (passed to HyperParams constructor).
        Keys: population_size, generations, mutation_width_fraction, etc.

    Returns
    -------
    bytes
        Msgpack-serialized RoutingResult. Deserialize with
        RoutingResult.from_msgpack(result_bytes).
    """
    from ship_routing.app.config import (
        ForcingConfig,
        HyperParams,
        JourneyConfig,
        RoutingConfig,
    )
    from ship_routing.app.routing import RoutingApp

    config = RoutingConfig(
        journey=JourneyConfig(**journey_config),
        forcing=ForcingConfig(**forcing_config),
        hyper=HyperParams(**hyper_params),
    )
    # TODO: Can we just return RoutingApp? msgpack was only introduced for / with Redis.
    result = RoutingApp(config).run()
    return result.to_msgpack()
