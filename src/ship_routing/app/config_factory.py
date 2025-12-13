"""Factory for sampling RoutingConfig instances from parameter spaces.

This module provides generic sampling logic that can be reused across different
tuning experiments. Parameter ranges are specified as dictionaries where values
can be:
- Single values (used as-is)
- Tuples/lists (sampled uniformly each time)

Example:
    >>> param_space = {
    ...     'journey': {
    ...         'lon_waypoints': [(-80.5, -11.0), (-11.0, -80.5)],
    ...         'lat_waypoints': [(30.0, 50.0), (50.0, 30.0)],
    ...         'time_start': ('2021-01-01T00:00:00', '2021-06-01T00:00:00'),
    ...         'speed_knots': (8.0, 10.0, 12.0),
    ...         'time_resolution_hours': 6.0,
    ...     },
    ...     'forcing': {
    ...         'currents_path': 'data/currents.zarr',
    ...         'waves_path': 'data/waves.zarr',
    ...     },
    ...     'hyper': {
    ...         'population_size': (128, 256),
    ...         'generations': (1, 2, 4),
    ...         'learning_rate_time': 0.5,
    ...     }
    ... }
    >>> configs = sample_routing_configs(param_space, n_samples=100, seed=42)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ship_routing.app.config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)


def sample_value(options: tuple | list | Any) -> Any:
    """Sample from options or return single value.

    Parameters
    ----------
    options : tuple | list | Any
        If tuple or list with multiple elements, sample uniformly.
        If single-element tuple/list, return the element.
        Otherwise, return as-is.

    Returns
    -------
    Any
        Sampled or fixed value
    """
    if isinstance(options, (tuple, list)):
        if len(options) == 1:
            return options[0]
        return random.choice(options)
    return options


def sample_routing_configs(
    param_space: dict[str, Any],
    n_samples: int,
    seed: int | None = None,
) -> list[RoutingConfig]:
    """Generate n random RoutingConfig instances from parameter space.

    Parameters
    ----------
    param_space : dict
        Parameter ranges organized by config component:
        - 'journey': dict of JourneyConfig parameters
        - 'forcing': dict of ForcingConfig parameters
        - 'hyper': dict of HyperParams parameters
        - 'ship': dict of Ship parameters (optional)
        - 'physics': dict of Physics parameters (optional)

        Values can be:
        - Single values (used as-is for all samples)
        - Tuples/lists (sampled uniformly for each sample)

    n_samples : int
        Number of configs to generate

    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[RoutingConfig]
        Randomly sampled routing configurations

    Examples
    --------
    >>> param_space = {
    ...     'journey': {
    ...         'lon_waypoints': [(-80, -11), (-11, -80)],
    ...         'lat_waypoints': [(30, 50), (50, 30)],
    ...         'speed_knots': (8.0, 10.0, 12.0),
    ...         'time_resolution_hours': 6.0,
    ...     },
    ...     'hyper': {
    ...         'population_size': (128, 256),
    ...         'generations': (1, 2, 4),
    ...     }
    ... }
    >>> configs = sample_routing_configs(param_space, n_samples=10, seed=42)
    >>> len(configs)
    10
    """
    # Initialize random state for parameter sampling
    if seed is not None:
        random.seed(seed)
        seed_seq = np.random.SeedSequence(seed)
    else:
        random.seed()
        seed_seq = np.random.SeedSequence()

    # Generate independent seeds for each experiment using SeedSequence
    # This ensures proper statistical independence between experiments
    experiment_seed_seqs = seed_seq.spawn(n_samples)

    configs = []
    for exp_seed_seq in experiment_seed_seqs:
        # Sample from parameter ranges
        sampled = _sample_dict(param_space)

        # Generate unique random seed for this experiment
        # SeedSequence guarantees statistical independence between streams
        experiment_seed = int(exp_seed_seq.generate_state(1)[0])

        # Build RoutingConfig from sampled params
        # ship and physics will use defaults if not provided
        hyper_params = sampled.get("hyper", {})
        hyper_params["random_seed"] = experiment_seed

        config = RoutingConfig(
            journey=JourneyConfig(**sampled["journey"]),
            forcing=ForcingConfig(**sampled.get("forcing", {})),
            hyper=HyperParams(**hyper_params),
        )
        configs.append(config)

    return configs


def _sample_dict(d: dict) -> dict:
    """Recursively sample values from nested dict.

    Parameters
    ----------
    d : dict
        Dictionary with potentially nested structure

    Returns
    -------
    dict
        Dictionary with same structure, values sampled
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _sample_dict(value)
        else:
            result[key] = sample_value(value)
    return result
