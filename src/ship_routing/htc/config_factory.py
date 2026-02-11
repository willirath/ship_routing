"""Factory for sampling RoutingConfig instances from parameter spaces.

This module provides generic sampling logic that can be reused across different
tuning experiments. Parameter ranges are specified as dictionaries where values
can be:
- Single values (used as-is)
- Tuples/lists (sampled uniformly each time)

Example:
    >>> param_space = {
    ...     'journey': {
    ...         'route': {
    ...             'Atlantic_forward': {
    ...                 'lon_waypoints': (-80.5, -11.0),
    ...                 'lat_waypoints': (30.0, 50.0),
    ...             },
    ...             'Atlantic_backward': {
    ...                 'lon_waypoints': (-11.0, -80.5),
    ...                 'lat_waypoints': (50.0, 30.0),
    ...             },
    ...         },
    ...         'time_start': ('2021-01-01T00:00:00', '2021-06-01T00:00:00'),
    ...         'speed_knots': (8.0, 10.0, 12.0),
    ...         'time_resolution_hours': 6.0,
    ...     },
    ...     'forcing': {
    ...         'currents_path': 'data/currents.zarr',
    ...         'waves_path': 'data/waves.zarr',
    ...         'winds_path': 'data/winds.zarr',
    ...     },
    ...     'hyper': {
    ...         'population_size': (128, 256),
    ...         'generations': (1, 2, 4),
    ...     }
    ... }
    >>> configs = sample_routing_configs(param_space, n_samples=100, seed=42)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ship_routing.app.config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)


def sample_value(options: tuple | list | Any, rng: np.random.Generator) -> Any:
    """Sample from options or return single value.

    Parameters
    ----------
    options : tuple | list | Any
        If tuple or list with multiple elements, sample uniformly.
        If single-element tuple/list, return the element.
        Otherwise, return as-is.
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    Any
        Sampled or fixed value
    """
    if isinstance(options, (tuple, list)):
        if len(options) == 1:
            return options[0]
        return rng.choice(options)
    return options


def sample_routing_configs(
    param_space: dict[str, Any],
    n_samples: int,
    seed: int | None = None,
    transform_fn: Callable[[dict], dict] | None = None,
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

    transform_fn : Callable[[dict], dict], optional
        Function to transform sampled parameters before creating configs.
        Receives the sampled dict (after _sample_dict) and should return
        a modified dict. Useful for computing derived parameters like
        offspring_size = int(population_size * offspring_ratio) where both
        population_size and offspring_ratio are sampled randomly.

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
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Spawn independent RNG streams for each experiment
    # This ensures proper statistical independence between experiments
    experiment_rngs = rng.spawn(n_samples)

    configs = []
    for exp_rng in experiment_rngs:
        # Sample from parameter ranges using this experiment's RNG
        sampled = _sample_dict(param_space, rng=exp_rng)

        # Apply custom transform if provided
        if transform_fn is not None:
            sampled = transform_fn(sampled)

        # Extract seed for config storage (for reproducibility)
        # Access the underlying SeedSequence entropy for logging/serialization
        experiment_seed = int(exp_rng.bit_generator.seed_seq.generate_state(1)[0])

        # Build RoutingConfig from sampled params
        # ship and physics will use defaults if not provided
        hyper_params = sampled.get("hyper", {})
        hyper_params["random_seed"] = experiment_seed

        # TODO: Support sampling ship and physics parameters
        # The RoutingConfig also has .physics and .ship fields that will be sampled
        # from param_space like journey/forcing/hyper are currently sampled.
        #
        # To implement:
        # 1. Add 'ship' and 'physics' sections to param_space dict  (just import defaults from ship_routing for now)
        # 2. Sample from these sections: sampled.get("ship", {}) and sampled.get("physics", {})
        # 3. Pass to RoutingConfig: ship=Ship(**sampled.get("ship", {}))
        # 4. Update docstring examples to show ship/physics param_space structure
        #
        # For now, using default Ship() and Physics() instances.
        config = RoutingConfig(
            journey=JourneyConfig(**sampled["journey"]),
            forcing=ForcingConfig(**sampled.get("forcing", {})),
            hyper=HyperParams(**hyper_params),
            # ship=Ship(**sampled.get("ship", {})),  # TODO: Implement
            # physics=Physics(**sampled.get("physics", {})),  # TODO: Implement
        )
        configs.append(config)

    return configs


def _is_named_options_dict(d: dict) -> bool:
    """Check if dict is a named options dict (all values are dicts).

    Named options dicts are used for paired parameters, where the key is a name
    and the value is a dict of related parameters that must be sampled together.

    Parameters
    ----------
    d : dict
        Dictionary to check

    Returns
    -------
    bool
        True if all values are dicts (and dict is non-empty)

    Examples
    --------
    >>> _is_named_options_dict({"route1": {"lon": ..., "lat": ...}, "route2": {...}})
    True
    >>> _is_named_options_dict({"lon": (1, 2), "lat": (3, 4)})
    False
    """
    if not d:
        return False
    return all(isinstance(v, dict) for v in d.values())


def _sample_dict(d: dict, rng: np.random.Generator) -> dict:
    """Recursively sample values from nested dict.

    Special handling for named options dicts:
    - If a parameter value is a dict where ALL values are also dicts,
      it's treated as a "named options" dict
    - One (name, options_dict) pair is sampled from the dict
    - The options_dict is merged with {"name": name} and returned
    - This enables atomic sampling of paired parameters

    Parameters
    ----------
    d : dict
        Dictionary with potentially nested structure
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    dict
        Dictionary with same structure, values sampled

    Examples
    --------
    >>> param = {
    ...     "route": {
    ...         "forward": {"lon": (1, 2), "lat": (3, 4)},
    ...         "backward": {"lon": (5, 6), "lat": (7, 8)},
    ...     },
    ...     "speed": (10.0, 12.0),
    ... }
    >>> result = _sample_dict(param)
    >>> # result will have lon, lat, name from one route, plus sampled speed
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is a named options dict
            if _is_named_options_dict(value):
                # Sample one (name, options_dict) pair
                name, options_dict = rng.choice(list(value.items()))
                # Merge options dict into result directly (no sampling of its values)
                # Values in options_dict are treated as fixed data, not sampling options
                result.update(options_dict)
                # Add the name
                result["name"] = name
            else:
                # Regular nested dict - recurse
                result[key] = _sample_dict(value, rng=rng)
        else:
            result[key] = sample_value(value, rng=rng)
    return result
