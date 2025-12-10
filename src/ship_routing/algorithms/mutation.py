from __future__ import annotations

import numpy as np

from ..core.routes import Route


def stochastic_mutation(
    route: Route = None,
    max_iterations: int = 1,
    mod_width: float | None = None,
    max_move_meters: float | None = None,
    rng=None,
) -> Route:
    """Apply non-local mutations.

    Performs stochastic mutations by moving waypoints perpendicular to the route
    within a specified width and maximum displacement. The actual number of
    iterations is sampled uniformly from [1, max_iterations].

    Parameters
    ----------
    route : Route
        Route to mutate
    max_iterations : int, default=1
        Maximum number of mutation iterations. Actual iterations will be
        sampled uniformly from {1, 2, ..., max_iterations}.
    mod_width : float, optional
        Width of the modification window in meters
    max_move_meters : float, optional
        Maximum distance to move waypoints in meters
    rng : random generator, optional
        Random number generator (defaults to np.random)

    Returns
    -------
    Route
        Mutated route

    Raises
    ------
    ValueError
        If mod_width or max_move_meters are not provided
    """
    if mod_width is None or max_move_meters is None:
        raise ValueError("mod_width and max_move_meters must be provided")

    rng = rng or np.random

    # Sample actual number of iterations uniformly from [1, max_iterations]
    num_iterations = rng.integers(1, max_iterations + 1)

    mutated = route
    for _ in range(num_iterations):
        mutated = mutated.move_waypoints_left_nonlocal(
            center_distance_meters=rng.uniform(
                max(0.0, mod_width / 2.0),
                min(mutated.length_meters, mutated.length_meters - mod_width / 2.0),
            ),
            width_meters=mod_width,
            max_move_meters=max_move_meters * rng.uniform(-1, 1),
        )
        # adapt width if route got shorter
        # TODO: This should be handled by always prescribing a length fraction
        # from higher up in the logic?
        mod_width = min(mod_width, mutated.length_meters)
    return mutated


__all__ = ["stochastic_mutation"]
