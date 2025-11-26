from __future__ import annotations

import numpy as np

from ..core.routes import Route


def stochastic_mutation(
    route: Route = None,
    number_of_iterations: int = 1,
    mod_width: float | None = None,
    max_move_meters: float | None = None,
    rng=None,
) -> Route:
    """Apply non-local mutations without acceptance logic (selection handles that)."""
    if mod_width is None or max_move_meters is None:
        raise ValueError("mod_width and max_move_meters must be provided")

    rng = rng or np.random
    mutated = route
    for _ in range(number_of_iterations):
        mutated = mutated.move_waypoints_left_nonlocal(
            center_distance_meters=rng.uniform(
                mod_width / 2.0, mutated.length_meters - mod_width / 2.0
            ),
            width_meters=mod_width,
            max_move_meters=max_move_meters * rng.uniform(-1, 1),
        )
    return mutated


__all__ = ["stochastic_mutation"]
