from __future__ import annotations

import numpy as np
import xarray as xr

from ..core.config import PHYSICS_DEFAULT, SHIP_DEFAULT, Physics, Ship
from ..core.routes import Route


def crossover_routes_random(
    route_0: Route = None,
    route_1: Route = None,
) -> Route:
    """Randomly cross over routes.

    Segments the two routes at their intersection points and randomly
    selects segments from each parent to create a new route.

    Parameters
    ----------
    route_0 : Route
        First parent route
    route_1 : Route
        Second parent route

    Returns
    -------
    Route
        New route created by random crossover of parent segments
    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    segments_mix = [
        s0s1[np.random.randint(0, 2)] for s0s1 in zip(segments_0, segments_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    ref_timestep_seconds = route_0.data_frame.time.diff().mean() / np.timedelta64(
        1, "s"
    )
    route_mix = route_mix.remove_consecutive_duplicate_timesteps(
        min_time_diff_seconds=ref_timestep_seconds / 5
    )
    return route_mix


def crossover_routes_minimal_cost(
    route_0: Route = None,
    route_1: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    ship: Ship = SHIP_DEFAULT,
    physics: Physics = PHYSICS_DEFAULT,
) -> Route:
    """Cross over routes to minimise cost.

    Segments the two routes at their intersection points and selects
    the lower-cost segment from each pair to create a new route.

    Parameters
    ----------
    route_0 : Route
        First parent route
    route_1 : Route
        Second parent route
    current_data_set : xr.Dataset
        Ocean current forcing data
    wind_data_set : xr.Dataset
        Wind forcing data
    wave_data_set : xr.Dataset
        Wave forcing data
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters

    Returns
    -------
    Route
        New route created by selecting minimum-cost segments
    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    cost_0 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_0
    ]
    cost_1 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_1
    ]
    segments_mix = [
        s0s1c0c1[int(s0s1c0c1[-1] < s0s1c0c1[-2])]
        for s0s1c0c1 in zip(segments_0, segments_1, cost_0, cost_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    route_mix = route_mix.remove_consecutive_duplicate_timesteps()
    return route_mix


__all__ = ["crossover_routes_random", "crossover_routes_minimal_cost"]
