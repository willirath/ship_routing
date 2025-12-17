from __future__ import annotations

import numpy as np
import xarray as xr

from ..core.config import PHYSICS_DEFAULT, SHIP_DEFAULT, Physics, Ship
from ..core.population import PopulationMember
from ..core.routes import Route


def crossover_routes_random(
    parent_a: PopulationMember,
    parent_b: PopulationMember,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    ship: Ship = SHIP_DEFAULT,
    physics: Physics = PHYSICS_DEFAULT,
    hazard_penalty_multiplier: float = 100.0,
) -> PopulationMember:
    """Randomly cross over routes from population members.

    Segments the two routes at their intersection points and randomly
    selects segments from each parent to create a new offspring route.
    If the offspring is invalid (crosses land), returns a random parent.

    Parameters
    ----------
    parent_a : PopulationMember
        First parent member
    parent_b : PopulationMember
        Second parent member
    current_data_set : xr.Dataset, optional
        Ocean current forcing data for cost computation
    wind_data_set : xr.Dataset, optional
        Wind forcing data for cost computation
    wave_data_set : xr.Dataset, optional
        Wave forcing data for cost computation
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics for cost computation
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters for cost computation
    hazard_penalty_multiplier : float, default=100.0
        Hazard penalty multiplier for cost computation

    Returns
    -------
    PopulationMember
        New member created by random crossover with cost, or random parent if offspring invalid
    """
    route_0 = parent_a.route
    route_1 = parent_b.route

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

    # Compute full cost with hazard penalty (benefits from caching)
    child_cost = route_mix.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
        hazard_penalty_multiplier=hazard_penalty_multiplier,
    )

    # If invalid, return random parent
    if np.isnan(child_cost) or np.isinf(child_cost):
        return parent_a if np.random.random() < 0.5 else parent_b

    return PopulationMember(route=route_mix, cost=child_cost)


def crossover_routes_minimal_cost(
    parent_a: PopulationMember,
    parent_b: PopulationMember,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    ship: Ship = SHIP_DEFAULT,
    physics: Physics = PHYSICS_DEFAULT,
    hazard_penalty_multiplier: float = 100.0,
) -> PopulationMember:
    """Cross over routes to minimize cost using segment-level selection.

    Segments the two routes at their intersection points and selects
    the lower-cost segment from each pair to create a new offspring route.
    Prefers valid segments over invalid ones. If the offspring is invalid
    (crosses land), returns the parent with lower cost.

    Parameters
    ----------
    parent_a : PopulationMember
        First parent member
    parent_b : PopulationMember
        Second parent member
    current_data_set : xr.Dataset, optional
        Ocean current forcing data for cost computation
    wind_data_set : xr.Dataset, optional
        Wind forcing data for cost computation
    wave_data_set : xr.Dataset, optional
        Wave forcing data for cost computation
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters
    hazard_penalty_multiplier : float, default=100.0
        Hazard penalty multiplier for cost computation

    Returns
    -------
    PopulationMember
        New member made of minimum-cost segments and cost, or best parent if offspring invalid
    """
    route_0 = parent_a.route
    route_1 = parent_b.route

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

    # Level 1: Defensive segment selection - prefer valid segments
    segments_mix = []
    for s0, s1, c0, c1 in zip(segments_0, segments_1, cost_0, cost_1):
        valid_0 = not (np.isnan(c0) or np.isinf(c0))
        valid_1 = not (np.isnan(c1) or np.isinf(c1))

        if not valid_0 and valid_1:
            segments_mix.append(s1)  # Only s1 valid
        elif valid_0 and not valid_1:
            segments_mix.append(s0)  # Only s0 valid
        elif not valid_0 and not valid_1:
            segments_mix.append(s0)  # Both invalid, arbitrary choice
        else:
            # Both valid: use lower cost (original behavior)
            segments_mix.append(s1 if c1 < c0 else s0)

    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    route_mix = route_mix.remove_consecutive_duplicate_timesteps()

    # Level 2: Validate complete child route with full cost (benefits from caching)
    child_cost = route_mix.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
        hazard_penalty_multiplier=hazard_penalty_multiplier,
    )

    # If child invalid, return parent with lower cost
    if np.isnan(child_cost) or np.isinf(child_cost):
        return parent_a if parent_a.cost <= parent_b.cost else parent_b

    return PopulationMember(route=route_mix, cost=child_cost)


__all__ = ["crossover_routes_random", "crossover_routes_minimal_cost"]
