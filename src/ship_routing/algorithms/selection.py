from __future__ import annotations

import numpy as np

from ..core.routes import Route


def select_from_population(
    members,
    quantile: float,
    target_size: int,
    rng,
):
    """Select members using quantile-based elite selection.

    Selects members from the top quantile of the population by cost,
    then randomly samples from this elite pool to reach target size.

    Parameters
    ----------
    members : list of PopulationMember
        Population members to select from
    quantile : float
        Quantile threshold for elite selection (0.0 to 1.0)
    target_size : int
        Number of members to select
    rng : random generator
        Random number generator

    Returns
    -------
    list of PopulationMember
        Selected members from elite pool
    """
    if not members:
        return []

    sorted_members = sorted(members, key=lambda m: m.cost)
    elite_count = int(np.ceil(len(sorted_members) * quantile))
    elite_pool = sorted_members[:elite_count]
    indices = rng.integers(0, len(elite_pool), size=target_size)
    return [elite_pool[idx] for idx in indices]


def select_from_pair(
    p: float,
    route_a: Route,
    route_b: Route,
    cost_a: float,
    cost_b: float,
    rng,
):
    """Select higher-cost route with probability p and select lower-cost route otherwise.

    Implements probabilistic selection allowing occasional acceptance of
    worse solutions to, e.g., maintain population diversity.

    Parameters
    ----------
    p : float
        Probability of selecting the higher-cost route
    route_a : Route
        First route candidate
    route_b : Route
        Second route candidate
    cost_a : float
        Cost of first route
    cost_b : float
        Cost of second route
    rng : random generator
        Random number generator

    Returns
    -------
    tuple of (Route, float)
        Selected (route, cost) pair
    """
    if cost_b > cost_a:
        higher_route, lower_route = route_b, route_a
        higher_cost, lower_cost = cost_b, cost_a
    else:
        higher_route, lower_route = route_a, route_b
        higher_cost, lower_cost = cost_a, cost_b

    if rng.random() < p:
        return higher_route, higher_cost
    return lower_route, lower_cost


__all__ = ["select_from_population", "select_from_pair"]
