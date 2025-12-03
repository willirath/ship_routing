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
    Members with invalid costs (NaN or inf) are filtered out before selection.

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

    # Filter out members with invalid costs (NaN or inf)
    valid_members = [m for m in members if _is_valid_cost(m.cost)]
    if not valid_members:
        return []

    sorted_members = sorted(valid_members, key=lambda m: m.cost)
    elite_count = int(np.ceil(len(sorted_members) * quantile))
    elite_pool = sorted_members[:elite_count]
    indices = rng.integers(0, len(elite_pool), size=target_size)
    return [elite_pool[idx] for idx in indices]


def _is_valid_cost(cost: float) -> bool:
    """Check if a cost value is valid (not NaN or inf)."""
    return not (np.isnan(cost) or np.isinf(cost))


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
    worse solutions while rejecting routes with invalid costs (NaN or inf).
    Routes with NaN or inf costs are always rejected in favor of valid routes.

    Parameters
    ----------
    p : float
        Probability of selecting the higher-cost route
    route_a : Route
        First route candidate
    route_b : Route
        Second route candidate
    cost_a : float
        Cost of first route (NaN/inf if invalid)
    cost_b : float
        Cost of second route (NaN/inf if invalid)
    rng : random generator
        Random number generator

    Returns
    -------
    tuple of (Route, float)
        Selected (route, cost) pair
    """
    # Fail-fast: if route_b is invalid, always return route_a
    if not _is_valid_cost(cost_b):
        return route_a, cost_a

    # Fail-fast: if route_a is invalid (cost_b is valid), return route_b
    if not _is_valid_cost(cost_a):
        return route_b, cost_b

    # Both routes are valid: proceed with normal selection logic
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
