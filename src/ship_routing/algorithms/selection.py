from __future__ import annotations

import numpy as np

from ..core.routes import Route


def select_from_population(
    members,
    quantile: float,
    target_size: int,
    rng,
):
    """Select members using quantile-based elite selection."""
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
    """Select higher-cost route with probability p and select lower-cost route otherwise."""
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
