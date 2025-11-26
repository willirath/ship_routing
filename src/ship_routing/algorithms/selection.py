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
    route_original: Route,
    route_mutated: Route,
    cost_original: float,
    cost_mutated: float,
    rng,
):
    """Select between original and mutated route with probability p."""
    if rng.random() < p:
        return route_mutated, cost_mutated
    return route_original, cost_original


__all__ = ["select_from_population", "select_from_pair"]
