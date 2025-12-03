import numpy as np
import pytest

from ship_routing.algorithms import select_from_pair, select_from_population
from ship_routing.core import Route, WayPoint
from ship_routing.core.population import PopulationMember


@pytest.fixture
def dummy_routes():
    """Create two dummy routes for testing."""
    route_a = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
        )
    )
    route_b = Route(
        way_points=(
            WayPoint(lon=1.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
        )
    )
    return route_a, route_b


def test_select_from_pair_both_valid(dummy_routes):
    """Test selection when both routes are valid - should select lower cost."""
    route_a, route_b = dummy_routes
    rng = np.random.default_rng(42)

    selected, cost = select_from_pair(
        p=0.0, route_a=route_a, route_b=route_b, cost_a=100.0, cost_b=200.0, rng=rng
    )

    assert selected is route_a
    assert cost == 100.0


@pytest.mark.parametrize("invalid_cost", [np.nan, np.inf])
def test_select_from_pair_route_b_invalid(dummy_routes, invalid_cost):
    """Test selection when route_b is invalid (NaN or inf) - should select route_a."""
    route_a, route_b = dummy_routes
    rng = np.random.default_rng(42)

    # Even with p=1.0 (probability of selecting higher cost), should reject invalid
    selected, cost = select_from_pair(
        p=1.0,
        route_a=route_a,
        route_b=route_b,
        cost_a=100.0,
        cost_b=invalid_cost,
        rng=rng,
    )

    assert selected is route_a
    assert cost == 100.0


@pytest.mark.parametrize("invalid_cost", [np.nan, np.inf])
def test_select_from_pair_route_a_invalid(dummy_routes, invalid_cost):
    """Test selection when route_a is invalid (NaN or inf) - should select route_b."""
    route_a, route_b = dummy_routes
    rng = np.random.default_rng(42)

    selected, cost = select_from_pair(
        p=0.0,
        route_a=route_a,
        route_b=route_b,
        cost_a=invalid_cost,
        cost_b=200.0,
        rng=rng,
    )

    assert selected is route_b
    assert cost == 200.0


@pytest.mark.parametrize(
    "invalid_cost_a,invalid_cost_b",
    [
        (np.nan, np.nan),
        (np.nan, np.inf),
        (np.inf, np.nan),
        (np.inf, np.inf),
    ],
)
def test_select_from_pair_both_invalid(dummy_routes, invalid_cost_a, invalid_cost_b):
    """Test selection when both routes are invalid (NaN/inf) - should fallback to route_a."""
    route_a, route_b = dummy_routes
    rng = np.random.default_rng(42)

    selected, cost = select_from_pair(
        p=0.0,
        route_a=route_a,
        route_b=route_b,
        cost_a=invalid_cost_a,
        cost_b=invalid_cost_b,
        rng=rng,
    )

    assert selected is route_a
    # cost should be invalid (NaN or inf)
    assert np.isnan(cost) or np.isinf(cost)


def test_select_from_pair_probabilistic_both_valid(dummy_routes):
    """Test that probabilistic selection works when both routes are valid."""
    route_a, route_b = dummy_routes
    rng = np.random.default_rng(42)

    # With p=1.0, should always select higher cost (route_b with 200.0)
    selected, cost = select_from_pair(
        p=1.0, route_a=route_a, route_b=route_b, cost_a=100.0, cost_b=200.0, rng=rng
    )

    assert selected is route_b
    assert cost == 200.0


@pytest.fixture
def dummy_population(dummy_routes):
    """Create a population with mixed valid and invalid members."""
    route_a, route_b = dummy_routes
    return [
        PopulationMember(route=route_a, cost=100.0),  # valid
        PopulationMember(route=route_b, cost=np.nan),  # invalid
        PopulationMember(route=route_a, cost=150.0),  # valid
        PopulationMember(route=route_b, cost=np.inf),  # invalid
        PopulationMember(route=route_a, cost=200.0),  # valid
    ]


def test_select_from_population_filters_invalid(dummy_population):
    """Test that select_from_population filters out invalid members."""
    rng = np.random.default_rng(42)

    selected = select_from_population(
        members=dummy_population,
        quantile=0.5,  # Select top 50% (lowest cost)
        target_size=1,
        rng=rng,
    )

    # Should only select from 3 valid members (100.0, 150.0, 200.0)
    # Top 50% elite pool contains lowest cost valid members: [100.0, 150.0]
    # Randomly sampled one from elite pool
    assert len(selected) == 1
    assert selected[0].cost in [100.0, 150.0]  # From valid elite pool
    assert not np.isnan(selected[0].cost)
    assert not np.isinf(selected[0].cost)


def test_select_from_population_all_valid(dummy_routes):
    """Test selection when all members are valid."""
    route_a, route_b = dummy_routes
    members = [
        PopulationMember(route=route_a, cost=100.0),
        PopulationMember(route=route_b, cost=150.0),
        PopulationMember(route=route_a, cost=200.0),
    ]
    rng = np.random.default_rng(42)

    selected = select_from_population(
        members=members, quantile=0.33, target_size=2, rng=rng  # Top 33%
    )

    assert len(selected) == 2
    # All should be valid
    for member in selected:
        assert not np.isnan(member.cost)
        assert not np.isinf(member.cost)


def test_select_from_population_all_invalid(dummy_routes):
    """Test selection when all members are invalid."""
    route_a, route_b = dummy_routes
    members = [
        PopulationMember(route=route_a, cost=np.nan),
        PopulationMember(route=route_b, cost=np.inf),
    ]
    rng = np.random.default_rng(42)

    selected = select_from_population(
        members=members, quantile=0.5, target_size=2, rng=rng
    )

    # Should return empty list when all are invalid
    assert len(selected) == 0


def test_select_from_population_empty():
    """Test selection with empty population."""
    rng = np.random.default_rng(42)

    selected = select_from_population(members=[], quantile=0.5, target_size=2, rng=rng)

    assert len(selected) == 0
