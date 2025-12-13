import numpy as np
import pytest
from pathlib import Path

from ship_routing.algorithms import (
    crossover_routes_minimal_cost,
    crossover_routes_random,
)
from ship_routing.core import Route, WayPoint
from ship_routing.core.population import PopulationMember
from ship_routing.core.data import load_currents, load_waves, load_winds
from conftest import TEST_DATA_DIR


@pytest.fixture(scope="session")
def forcing():
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2024-01_100W-020E_10N-65N.nc"
    )
    winds = load_winds(
        data_file=TEST_DATA_DIR
        / "winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2024-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )
    waves = load_waves(
        data_file=TEST_DATA_DIR
        / "waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2024-01_1d-max_100W-020E_10N-65N.nc"
    )
    return currents, winds, waves


@pytest.fixture
def routes_shared_endpoints():
    """Routes share start/end but do not intersect in between."""
    route_a = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=2.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    route_b = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=-2.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    return route_a, route_b


@pytest.fixture
def routes_identical():
    route = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    return route, route


@pytest.fixture
def routes_intersecting():
    """Routes intersect once."""
    route_a = Route(
        way_points=(
            WayPoint(lon=0.0, lat=-1.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=1.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=-1.0, time=np.datetime64("2001-01-03")),
        )
    )
    route_b = Route(
        way_points=(
            WayPoint(lon=0.0, lat=1.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=-1.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=1.0, time=np.datetime64("2001-01-03")),
        )
    )
    return route_a, route_b


@pytest.mark.parametrize("rng_seed", list(range(7)))
def test_crossover_random_shared_endpoints(routes_shared_endpoints, rng_seed):
    np.random.seed(rng_seed)
    route_a, route_b = routes_shared_endpoints

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_random(parent_a, parent_b)
    child_route = child_member.route

    assert child_route.way_points[0] == route_a.way_points[0]
    assert child_route.way_points[-1] == route_a.way_points[-1]
    assert child_route in (route_a, route_b)


def test_crossover_random_identical(routes_identical):
    route_a, route_b = routes_identical

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_random(parent_a, parent_b)
    child_route = child_member.route

    assert child_route == route_a


@pytest.mark.parametrize("rng_seed", list(range(7)))
def test_crossover_random_intersecting(routes_intersecting, rng_seed):
    np.random.seed(rng_seed)
    route_a, route_b = routes_intersecting

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_random(parent_a, parent_b)
    child_route = child_member.route

    assert child_route.way_points[0] in (route_a.way_points[0], route_b.way_points[0])
    assert child_route.way_points[-1] in (
        route_a.way_points[-1],
        route_b.way_points[-1],
    )
    assert len(child_route.way_points) >= 2


def test_crossover_minimal_cost_shared_endpoints(routes_shared_endpoints, forcing):
    route_a, route_b = routes_shared_endpoints
    currents, winds, waves = forcing

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_minimal_cost(
        parent_a,
        parent_b,
        current_data_set=currents,
        wind_data_set=winds,
        wave_data_set=waves,
    )
    child_route = child_member.route

    assert child_route.way_points[0] == route_a.way_points[0]
    assert child_route.way_points[-1] == route_a.way_points[-1]
    assert child_route in (route_a, route_b)


def test_crossover_minimal_cost_identical(routes_identical, forcing):
    route_a, route_b = routes_identical
    currents, winds, waves = forcing

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_minimal_cost(
        parent_a,
        parent_b,
        current_data_set=currents,
        wind_data_set=winds,
        wave_data_set=waves,
    )
    child_route = child_member.route

    assert child_route == route_a


def test_crossover_minimal_cost_intersecting(routes_intersecting, forcing):
    route_a, route_b = routes_intersecting
    currents, winds, waves = forcing

    # Wrap routes in PopulationMembers with dummy costs
    parent_a = PopulationMember(route=route_a, cost=100.0)
    parent_b = PopulationMember(route=route_b, cost=100.0)

    child_member = crossover_routes_minimal_cost(
        parent_a,
        parent_b,
        current_data_set=currents,
        wind_data_set=winds,
        wave_data_set=waves,
    )
    child_route = child_member.route

    assert child_route.way_points[0] in (route_a.way_points[0], route_b.way_points[0])
    assert child_route.way_points[-1] in (
        route_a.way_points[-1],
        route_b.way_points[-1],
    )
    assert len(child_route.way_points) >= 2
