from ship_routing.algorithms import gradient_descent_time_shift

from ship_routing.core import Route, WayPoint
from ship_routing.currents import load_currents

import numpy as np

from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


def test_gradient_descent_time_shift_atopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    # load currents and make zero
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    currents = (0.0 * currents).fillna(0.0)

    # The route above is already optimal. Let's check it's not changed.
    route_1 = gradient_descent_time_shift(
        route=route_0,
        current_data_set=currents,
        time_shift_seconds=1200.0,
        learning_rate_percent=1.0,
    )

    # calc cost
    cost_0 = route_0.cost_through(currents)
    cost_1 = route_1.cost_through(currents)

    # ensure reduction
    np.testing.assert_almost_equal(cost_1, cost_0)
    np.testing.assert_array_almost_equal(
        route_0.data_frame.lon, route_1.data_frame.lon, decimal=2
    )
    np.testing.assert_array_almost_equal(
        route_0.data_frame.lat, route_1.data_frame.lat, decimal=2
    )
    np.testing.assert_array_almost_equal(
        0.0,
        (route_0.data_frame.time - route_1.data_frame.time) / np.timedelta64(1, "s"),
        decimal=2,
    )


def test_gradient_descent_time_shift_nonopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=4.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=20.0, lat=0.0, time=np.datetime64("2001-01-04")),
        )
    )
    # load currents and make zero
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    currents = (0.0 * currents).fillna(0.0)

    # With zero currents, cost is solely determined by speed over ground.
    # The route above has three legs with increasing speeds over ground.
    # The more equal the speeds of the two legs are the lower the cost.
    # So making the low legs faster will reduce cost.
    #
    # update route
    route_1 = gradient_descent_time_shift(
        route=route_0,
        current_data_set=currents,
        time_shift_seconds=1200.0,
        learning_rate_percent=1.0,
    )

    # calc cost
    cost_0 = route_0.cost_through(currents)
    cost_1 = route_1.cost_through(currents)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-1.0, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)
