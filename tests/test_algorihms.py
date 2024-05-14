from ship_routing.algorithms import (
    gradient_descent_time_shift,
    gradient_descent_along_track,
    gradient_descent_across_track_left,
    InvalidGradientError,
    ZeroGradientsError,
    LargeIncrementError,
)

from ship_routing.core import Route, WayPoint
from ship_routing.currents import load_currents

import numpy as np
import pytest

from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


def test_gradient_descent_across_track_left_zero_gradients_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(ZeroGradientsError):
        _ = gradient_descent_across_track_left(
            route=route_0,
            current_data_set=zero_currents,
            distance_meters=2_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_along_track_zero_gradients_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(ZeroGradientsError):
        _ = gradient_descent_along_track(
            route=route_0,
            current_data_set=zero_currents,
            distance_meters=2_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_time_shift_zero_gradients_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(ZeroGradientsError):
        _ = gradient_descent_time_shift(
            route=route_0,
            current_data_set=zero_currents,
            time_shift_seconds=1_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_across_track_left_large_increment_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=1.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(LargeIncrementError):
        _ = gradient_descent_across_track_left(
            route=route_0,
            current_data_set=zero_currents,
            distance_meters=2_000.0,
            learning_rate_percent=10,
        )


def test_gradient_descent_along_track_large_increment_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=6.0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(LargeIncrementError):
        _ = gradient_descent_along_track(
            route=route_0,
            current_data_set=zero_currents,
            distance_meters=2_000.0,
            learning_rate_percent=10,
        )


def test_gradient_descent_time_shift_large_increment_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=0.0, time=np.datetime64("2001-01-02T01:00:00")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    with pytest.raises(LargeIncrementError):
        _ = gradient_descent_time_shift(
            route=route_0,
            current_data_set=zero_currents,
            time_shift_seconds=100.0,
            learning_rate_percent=10.0,
        )


def test_gradient_descent_across_track_left_invalid_gradient_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=1.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    zero_currents_with_invalid_region = zero_currents.where(zero_currents.lat <= 1.0)
    with pytest.raises(InvalidGradientError):
        _ = gradient_descent_across_track_left(
            route=route_0,
            current_data_set=zero_currents_with_invalid_region,
            distance_meters=100_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_along_track_invalid_gradient_error():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=5.0, lat=5.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=0.0, time=np.datetime64("2001-01-03")),
        )
    )
    zero_currents = (
        0.0
        * load_currents(
            data_file=TEST_DATA_DIR
            / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
        )
    ).fillna(0.0)
    lon = zero_currents.lon
    lat = zero_currents.lat
    zero_currents_with_invalid_region = zero_currents.where(
        ((lon - lat) > 0.01) & (((10 - lon) - lat) > 0.01) & (lat >= 0)
    )
    with pytest.raises(InvalidGradientError):
        _ = gradient_descent_along_track(
            route=route_0,
            current_data_set=zero_currents_with_invalid_region,
            distance_meters=100_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_across_track_left_nonopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=4.0, lat=1.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=10.0, lat=-1.0, time=np.datetime64("2001-01-03")),
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
    route_1 = gradient_descent_across_track_left(
        route=route_0,
        current_data_set=currents,
        distance_meters=2_000.0,
        learning_rate_percent=0.1,
    )

    # calc cost
    cost_0 = route_0.cost_through(currents)
    cost_1 = route_1.cost_through(currents)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)


def test_gradient_descent_along_track_nonopt():
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
    route_1 = gradient_descent_along_track(
        route=route_0,
        current_data_set=currents,
        distance_meters=2_000.0,
        learning_rate_percent=0.1,
    )

    # calc cost
    cost_0 = route_0.cost_through(currents)
    cost_1 = route_1.cost_through(currents)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)


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
        learning_rate_percent=0.1,
    )

    # calc cost
    cost_0 = route_0.cost_through(currents)
    cost_1 = route_1.cost_through(currents)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)
