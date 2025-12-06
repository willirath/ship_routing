from ship_routing.algorithms import (
    gradient_descent_time_shift,
    gradient_descent_along_track,
    gradient_descent_across_track_left,
    InvalidGradientError,
    ZeroGradientsError,
    LargeIncrementError,
)

from ship_routing.core import Route, WayPoint
from ship_routing.core.data import (
    load_currents,
    load_waves,
    load_winds,
    make_hashable,
)

import numpy as np
import pytest

from pathlib import Path
from conftest import TEST_DATA_DIR


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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents),
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
            current_data_set=make_hashable(zero_currents_with_invalid_region),
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
            current_data_set=make_hashable(zero_currents_with_invalid_region),
            distance_meters=100_000.0,
            learning_rate_percent=0.1,
        )


def test_gradient_descent_across_track_left_nonopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=-53.0, lat=40.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=-43.0, lat=41.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=-33.0, lat=39.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=-23.0, lat=40.0, time=np.datetime64("2001-01-04")),
        )
    )
    # load currents, winds, waves
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    waves = load_waves(
        data_file=TEST_DATA_DIR
        / "waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2021-01_1d-max_100W-020E_10N-65N.nc"
    )
    winds = load_winds(
        data_file=TEST_DATA_DIR
        / "winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2021-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )

    # update route
    route_1 = gradient_descent_across_track_left(
        route=route_0,
        current_data_set=currents,
        wave_data_set=waves,
        wind_data_set=winds,
        distance_meters=2_000.0,
        learning_rate_percent=0.1,
        ignore_hazards=True,
    )

    # calc cost
    # We don't care about hazards or cost here and are just testing.
    # So let's ignore hazards for simlicity here.
    cost_0 = route_0.cost_through(currents, ignore_hazards=True)
    cost_1 = route_1.cost_through(currents, ignore_hazards=True)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)


def test_gradient_descent_along_track_nonopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=-53.0, lat=40.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=-42.0, lat=40.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=-34.0, lat=40.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=-23.0, lat=40.0, time=np.datetime64("2001-01-04")),
        )
    )
    # load currents, winds, waves
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    waves = load_waves(
        data_file=TEST_DATA_DIR
        / "waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2021-01_1d-max_100W-020E_10N-65N.nc"
    )
    winds = load_winds(
        data_file=TEST_DATA_DIR
        / "winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2021-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )

    # update route
    route_1 = gradient_descent_along_track(
        route=route_0,
        current_data_set=currents,
        wave_data_set=waves,
        wind_data_set=winds,
        distance_meters=5_000.0,
        learning_rate_percent=0.1,
        ignore_hazards=True,
    )

    # calc cost
    # We don't care about hazards or cost here and are just testing.
    # So let's ignore hazards for simlicity here.
    cost_0 = route_0.cost_through(currents, ignore_hazards=True)
    cost_1 = route_1.cost_through(currents, ignore_hazards=True)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)


def test_gradient_descent_time_shift_nonopt():
    route_0 = Route(
        way_points=(
            WayPoint(lon=-53.0, lat=40.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=-43.0, lat=41.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=-33.0, lat=39.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=-23.0, lat=40.0, time=np.datetime64("2001-01-04")),
        )
    )
    # load currents, winds, waves
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    waves = load_waves(
        data_file=TEST_DATA_DIR
        / "waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2021-01_1d-max_100W-020E_10N-65N.nc"
    )
    winds = load_winds(
        data_file=TEST_DATA_DIR
        / "winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2021-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )

    # update route
    route_1 = gradient_descent_time_shift(
        route=route_0,
        current_data_set=currents,
        wave_data_set=waves,
        wind_data_set=winds,
        time_shift_seconds=1200.0,
        learning_rate_percent=0.1,
        ignore_hazards=True,
    )

    # calc cost
    # We don't care about hazards or cost here and are just testing.
    # So let's ignore hazards for simlicity here.
    cost_0 = route_0.cost_through(currents, ignore_hazards=True)
    cost_1 = route_1.cost_through(currents, ignore_hazards=True)

    # ensure reduction
    assert cost_1 < cost_0
    np.testing.assert_almost_equal(-0.1, 100.0 * (cost_1 - cost_0) / cost_0, decimal=1)
