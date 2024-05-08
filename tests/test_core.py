from ship_routing.core import (
    Leg,
    Route,
    WayPoint,
)

from ship_routing.currents import load_currents

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pint
import xarray as xr

import pytest


# fixtures etc.

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


# way points


def test_way_point_is_immutable():
    way_point = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    with pytest.raises(FrozenInstanceError):
        way_point.lon = 5.1


def test_way_point_data_frame_length():
    way_point = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    assert len(way_point.data_frame) == 1


def test_way_point_data_frame_columns():
    way_point = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    assert "lon" in way_point.data_frame.columns
    assert "lat" in way_point.data_frame.columns
    assert "time" in way_point.data_frame.columns


def test_way_point_from_data_frame_roundtrip():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    data_frame = way_point_orig.data_frame
    way_point_new = WayPoint.from_data_frame(data_frame=data_frame)
    assert way_point_new == way_point_orig


def test_way_point_point_lon_lat():
    way_point = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    point = way_point.point
    np.testing.assert_almost_equal(way_point.lon, point.x)
    np.testing.assert_almost_equal(way_point.lat, point.y)


def test_way_point_from_point_roundtrip():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    point = way_point_orig.point
    way_point_new = WayPoint.from_point(point=point, time=way_point_orig.time)
    assert way_point_new == way_point_orig


def test_way_point_move_space_zero():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    way_point_moved = way_point_orig.move_space(
        azimuth_degrees=90.0,
        distance_meters=0.0,
    )
    assert way_point_orig == way_point_moved


def test_way_point_move_space_adds_up():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    way_point_moved_two_small_steps = way_point_orig.move_space(
        azimuth_degrees=90.0, distance_meters=1.0
    ).move_space(azimuth_degrees=90.0, distance_meters=1.0)
    way_point_moved_one_bigger_step = way_point_orig.move_space(
        azimuth_degrees=90.0, distance_meters=2.0
    )

    assert way_point_orig != way_point_moved_two_small_steps
    assert way_point_orig.time == way_point_moved_one_bigger_step.time
    assert way_point_orig.time == way_point_moved_two_small_steps.time
    np.testing.assert_almost_equal(
        way_point_moved_one_bigger_step.lon,
        way_point_moved_two_small_steps.lon,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        way_point_moved_one_bigger_step.lat,
        way_point_moved_two_small_steps.lat,
        decimal=2,
    )


def test_way_point_move_time_zero():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    way_point_moved = way_point_orig.move_time(time_diff=np.timedelta64(0, "ms"))
    assert way_point_orig == way_point_moved


def test_way_point_move_time_back_forth():
    way_point_orig = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01"))
    way_point_moved_forward = way_point_orig.move_time(
        time_diff=np.timedelta64(1_000, "s")
    )
    way_point_moved_back = way_point_moved_forward.move_time(
        time_diff=-np.timedelta64(1_000, "s")
    )
    assert way_point_orig != way_point_moved_forward
    assert way_point_orig == way_point_moved_back


# legs


def test_leg_is_immutable():
    leg = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
    )
    with pytest.raises(FrozenInstanceError):
        leg.way_point_end = WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-02"))


def test_leg_data_frame_length():
    leg = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    assert len(leg.data_frame) == 2


def test_leg_data_frame_columns():
    leg = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    assert "lon" in leg.data_frame.columns
    assert "lat" in leg.data_frame.columns
    assert "time" in leg.data_frame.columns


def test_leg_data_frame_order():
    leg = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    assert leg.data_frame.iloc[0].lon == leg.way_point_start.lon
    assert leg.data_frame.iloc[0].lat == leg.way_point_start.lat
    assert leg.data_frame.iloc[0].time == leg.way_point_start.time
    assert leg.data_frame.iloc[1].lon == leg.way_point_end.lon
    assert leg.data_frame.iloc[1].lat == leg.way_point_end.lat
    assert leg.data_frame.iloc[1].time == leg.way_point_end.time


def test_leg_from_data_frame_roundtrip():
    leg_orig = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    data_frame = leg_orig.data_frame
    leg_new = Leg.from_data_frame(data_frame=data_frame)
    assert leg_orig == leg_new


def test_leg_line_string():
    leg = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    line_string = leg.line_string
    assert len(line_string.coords) == 2
    x, y = line_string.xy
    np.testing.assert_almost_equal(x[0], leg.way_point_start.lon)
    np.testing.assert_almost_equal(x[1], leg.way_point_end.lon)
    np.testing.assert_almost_equal(y[0], leg.way_point_start.lat)
    np.testing.assert_almost_equal(y[1], leg.way_point_end.lat)


def test_leg_from_line_string_roundtrip():
    leg_orig = Leg(
        way_point_start=WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=4, lat=3, time=np.datetime64("2001-01-02")),
    )
    line_string = leg_orig.line_string
    leg_new = Leg.from_line_string(
        line_string=line_string,
        time=(leg_orig.way_point_start.time, leg_orig.way_point_end.time),
    )
    assert leg_new == leg_orig


def test_leg_length_meters():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")
        ),
    )
    ureg = pint.UnitRegistry()
    np.testing.assert_almost_equal(
        1.0, leg.length_meters / float(1 * ureg.nautical_mile / ureg.meter), decimal=2
    )


def test_leg_duration_seconds_forward_order():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")
        ),
    )
    np.testing.assert_almost_equal(24 * 3600, leg.duration_seconds, decimal=2)


def test_leg_duration_seconds_backward_order():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-02")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-01")
        ),
    )
    np.testing.assert_almost_equal(24 * 3600, leg.duration_seconds, decimal=2)


def test_leg_speed_ms():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    ureg = pint.UnitRegistry()
    speed_ms_true = float(1.0 * ureg.knots / ureg.meters * ureg.second)
    speed_ms_test = leg.speed_ms
    np.testing.assert_almost_equal(speed_ms_test, speed_ms_true, decimal=2)


def test_leg_speed_through_water_zero_currents():
    # high res currents
    # load currents and make zero
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    currents = (0.0 * currents).fillna(0.0)
    leg = Leg(
        way_point_start=WayPoint(lon=0, lat=12.0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=10, lat=13.0, time=np.datetime64("2001-01-03")),
    )
    np.testing.assert_almost_equal(
        leg.speed_ms, leg.speed_through_water_ms(current_data_set=currents)
    )


def test_leg_speed_through_water_nearly_zero_speed_over_ground():
    # high res currents
    # load currents and make zero
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    currents = (0.0 * currents).fillna(0.0)
    currents["uo"] += 1.0  # eastward current 1m/s
    # displace by very little just to pin down eastward direction
    leg = Leg(
        way_point_start=WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0 + 1e-6, lat=0.0, time=np.datetime64("2001-12-31")),
    )
    np.testing.assert_almost_equal(
        -1.0, leg.speed_through_water_ms(current_data_set=currents)
    )


def test_leg_speed_through_water_no_displacement():
    # high res currents
    # load currents and make zero
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
    )
    # no displacement
    leg = Leg(
        way_point_start=WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-12-31")),
    )


def test_leg_time_after_distance_halftime():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    half_time = leg.time_at_distance(distance_meters=leg.length_meters / 2.0)
    np.testing.assert_almost_equal(
        1.0,
        (leg.way_point_end.time - half_time) / (half_time - leg.way_point_start.time),
        decimal=3,
    )


def test_leg_split_at_distance():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 60, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 60, time=np.datetime64("2001-01-01T00:02:00")
        ),
    )
    split_distance = leg.length_meters / 2.0
    leg_0, leg_1 = leg.split_at_distance(distance_meters=split_distance)
    assert leg_0.way_point_end == leg_1.way_point_start
    np.testing.assert_almost_equal(leg_0.way_point_end.lon, 0, decimal=2)
    np.testing.assert_almost_equal(leg_0.way_point_end.lat, 0, decimal=2)
    np.testing.assert_almost_equal(
        0.0,
        (leg_0.way_point_end.time - np.datetime64("2001-01-01T00:01:00"))
        / np.timedelta64(1, "s"),
        decimal=1,
    )


@pytest.mark.parametrize("num_refine", [2, 4, 7])
def test_leg_refine(num_refine):
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-01T02:00:00")
        ),
    )
    new_distance = leg.length_meters / (num_refine - 1e-3)
    legs_refined = leg.refine(distance_meters=new_distance)
    assert len(legs_refined) == num_refine


def test_leg_overlaps_time():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-01T02:00:00")
        ),
    )
    assert leg.overlaps_time(time=np.datetime64("2001-01-01T01:00:00"))
    assert leg.overlaps_time(time=leg.way_point_start.time)
    assert leg.overlaps_time(time=leg.way_point_end.time)


def test_leg_cost_through_zero_currents():
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-01T12:00:00")),
    )
    # load currents and make zero
    current_data_set = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    current_data_set["uo"] = (current_data_set["uo"] * 0.0).fillna(0.0)
    current_data_set["uo"] = (current_data_set["vo"] * 0.0).fillna(0.0)
    speed_ms = leg.speed_ms
    cost_true = 12 * 3600.0 * speed_ms**3
    cost_test = leg.cost_through(current_data_set=current_data_set)
    np.testing.assert_almost_equal(1.0, cost_true / cost_test, decimal=2)


def test_leg_cost_over_land_is_nan():
    current_data_set = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    leg = Leg(
        way_point_start=WayPoint(lon=-180, lat=0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=0, time=np.datetime64("2001-02-01")),
    )
    assert np.isnan(leg.cost_through(current_data_set=current_data_set))


def test_leg_azimuth_degrees_north_east_south_west():
    leg_north = Leg(
        way_point_start=WayPoint(lon=0, lat=-1, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-02")),
    )
    leg_east = Leg(
        way_point_start=WayPoint(lon=-1, lat=0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=1, lat=0, time=np.datetime64("2001-01-02")),
    )
    leg_south = Leg(
        way_point_start=WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=-1, time=np.datetime64("2001-01-02")),
    )
    leg_west = Leg(
        way_point_start=WayPoint(lon=1, lat=0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=-1, lat=0, time=np.datetime64("2001-01-02")),
    )
    np.testing.assert_almost_equal(0.0, leg_north.azimuth_degrees % 360.0, decimal=2)
    np.testing.assert_almost_equal(90.0, leg_east.azimuth_degrees % 360.0, decimal=2)
    np.testing.assert_almost_equal(180.0, leg_south.azimuth_degrees % 360.0, decimal=2)
    np.testing.assert_almost_equal(270.0, leg_west.azimuth_degrees % 360.0, decimal=2)


def test_leg_azimuth_degrees_north_east_south_west():
    leg_north = Leg(
        way_point_start=WayPoint(lon=0, lat=-1, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-02")),
    )
    leg_east = Leg(
        way_point_start=WayPoint(lon=-1, lat=0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=1, lat=0, time=np.datetime64("2001-01-02")),
    )
    leg_south = Leg(
        way_point_start=WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=0, lat=-1, time=np.datetime64("2001-01-02")),
    )
    leg_west = Leg(
        way_point_start=WayPoint(lon=1, lat=0, time=np.datetime64("2001-01-01")),
        way_point_end=WayPoint(lon=-1, lat=0, time=np.datetime64("2001-01-02")),
    )
    np.testing.assert_almost_equal(0.0, leg_north.bw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(90.0, leg_east.bw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(180.0, leg_south.bw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(270.0, leg_west.bw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(0.0, leg_north.fw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(90.0, leg_east.fw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(180.0, leg_south.fw_azimuth_degrees % 360, decimal=2)
    np.testing.assert_almost_equal(270.0, leg_west.fw_azimuth_degrees % 360, decimal=2)


def test_leg_uv_over_ground_ms_north():
    ureg = pint.UnitRegistry()
    speed_ms_true = float(1.0 * ureg.knots / ureg.meters * ureg.second)
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    u_true, v_true = 0, speed_ms_true
    u_test, v_test = leg.uv_over_ground_ms
    np.testing.assert_almost_equal(u_true, u_test, decimal=2)
    np.testing.assert_almost_equal(v_true, v_test, decimal=2)


def test_leg_uv_over_ground_ms_south():
    ureg = pint.UnitRegistry()
    speed_ms_true = float(1.0 * ureg.knots / ureg.meters * ureg.second)
    leg = Leg(
        way_point_start=WayPoint(
            lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    u_true, v_true = 0, -speed_ms_true
    u_test, v_test = leg.uv_over_ground_ms
    np.testing.assert_almost_equal(u_true, u_test, decimal=2)
    np.testing.assert_almost_equal(v_true, v_test, decimal=2)


def test_leg_uv_over_ground_ms_east():
    ureg = pint.UnitRegistry()
    speed_ms_true = float(1.0 * ureg.knots / ureg.meters * ureg.second)
    leg = Leg(
        way_point_start=WayPoint(
            lon=-1 / 2 / 60.0, lat=0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=1 / 2 / 60.0, lat=0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    u_true, v_true = speed_ms_true, 0
    u_test, v_test = leg.uv_over_ground_ms
    np.testing.assert_almost_equal(u_true, u_test, decimal=2)
    np.testing.assert_almost_equal(v_true, v_test, decimal=2)


def test_leg_uv_over_ground_ms_west():
    ureg = pint.UnitRegistry()
    speed_ms_true = float(1.0 * ureg.knots / ureg.meters * ureg.second)
    leg = Leg(
        way_point_start=WayPoint(
            lon=1 / 2 / 60.0, lat=0, time=np.datetime64("2001-01-01T00:00:00")
        ),
        way_point_end=WayPoint(
            lon=-1 / 2 / 60.0, lat=0, time=np.datetime64("2001-01-01T01:00:00")
        ),
    )
    u_true, v_true = -speed_ms_true, 0
    u_test, v_test = leg.uv_over_ground_ms
    np.testing.assert_almost_equal(u_true, u_test, decimal=2)
    np.testing.assert_almost_equal(v_true, v_test, decimal=2)


# route


def test_route_is_immutable():
    route = Route(
        way_points=(
            WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),
            WayPoint(lon=1, lat=3, time=np.datetime64("2001-01-02")),
        )
    )
    with pytest.raises(FrozenInstanceError):
        route.way_points = route.way_points + (
            WayPoint(lon=2, lat=3, time=np.datetime64("2024-02-03")),
        )


def test_route_no_singleton_routes():
    with pytest.raises(ValueError) as valerr:
        Route(way_points=(WayPoint(lon=1, lat=2, time=np.datetime64("2001-01-01")),))
    assert "at least two" in str(valerr.value)


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_legs_number(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert len(route.legs) == (num_way_points - 1)


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_legs_types(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert all(isinstance(l, Leg) for l in route.legs)


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_legs_roundtrip(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    legs = route.legs
    route_2 = Route.from_legs(legs)
    assert route == route_2


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_data_frame_length(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert len(route.data_frame) == num_way_points


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_data_frame_columns(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert "lon" in route.data_frame.columns
    assert "lat" in route.data_frame.columns
    assert "time" in route.data_frame.columns


def test_route_data_frame_order():
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(2)
            )
        )
    )
    np.testing.assert_almost_equal(
        route.data_frame.iloc[0].lon, route.way_points[0].lon
    )
    np.testing.assert_almost_equal(
        route.data_frame.iloc[0].lat, route.way_points[0].lat
    )
    np.testing.assert_almost_equal(
        route.data_frame.iloc[1].lon, route.way_points[1].lon
    )
    np.testing.assert_almost_equal(
        route.data_frame.iloc[1].lat, route.way_points[1].lat
    )


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_from_data_frame_roundtrip(num_way_points):
    route_orig = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    data_frame = route_orig.data_frame
    route_new = Route.from_data_frame(data_frame=data_frame)

    assert route_new == route_orig


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_line_string_len(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    line_string = route.line_string
    assert len(line_string.coords) == num_way_points


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_line_string_lon_lat(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    line_string = route.line_string
    x, y = line_string.xy
    np.testing.assert_almost_equal(x, [w.lon for w in route.way_points])
    np.testing.assert_almost_equal(y, [w.lat for w in route.way_points])


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_from_line_string_roundtrip(num_way_points):
    route_orig = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    line_string = route_orig.line_string
    route_new = Route.from_line_string(
        line_string=line_string, time=(w.time for w in route_orig.way_points)
    )


@pytest.mark.parametrize("num_way_points", (2, 3, 15))
def test_route_len(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert len(route) == num_way_points


@pytest.mark.parametrize("num_way_points", (3, 5, 15))
def test_route_slicing_len(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    assert len(route[0:2]) == 2
    assert len(route[0:3]) == 3
    assert len(route[:]) == num_way_points


def test_route_slicing_type():
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(2)
            )
        )
    )
    assert isinstance(route[:2], Route)
    assert isinstance(route[:].way_points[0], WayPoint)


@pytest.mark.parametrize("num_way_points", (3, 5, 15))
def test_route_slicing_no_singletons(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    with pytest.raises(ValueError) as valerr:
        r = route[1:2]
    assert "at least two way points" in str(valerr.value)
    with pytest.raises(ValueError) as valerr:
        r = route[2]
    assert "at least two way points" in str(valerr.value)


@pytest.mark.parametrize("num_way_points", (3, 5, 15))
def test_route_slicing_values(num_way_points):
    route = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(num_way_points)
            )
        )
    )
    route_1_3_sliced = route[1:3]
    route_1_3 = Route(way_points=route.way_points[1:3])
    assert route_1_3 == route_1_3_sliced


def test_route_add_identical_wp_dropped():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=2, time=np.datetime64("2001-01-02")),
        )
    )
    route_1 = Route(
        way_points=(
            WayPoint(
                lon=0, lat=2, time=np.datetime64("2001-01-02")
            ),  # same as last above
            WayPoint(lon=0, lat=4, time=np.datetime64("2001-01-03")),
        )
    )
    assert len(route_0 + route_1) == len(route_0) + len(route_1) - 1


def test_route_add_len():
    route_0 = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(3)
            )
        )
    )
    # Second route does not overlap
    route_1 = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=(n + 3) * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(3)
            )
        )
    )
    route_0_1 = route_0 + route_1
    assert len(route_0_1) == len(route_0) + len(route_1)


def test_route_add_order():
    route_0 = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=n * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(3)
            )
        )
    )
    # Second route does not overlap
    route_1 = Route(
        way_points=tuple(
            (
                WayPoint(
                    lon=np.random.uniform(-180, 180),
                    lat=np.random.uniform(-90, 90),
                    time=(n + 3) * np.timedelta64(1, "D") + np.datetime64("2001-01-01"),
                )
                for n in range(3)
            )
        )
    )
    route_0_1 = route_0 + route_1
    assert route_0_1[0:3] == route_0
    assert route_0_1[3:6] == route_1


def test_route_length_meters():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    ureg = pint.UnitRegistry()
    np.testing.assert_almost_equal(
        1.0, route.length_meters / float(2 * ureg.nautical_mile / ureg.meter), decimal=2
    )


def test_route_strict_monotonic_time():
    route_strictly_monotonic = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    assert route_strictly_monotonic.strictly_monotonic_time is True
    route_simply_monotonic = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
        )
    )
    assert route_simply_monotonic.strictly_monotonic_time is False
    route_non_monotonic = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
        )
    )
    assert route_non_monotonic.strictly_monotonic_time is False


def test_route_sort_in_time():
    route_unsorted = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
        )
    )
    route_sorted_truth = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    route_sorted_test = route_unsorted.sort_in_time()
    assert route_sorted_test == route_sorted_truth


def test_route_sort_in_time_idempotency():
    route_unsorted = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
        )
    )
    route_sorted_once = route_unsorted.sort_in_time()
    assert route_sorted_once != route_unsorted
    route_sorted_twice = route_sorted_once.sort_in_time()
    assert route_sorted_twice == route_sorted_once


def test_route_remove_consecutive_duplicate_timesteps():
    route_with_duplicates = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=3 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    route_without_duplicates_truth = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 2 / 60.0, time=np.datetime64("2001-01-02")),
            # second point will be deleted
            WayPoint(lon=0, lat=-1 / 2 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    assert (
        route_with_duplicates.remove_consecutive_duplicate_timesteps()
        == route_without_duplicates_truth
    )


def test_route_distance_meters():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0 / 10.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=1 / 60.0 / 10.0, time=np.datetime64("2001-01-03")),
        )
    )
    ureg = pint.UnitRegistry()
    nm_in_meters = float(ureg.nautical_mile / ureg.meter)
    distance_truth = (0, nm_in_meters / 10.0, 2 * nm_in_meters / 10.0)
    distance_test = route.distance_meters
    np.testing.assert_almost_equal(distance_truth[0], distance_test[0])
    np.testing.assert_almost_equal(
        1.0, np.array(distance_truth[1:]) / np.array(distance_test[1:]), decimal=2
    )


def test_route_refine_2_to_many():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    new_dist = route.length_meters / (2.0 - 1e-3)
    route_refined = route.refine(distance_meters=new_dist)
    assert len(route_refined) > len(route)
    assert route_refined.strictly_monotonic_time


def test_route_refine_no_refinement():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    new_dist = 1.5 * route.length_meters
    route_refined = route.refine(distance_meters=new_dist)
    assert route_refined == route


def test_route_move_waypoint_zero():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    route_moved = route.move_waypoint(n=1, azimuth_degrees=90.0, distance_meters=0.0)
    df = route.data_frame
    df_moved = route_moved.data_frame
    np.testing.assert_array_almost_equal(df.lon, df_moved.lon)
    np.testing.assert_array_almost_equal(df.lat, df_moved.lat)


def test_route_move_waypoint_north():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 10 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 10 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    ureg = pint.UnitRegistry()
    route_moved = route.move_waypoint(
        n=0,
        azimuth_degrees=0.0,
        distance_meters=float(ureg.nautical_mile / ureg.meter / 10.0),
    )
    np.testing.assert_almost_equal(0, route_moved.way_points[0].lat, decimal=3)


def test_route_cost_through_zero_currents():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    current_data_set = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    current_data_set["uo"] = (current_data_set["uo"] * 0.0).fillna(0.0)
    current_data_set["vo"] = (current_data_set["vo"] * 0.0).fillna(0.0)
    speeds = np.array([l.speed_ms for l in route.legs])
    durations = np.array([l.duration_seconds for l in route.legs])
    cost_true = np.sum(durations * speeds**3)
    cost_test = route.cost_through(current_data_set=current_data_set)
    np.testing.assert_almost_equal(1.0, cost_true / cost_test, decimal=2)


def test_route_cost_through_zero_currents_scaling():
    route_slow = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01T00:00:00")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01T06:00:00")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-01T12:00:00")),
        )
    )
    route_fast = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01T00:00:00")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01T01:00:00")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-01T02:00:00")),
        )
    )
    current_data_set = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    current_data_set["uo"] = (current_data_set["uo"] * 0.0).fillna(0.0)
    current_data_set["vo"] = (current_data_set["vo"] * 0.0).fillna(0.0)
    cost_slow = route_slow.cost_through(current_data_set=current_data_set)
    cost_fast = route_fast.cost_through(current_data_set=current_data_set)
    np.testing.assert_almost_equal(6.0**2, cost_fast / cost_slow, decimal=2)


def test_route_waypoint_azimuth():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-02")),
            WayPoint(lon=1 / 60.0, lat=0, time=np.datetime64("2001-01-03")),
            WayPoint(lon=2 / 60.0, lat=0, time=np.datetime64("2001-01-04")),
            WayPoint(lon=2 / 60.0, lat=-1 / 60.0, time=np.datetime64("2001-01-05")),
            WayPoint(lon=2 / 60.0, lat=-2 / 60.0, time=np.datetime64("2001-01-06")),
        )
    )
    wp_az_true = (0, 45, 90, 135, 180, 180)
    wp_az_test = tuple(route.waypoint_azimuth(n=n) for n in range(len(route)))
    np.testing.assert_almost_equal(wp_az_test, wp_az_true)


def test_route_segment_at_assert_matching_wps():
    route_0 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=0.1, time=np.datetime64("2001-01-01T00:00:00")),
            WayPoint(lon=1.0, lat=0.0, time=np.datetime64("2001-01-01T00:01:00")),
            WayPoint(lon=1.0, lat=1.0, time=np.datetime64("2001-01-01T00:02:00")),
            WayPoint(lon=-0.5, lat=-1.5, time=np.datetime64("2001-01-01T00:03:00")),
        )
    )
    route_1 = Route(
        way_points=(
            WayPoint(lon=0.0, lat=-0.1, time=np.datetime64("2001-01-01T01:00:00")),
            WayPoint(lon=-0.5, lat=-0.5, time=np.datetime64("2001-01-01T02:00:00")),
            WayPoint(lon=0.5, lat=-1.0, time=np.datetime64("2001-01-01T03:00:00")),
            WayPoint(lon=-0.5, lat=-1.5, time=np.datetime64("2001-01-01T03:00:00")),
        )
    )
    segments_of_route_0, segments_of_route_1 = route_0.segment_at(other=route_1)
    # segments of route 0 meet each other
    for s0, s1 in zip(segments_of_route_0[:-1], segments_of_route_0[1:]):
        assert s0.way_points[-1] == s1.way_points[0]
    # segments of route 1 meet each other
    for s0, s1 in zip(segments_of_route_1[:-1], segments_of_route_1[1:]):
        assert s0.way_points[-1] == s1.way_points[0]
    # ends of segs of r0 meet starts of segs of r1
    for s0, s1 in zip(segments_of_route_0[:-1], segments_of_route_1[1:]):
        assert s0.way_points[-1].point == s1.way_points[0].point
    for s0, s1 in zip(segments_of_route_1[:-1], segments_of_route_0[1:]):
        assert s0.way_points[-1].point == s1.way_points[0].point


def test_route_split_at_distance():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=2, time=np.datetime64("2001-01-03")),
            WayPoint(lon=0, lat=3, time=np.datetime64("2001-01-04")),
        )
    )
    dist_split = (route.distance_meters[1] + route.distance_meters[2]) / 2.0
    route_0, route_1 = route.split_at_distance(distance_meters=dist_split)
    assert route_0.way_points[-1] == route_1.way_points[0]
    assert route_0.way_points[0] == route.way_points[0]
    assert route_1.way_points[-1] == route.way_points[-1]


def test_route_replace_way_point():
    route_orig = Route(
        way_points=(
            WayPoint(lon=0, lat=0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=1, lat=1, time=np.datetime64("2001-01-02")),
            WayPoint(lon=0, lat=2, time=np.datetime64("2001-01-03")),
        )
    )
    new_wp_1 = WayPoint(lon=-1, lat=1, time=np.datetime64("2001-01-02"))
    route_changed = route_orig.replace_waypoint(
        n=1,
        new_way_point=new_wp_1,
    )
    assert route_changed != route_orig
    assert route_changed.way_points[0] == route_orig.way_points[0]
    assert route_changed.way_points[1] == new_wp_1
    assert route_changed.way_points[-1] == route_orig.way_points[-1]


def test_route_wp_at_distance_middle():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 10 / 60.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=0, lat=1 / 10 / 60.0, time=np.datetime64("2001-01-03")),
        )
    )
    query_dist = route.length_meters / 2.0
    wp_test = route.waypoint_at_distance(distance_meters=query_dist)
    wp_true = WayPoint(lon=0, lat=0, time=np.datetime64("2001-01-02"))
    np.testing.assert_almost_equal(wp_test.lon, wp_true.lon, decimal=2)
    np.testing.assert_almost_equal(wp_test.lat, wp_true.lat, decimal=2)
    np.testing.assert_almost_equal(
        0.0, (wp_test.time - wp_true.time) / np.timedelta64(1, "s"), decimal=0
    )


def test_route_wp_at_distance_edge_at_start_and_end():
    route = Route(
        way_points=(
            WayPoint(lon=10, lat=-23.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=50, lat=32.0, time=np.datetime64("2001-01-13")),
        )
    ).refine(distance_meters=500_000)

    # select at start
    wp_test = route.waypoint_at_distance(distance_meters=0)
    wp_true = route.way_points[0]
    np.testing.assert_almost_equal(wp_test.lon, wp_true.lon, decimal=2)
    np.testing.assert_almost_equal(wp_test.lat, wp_true.lat, decimal=2)
    np.testing.assert_almost_equal(
        0.0, (wp_test.time - wp_true.time) / np.timedelta64(1, "s"), decimal=0
    )

    # select at end
    wp_test = route.waypoint_at_distance(distance_meters=route.length_meters)
    wp_true = route.way_points[-1]
    np.testing.assert_almost_equal(wp_test.lon, wp_true.lon, decimal=2)
    np.testing.assert_almost_equal(wp_test.lat, wp_true.lat, decimal=2)
    np.testing.assert_almost_equal(
        0.0, (wp_test.time - wp_true.time) / np.timedelta64(1, "s"), decimal=0
    )


def test_route_resample_with_distance():
    route = Route(
        way_points=(
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01")),
            WayPoint(lon=10, lat=0.0, time=np.datetime64("2001-01-11")),
        )
    )
    new_distances = tuple(np.linspace(0, route.length_meters, 11))
    route_resampled = route.resample_with_distance(distances_meters=new_distances)
    for n in range(11):
        np.testing.assert_array_almost_equal(
            route_resampled.way_points[n].lon, 1.0 * n, decimal=2
        )
        np.testing.assert_array_almost_equal(
            route_resampled.way_points[n].lat, 0.0, decimal=2
        )
        np.testing.assert_array_almost_equal(
            (route_resampled.way_points[n].time - np.datetime64("2001-01-01"))
            / np.timedelta64(1, "s"),
            n * 24 * 3600,
            decimal=2,
        )


def test_route_calc_gradient_across_track_zero_currents():
    # straight route with one center point
    route = Route(
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
    # With zero currents, a straight line has minimal cost already.
    # So we expect a zero gradient independent of the length of the shift.
    cost_gradient_across_track = route.cost_gradient_across_track(
        n=1, distance_meters=100.0, current_data_set=currents
    )
    np.testing.assert_almost_equal(0.0, cost_gradient_across_track)


def test_route_calc_gradient_along_track_zero_currents():
    # straight route with one center point
    route = Route(
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
    # With zero currents, a straight line has minimal cost already.
    # So we expect a zero gradient independent of the length of the shift.
    cost_gradient_along_track = route.cost_gradient_along_track(
        n=1, distance_meters=100.0, current_data_set=currents
    )
    np.testing.assert_almost_equal(0.0, cost_gradient_along_track)


def test_route_calc_gradient_time_shift_zero_currents():
    # straight route with one center point
    route = Route(
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
    # With zero currents, a straight line has minimal cost already.
    # So we expect a zero gradient independent of the length of the shift.
    cost_gradient_time_shift = route.cost_gradient_time_shift(
        n=1, time_shift_seconds=1200.0, current_data_set=currents
    )
    np.testing.assert_almost_equal(0.0, cost_gradient_time_shift)
