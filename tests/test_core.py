from ship_routing.core import (
    Leg,
    Route,
    WayPoint,
)

from dataclasses import FrozenInstanceError

import numpy as np

import pytest


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
