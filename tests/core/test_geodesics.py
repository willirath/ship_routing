from ship_routing.core.geodesics import (
    compute_ellipse_bbox,
    get_refinement_factor,
    knots_to_ms,
    ms_to_knots,
)

from shapely.geometry import LineString

import numpy as np

import pint


def test_refinement_factor():
    """Test different refinement factors."""
    assert 10 == get_refinement_factor(original_dist=1000, new_dist=100)
    assert 34 == get_refinement_factor(original_dist=100, new_dist=3)
    assert 1 == get_refinement_factor(original_dist=1234, new_dist=1234)


def test_knots_to_ms():
    """Test conversion from knots to meters per second."""
    result = knots_to_ms(1.0)
    assert np.isclose(result, 0.514444, rtol=1e-6)
    assert knots_to_ms(0.0) == 0.0
    result_10kt = knots_to_ms(10.0)
    assert np.isclose(result_10kt, 5.14444, rtol=1e-6)


def test_ms_to_knots():
    """Test conversion from meters per second to knots."""
    result = ms_to_knots(0.514444)
    assert np.isclose(result, 1.0, rtol=1e-6)
    assert ms_to_knots(0.0) == 0.0
    result_5ms = ms_to_knots(5.14444)
    assert np.isclose(result_5ms, 10.0, rtol=1e-6)


def test_speed_conversion_round_trip():
    """Test that converting knots -> m/s -> knots preserves the value."""
    test_speeds = [0.0, 1.0, 7.0, 10.0, 20.0, 50.0]

    for speed_knots in test_speeds:
        speed_ms = knots_to_ms(speed_knots)
        speed_knots_back = ms_to_knots(speed_ms)
        assert np.isclose(speed_knots, speed_knots_back, rtol=1e-10)

    # Also test the reverse direction
    for speed_ms in [0.0, 1.0, 5.14444, 10.0]:
        speed_knots = ms_to_knots(speed_ms)
        speed_ms_back = knots_to_ms(speed_knots)
        assert np.isclose(speed_ms, speed_ms_back, rtol=1e-10)


def test_ellipse_bbox_simple_route():
    """Test ellipse bbox for a simple east-west route."""
    lon_start, lat_start = -75.0, 35.0
    lon_end, lat_end = -65.0, 35.0

    lon_min, lon_max, lat_min, lat_max = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=4.0,
        buffer_degrees=1.0,
    )

    # Bbox should contain start and end points
    assert lon_min <= lon_start <= lon_max
    assert lon_min <= lon_end <= lon_max
    assert lat_min <= lat_start <= lat_max
    assert lat_min <= lat_end <= lat_max

    # Bbox should be valid
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert -180 <= lon_min and lon_max <= 180
    assert -90 <= lat_min and lat_max <= 90


def test_ellipse_bbox_different_multipliers():
    """Test that larger multiplier produces larger bbox."""
    lon_start, lat_start = -75.0, 35.0
    lon_end, lat_end = -65.0, 35.0

    bbox_2 = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=2.0,
        buffer_degrees=0.5,
    )

    bbox_4 = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=4.0,
        buffer_degrees=0.5,
    )

    # Larger multiplier should produce wider bbox
    area_2 = (bbox_2[1] - bbox_2[0]) * (bbox_2[3] - bbox_2[2])
    area_4 = (bbox_4[1] - bbox_4[0]) * (bbox_4[3] - bbox_4[2])
    assert area_4 > area_2


def test_ellipse_bbox_polar_route():
    """Test ellipse bbox for a high-latitude route."""
    lon_start, lat_start = 0.0, 80.0
    lon_end, lat_end = 10.0, 85.0

    lon_min, lon_max, lat_min, lat_max = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=3.0,
        buffer_degrees=1.0,
    )

    # Bbox should contain start and end points
    assert lon_min <= lon_start <= lon_max
    assert lon_min <= lon_end <= lon_max
    assert lat_min <= lat_start <= lat_max
    assert lat_min <= lat_end <= lat_max

    # Bbox should be valid
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert -180 <= lon_min and lon_max <= 180
    assert -90 <= lat_min and lat_max <= 90


def test_ellipse_bbox_antimeridian():
    """Test ellipse bbox for a route crossing the antimeridian."""
    lon_start, lat_start = 170.0, 35.0
    lon_end, lat_end = -170.0, 35.0

    lon_min, lon_max, lat_min, lat_max = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=3.0,
        buffer_degrees=1.0,
    )

    # Bbox should be valid (might wrap around antimeridian)
    assert -180 <= lon_min and lon_max <= 180
    assert -90 <= lat_min and lat_max <= 90
    assert lat_min < lat_max


def test_ellipse_bbox_small_route():
    """Test ellipse bbox for a very short route."""
    lon_start, lat_start = 0.0, 0.0
    lon_end, lat_end = 0.1, 0.1

    lon_min, lon_max, lat_min, lat_max = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=2.0,
        buffer_degrees=0.5,
    )

    # Bbox should not be degenerate
    assert lon_min < lon_max
    assert lat_min < lat_max

    # Bbox should contain start and end points
    assert lon_min <= lon_start <= lon_max
    assert lon_min <= lon_end <= lon_max
    assert lat_min <= lat_start <= lat_max
    assert lat_min <= lat_end <= lat_max
