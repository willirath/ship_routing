from ship_routing.core.geodesics import (
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
