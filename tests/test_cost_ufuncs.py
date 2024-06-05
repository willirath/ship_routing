from ship_routing.cost_ufuncs import (
    power_maintain_speed_ufunc,
    hazard_conditions_wave_height_ufunc,
)
from ship_routing.config import SHIP_DEFAULT

from random import uniform

import numpy as np

import pytest


def test_cost_positivity():
    """Ensure power is always positive."""
    for _ in range(1_000):
        assert 0 <= power_maintain_speed_ufunc(
            u_current_ms=uniform(-1, 1),
            v_current_ms=uniform(-1, 1),
            u_ship_og_ms=uniform(-10, 10),
            v_ship_og_ms=uniform(-10, 10),
            u_wind_ms=uniform(0, 10),
            v_wind_ms=uniform(-1, 1),
            w_wave_height=uniform(-1, 1),
        )


def test_cost_power_law_scaling_speed_over_ground():
    """Ensure that for steaming through calm water, power scales with the third power of speed."""
    for _ in range(1_000):
        us = uniform(-1, 1)
        vs = uniform(-1, 1)
        power_1 = power_maintain_speed_ufunc(
            u_current_ms=0.0,
            v_current_ms=0.0,
            u_ship_og_ms=us,
            v_ship_og_ms=vs,
            w_wave_height=0.0,
        )
        power_2 = power_maintain_speed_ufunc(
            u_current_ms=0.0,
            v_current_ms=0.0,
            u_ship_og_ms=2 * us,
            v_ship_og_ms=2 * vs,
            w_wave_height=0.0,
        )
        np.testing.assert_almost_equal(2**3, power_2 / power_1)


@pytest.mark.parametrize("wave_height", [0.0, 1.0, 10.0])
def test_cost_power_maintain_speed_realistic_isotropic_for_zero_currents_winds(
    wave_height,
):
    """Ensure that rotating _all_ velocities does not alter power."""
    spd = 10.0
    power_east = power_maintain_speed_ufunc(
        u_ship_og_ms=spd, v_ship_og_ms=0.0, w_wave_height=wave_height
    )
    for bearing in np.random.uniform(-180, 180, size=(999,)):
        us, vs = spd * np.sin(np.deg2rad(bearing)), spd * np.cos(np.deg2rad(bearing))
        np.testing.assert_almost_equal(
            power_east,
            power_maintain_speed_ufunc(
                u_ship_og_ms=us, v_ship_og_ms=vs, w_wave_height=wave_height
            ),
        )


@pytest.mark.parametrize("wave_height", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("wind_speed", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("current_speed", [0.0, 1.0, 10.0])
def test_cost_power_maintain_speed_realistic_reflection_invariant(
    wave_height,
    wind_speed,
    current_speed,
):
    """Ensure that reflecting _all_ velocities does not change power."""
    ship_speed = 10.0
    for _ in range(9):
        bearing_ship = np.random.uniform(-180, 180)
        u_ship_og = ship_speed * np.sin(np.deg2rad(bearing_ship))
        v_ship_og = ship_speed * np.cos(np.deg2rad(bearing_ship))
        bearing_wind = np.random.uniform(-180, 180)
        u_wind = wind_speed * np.sin(np.deg2rad(bearing_wind))
        v_wind = wind_speed * np.cos(np.deg2rad(bearing_wind))
        bearing_current = np.random.uniform(-180, 180)
        u_current = current_speed * np.sin(np.deg2rad(bearing_current))
        v_current = current_speed * np.cos(np.deg2rad(bearing_current))
        np.testing.assert_almost_equal(
            power_maintain_speed_ufunc(
                u_ship_og_ms=u_ship_og,
                v_ship_og_ms=v_ship_og,
                u_wind_ms=u_wind,
                v_wind_ms=v_wind,
                u_current_ms=u_current,
                v_current_ms=v_current,
                w_wave_height=wave_height,
            ),
            power_maintain_speed_ufunc(
                u_ship_og_ms=-u_ship_og,
                v_ship_og_ms=-v_ship_og,
                u_wind_ms=-u_wind,
                v_wind_ms=-v_wind,
                u_current_ms=-u_current,
                v_current_ms=-v_current,
                w_wave_height=wave_height,
            ),
        )


def test_hazard_conditions_wave_height_ufunc_no_hazard_without_waves():
    """Ensure that for absent waves, there is no hazard."""
    assert not hazard_conditions_wave_height_ufunc(w_wave_height_m=0.0)


def test_hazard_conditions_wave_height_ufunc_hazard_for_high_waves():
    """Ensure that for high waves (> 1/20 of the ship length), there is a hazard."""
    assert hazard_conditions_wave_height_ufunc(
        w_wave_height_m=SHIP_DEFAULT.waterline_length_m / 20.0
    )
    assert hazard_conditions_wave_height_ufunc(
        w_wave_height_m=SHIP_DEFAULT.waterline_length_m / 10.0
    )
