from numpy._typing._array_like import NDArray
from ship_routing.cost import power_maintain_speed


import numpy as np
import xarray as xr
import pandas as pd

import pytest


def test_cost_positivity():
    num_test = 1_000_000
    uo = np.random.uniform(-1, 1, size=(num_test,))
    vo = np.random.uniform(-1, 1, size=(num_test,))
    us = np.random.uniform(-1, 1, size=(num_test,))
    vs = np.random.uniform(-1, 1, size=(num_test,))
    np.testing.assert_array_less(
        0,
        power_maintain_speed(
            u_current_ms=uo,
            v_current_ms=vo,
            u_ship_og_ms=us,
            v_ship_og_ms=vs,
        ),
    )


def test_cost_power_law_scaling_speed_over_ground():
    num_test = 1_000_000
    us = np.random.uniform(-1, 1, size=(num_test,))
    vs = np.random.uniform(-1, 1, size=(num_test,))

    power_1 = power_maintain_speed(
        u_current_ms=0,
        v_current_ms=0,
        u_ship_og_ms=us,
        v_ship_og_ms=vs,
        w_wave_height=0.0,
    )
    power_2 = power_maintain_speed(
        u_current_ms=0,
        v_current_ms=0,
        u_ship_og_ms=2 * us,
        v_ship_og_ms=2 * vs,
        w_wave_height=0.0,
    )
    np.testing.assert_almost_equal(2**3, power_2 / power_1)


@pytest.mark.parametrize(
    "uo_vo_us_vs_wh",
    [
        [1, 2, 3, 4, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [np.ones(shape=(123,)).copy() for n in range(5)],
        [1.0, 1.0] + [np.ones(shape=(123,)).copy() for n in range(2)] + [23.5],
        [xr.DataArray(np.ones(shape=(123,))) for n in range(5)],
        [pd.Series(data=[1, 2, 3, 4, 5]) for n in range(5)],
    ],
)
def test_cost_dtypes(uo_vo_us_vs_wh):
    """Test for many different data types."""
    uo, vo, us, vs, wh = uo_vo_us_vs_wh
    power_maintain_speed(
        u_current_ms=uo,
        v_current_ms=vo,
        u_ship_og_ms=us,
        v_ship_og_ms=vs,
        w_wave_height=wh,
    )


@pytest.mark.parametrize("wave_height", [0.0, 1.0, 10.0])
def test_cost_power_maintain_speed_realistic_isotropic_for_zero_currents_winds(
    wave_height,
):
    spd = 10.0
    power_east = power_maintain_speed(
        u_ship_og_ms=spd, v_ship_og_ms=0.0, w_wave_height=wave_height
    )
    for bearing in np.random.uniform(-180, 180, size=(999,)):
        us, vs = spd * np.sin(np.deg2rad(bearing)), spd * np.cos(np.deg2rad(bearing))
        np.testing.assert_almost_equal(
            power_east,
            power_maintain_speed(
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
    ship_speed = 10.0
    for n in range(9):
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
            power_maintain_speed(
                u_ship_og_ms=u_ship_og,
                v_ship_og_ms=v_ship_og,
                u_wind_ms=u_wind,
                v_wind_ms=v_wind,
                u_current_ms=u_current,
                v_current_ms=v_current,
                w_wave_height=wave_height,
            ),
            power_maintain_speed(
                u_ship_og_ms=-u_ship_og,
                v_ship_og_ms=-v_ship_og,
                u_wind_ms=-u_wind,
                v_wind_ms=-v_wind,
                u_current_ms=-u_current,
                v_current_ms=-v_current,
                w_wave_height=wave_height,
            ),
        )
