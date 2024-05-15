from numpy._typing._array_like import NDArray
from ship_routing.cost import power_maintain_speed, power_maintain_speed_realistic


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
    np.testing.assert_array_less(0, power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs))


def test_cost_power_law():
    num_test = 1_000_000
    uo = np.random.uniform(-1, 1, size=(num_test,))
    vo = np.random.uniform(-1, 1, size=(num_test,))
    us = np.random.uniform(-1, 1, size=(num_test,))
    vs = np.random.uniform(-1, 1, size=(num_test,))

    power_1 = power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs)
    power_2 = power_maintain_speed(uo=2 * uo, vo=2 * vo, us=2 * us, vs=2 * vs)
    np.testing.assert_almost_equal(2**3, power_2 / power_1)


def test_cost_coeff_dependency():
    num_test = 1_000_000
    uo = np.random.uniform(-1, 1, size=(num_test,))
    vo = np.random.uniform(-1, 1, size=(num_test,))
    us = np.random.uniform(-1, 1, size=(num_test,))
    vs = np.random.uniform(-1, 1, size=(num_test,))

    power_1 = power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=1.0)
    power_2 = power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=3.0)
    np.testing.assert_almost_equal(3.0, power_2 / power_1)


@pytest.mark.parametrize(
    "uovousvs",
    [
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        [np.ones(shape=(123,)).copy() for n in range(4)],
        [1.0, 1.0] + [np.ones(shape=(123,)).copy() for n in range(2)],
        [xr.DataArray(np.ones(shape=(123,))) for n in range(4)],
        [pd.Series(data=[1, 2, 3, 4]) for n in range(4)],
    ],
)
def test_cost_dtypes(uovousvs):
    """Test for many different data types."""
    uo, vo, us, vs = uovousvs
    power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=1.0)


@pytest.mark.parametrize("wave_height", [0.0, 1.0, 10.0])
def test_cost_power_maintain_speed_realistic_isotropic_for_zero_currents_winds(
    wave_height,
):
    spd = 10.0
    power_east = power_maintain_speed_realistic(
        u_ship_og=spd, v_ship_og=0.0, w_wave_height=wave_height
    )
    for bearing in np.random.uniform(-180, 180, size=(999,)):
        us, vs = spd * np.sin(np.deg2rad(bearing)), spd * np.cos(np.deg2rad(bearing))
        np.testing.assert_almost_equal(
            power_east,
            power_maintain_speed_realistic(
                u_ship_og=us, v_ship_og=vs, w_wave_height=wave_height
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
            power_maintain_speed_realistic(
                u_ship_og=u_ship_og,
                v_ship_og=v_ship_og,
                u_wind=u_wind,
                v_wind=v_wind,
                u_current=u_current,
                v_current=v_current,
                w_wave_height=wave_height,
            ),
            power_maintain_speed_realistic(
                u_ship_og=-u_ship_og,
                v_ship_og=-v_ship_og,
                u_wind=-u_wind,
                v_wind=-v_wind,
                u_current=-u_current,
                v_current=-v_current,
                w_wave_height=wave_height,
            ),
        )
