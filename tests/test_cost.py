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
