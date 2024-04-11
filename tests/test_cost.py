from numpy._typing._array_like import NDArray
from ship_routing.cost import power_maintain_speed


import numpy as np

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
    np.testing.assert_almost_equal(2 ** 3, power_2 / power_1)


def test_cost_coeff_dependency():
    num_test = 1_000_000
    uo = np.random.uniform(-1, 1, size=(num_test,))
    vo = np.random.uniform(-1, 1, size=(num_test,))
    us = np.random.uniform(-1, 1, size=(num_test,))
    vs = np.random.uniform(-1, 1, size=(num_test,))

    power_1 = power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=1.0)
    power_2 = power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=3.0)
    np.testing.assert_almost_equal(3.0, power_2 / power_1)


@pytest.mark.parametrize("uuvv", [
    [1, 2, 3, 4],
    [1.0, 2.0, 3.0, 4.0],
    [np.ones(shape=(123, )).copy() for n in range(4)]
])
def test_cost_dtypes(uuvv: list[int] | list[float] | list[NDArray[np.float64]]):
    uo, vo, us, vs = uuvv
    power_maintain_speed(uo=uo, vo=vo, us=us, vs=vs, coeff=1.0)