from ship_routing.core.cost import (
    align_along_track_arrays,
    maybe_cast_number_to_data_array,
    power_maintain_speed,
)


import xarray as xr
import numpy as np


def test_align_along_track_arrrays():
    a0 = xr.DataArray(
        list(range(7)),
        dims=("along",),
        coords={"along": np.linspace(0, 1, 7)},
    )
    a1 = xr.DataArray(
        list(range(1)),
        dims=("along",),
        coords={"along": np.linspace(0, 1, 1)},
    )
    a2 = xr.DataArray(
        list(range(17)),
        dims=("along",),
        coords={"along": np.linspace(0, 1, 17)},
    )
    a0a, a1a, a2a = align_along_track_arrays(a0, a1, a2)
    assert a0a.sizes["along"] == a1a.sizes["along"]
    assert a0a.sizes["along"] == a2a.sizes["along"]


def test_maybe_cast_number_to_data_array():
    assert "along" in maybe_cast_number_to_data_array(1).dims


def test_power_maintain_speed_just_call():
    power_maintain_speed()
