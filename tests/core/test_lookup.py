"""Tests for the numpy lookup layer in ``ship_routing.core.lookup``.

The lookup layer exists only to be faster than going through
``Dataset.sel``/``Dataset.isel`` per leg, so the property that matters is that
it samples *exactly* the same values as
:func:`ship_routing.core.data.select_data_for_leg`. These tests check the two
against each other on the packaged test grids, including the nearest-neighbour
tie cases that a naive implementation gets wrong.
"""

import numpy as np
import pytest
import xarray as xr

from conftest import TEST_DATA_DIR

from ship_routing.core.data import load_currents, load_waves, load_winds, select_data_for_leg
from ship_routing.core.lookup import ForcingGrid, grid_for, nearest_index, sample_values_for_leg


@pytest.fixture(scope="module")
def currents():
    (path,) = sorted(TEST_DATA_DIR.glob("currents/*1deg_5day.nc"))
    return load_currents(data_file=path)


@pytest.fixture(scope="module")
def winds():
    (path,) = sorted(TEST_DATA_DIR.glob("winds/cmems_*.nc"))
    return load_winds(data_file=path)


@pytest.fixture(scope="module")
def waves():
    (path,) = sorted(TEST_DATA_DIR.glob("waves/cmems_*.nc"))
    return load_waves(data_file=path)


def _reference_index(ds, dim, value):
    """Nearest index along ``dim`` via xarray itself."""
    n = ds.sizes[dim]
    probe = ds.assign_coords(_idx=((dim,), np.arange(n)))
    return int(probe._idx.sel({dim: value}, method="nearest").data[()])


def test_nearest_index_matches_xarray_on_random_points(currents):
    """Random probes agree with DataArray.sel(method='nearest')."""
    rng = np.random.default_rng(0)
    lon = currents.lon.values
    lat = currents.lat.values
    for value in rng.uniform(lon.min(), lon.max(), 50):
        assert nearest_index(lon, value) == _reference_index(currents, "lon", value)
    for value in rng.uniform(lat.min(), lat.max(), 50):
        assert nearest_index(lat, value) == _reference_index(currents, "lat", value)


def test_nearest_index_matches_xarray_on_exact_midpoints(currents):
    """Exact midpoints are the tie case; xarray resolves them to the higher index."""
    lon = currents.lon.values
    midpoints = (lon[:-1] + lon[1:]) / 2.0
    for value in midpoints[:40]:
        assert nearest_index(lon, value) == _reference_index(currents, "lon", value)


def test_nearest_index_matches_xarray_on_time(currents):
    """Datetime axes work the same way as float axes."""
    time = currents.time.values
    for value in time:
        assert nearest_index(time, value) == _reference_index(currents, "time", value)


def test_nearest_index_clamps_outside_range(currents):
    """Values beyond either end select the corresponding edge."""
    lon = currents.lon.values
    assert nearest_index(lon, lon[0] - 1000.0) == 0
    assert nearest_index(lon, lon[-1] + 1000.0) == lon.size - 1


@pytest.mark.parametrize("dataset_name", ["currents", "winds", "waves"])
def test_sample_values_matches_select_data_for_leg(dataset_name, request):
    """Sampling reproduces the xarray selection bit for bit."""
    ds = request.getfixturevalue(dataset_name)
    rng = np.random.default_rng(17)

    lon = ds.lon.values
    lat = ds.lat.values
    time = ds.time.values
    span_seconds = float((time[-1] - time[0]) / np.timedelta64(1, "s"))

    for _ in range(25):
        lon_start, lon_end = rng.uniform(lon.min(), lon.max(), 2)
        lat_start, lat_end = rng.uniform(lat.min(), lat.max(), 2)
        offset = rng.uniform(0.0, span_seconds * 0.8)
        time_start = time[0] + np.timedelta64(int(offset), "s")
        time_end = time_start + np.timedelta64(int(span_seconds * 0.1), "s")

        kwargs = dict(
            lon_start=float(lon_start),
            lat_start=float(lat_start),
            time_start=time_start,
            lon_end=float(lon_end),
            lat_end=float(lat_end),
            time_end=time_end,
        )
        reference = select_data_for_leg(ds=ds, **kwargs)
        sampled = sample_values_for_leg(ds=ds, **kwargs)

        assert set(sampled) == set(ds.data_vars)
        for name in ds.data_vars:
            np.testing.assert_array_equal(
                sampled[name],
                reference[name].values,
                err_msg=f"{dataset_name}.{name} with {kwargs}",
            )


def test_sampled_arrays_are_readonly(currents):
    """Cached samples are shared, so callers must not be able to mutate them."""
    sampled = sample_values_for_leg(
        ds=currents,
        lon_start=float(currents.lon.values[1]),
        lat_start=float(currents.lat.values[1]),
        time_start=currents.time.values[0],
        lon_end=float(currents.lon.values[-2]),
        lat_end=float(currents.lat.values[-2]),
        time_end=currents.time.values[-1],
    )
    for values in sampled.values():
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 0.0


def test_grid_for_is_cached(currents):
    """The dataset is unpacked to numpy once, not per leg."""
    assert grid_for(currents) is grid_for(currents)


def test_grid_from_dataset_axes_match(currents):
    """Grid axes are the dataset's own coordinate values."""
    grid = ForcingGrid.from_dataset(currents)
    np.testing.assert_array_equal(grid.lon, currents.lon.values)
    np.testing.assert_array_equal(grid.lat, currents.lat.values)
    np.testing.assert_array_equal(grid.time, currents.time.values)
    assert set(grid.names) == set(currents.data_vars)


def test_grid_handles_transposed_dimension_order(currents):
    """Variables stored as (lon, lat, time) are normalised, not mis-sampled."""
    transposed = currents.transpose("lon", "lat", "time")
    grid = ForcingGrid.from_dataset(transposed)
    reference = ForcingGrid.from_dataset(currents)
    for name in reference.names:
        np.testing.assert_array_equal(
            grid.values[grid.names.index(name)],
            reference.values[reference.names.index(name)],
        )
