"""Tests for the numpy-native along-track cost path in ``ship_routing.core.cost``.

The fast path exists purely for speed, so what matters is that it reproduces the
xarray reference implementation exactly. In particular the nearest-neighbour
tie-breaking rule must match ``DataArray.sel(method="nearest")``: closed-form
index arithmetic (``round(k * (n_src - 1) / (n_ref - 1))``) does *not*, because
``np.linspace`` values do not sit exactly on the analytic midpoints.
"""

import numpy as np
import pytest
import xarray as xr

from ship_routing.core.cost import (
    align_along_track_arrays,
    align_along_track_values,
    hazard_conditions_wave_height,
    maybe_cast_number_to_values,
    nearest_along_index,
    power_and_hazard_values,
    power_maintain_speed,
)


def _reference_nearest(n_src: int, n_ref: int) -> np.ndarray:
    """Nearest index map via xarray itself."""
    src = np.linspace(0.0, 1.0, n_src)
    ref = np.linspace(0.0, 1.0, n_ref)
    da = xr.DataArray(
        np.arange(n_src, dtype=float), dims=("along",), coords={"along": src}
    )
    return da.sel(along=ref, method="nearest").values.astype(int)


@pytest.mark.parametrize("n_ref", [2, 3, 5, 8, 13, 24, 47, 61])
@pytest.mark.parametrize("n_src", [2, 3, 4, 7, 12, 23, 46, 60])
def test_nearest_along_index_matches_xarray(n_src, n_ref):
    """Index map reproduces DataArray.sel(method='nearest') exactly."""
    np.testing.assert_array_equal(
        nearest_along_index(n_src, n_ref), _reference_nearest(n_src, n_ref)
    )


def test_nearest_along_index_matches_xarray_exhaustively():
    """Sweep all small size pairs, including the exact-tie cases."""
    for n_ref in range(2, 40):
        for n_src in range(2, 40):
            np.testing.assert_array_equal(
                nearest_along_index(n_src, n_ref),
                _reference_nearest(n_src, n_ref),
                err_msg=f"n_src={n_src}, n_ref={n_ref}",
            )


def test_nearest_along_index_is_readonly():
    """Cached index arrays must not be mutable by callers."""
    idx = nearest_along_index(5, 9)
    assert not idx.flags.writeable
    with pytest.raises(ValueError):
        idx[0] = 99


def test_closed_form_rounding_would_be_wrong():
    """Guard the reason this module does not use closed-form index arithmetic."""
    n_src, n_ref = 2, 3
    closed_form = np.round(np.arange(n_ref) * (n_src - 1) / (n_ref - 1)).astype(int)
    assert not np.array_equal(closed_form, _reference_nearest(n_src, n_ref))


def test_maybe_cast_number_to_values_scalar():
    """Scalars become two-element arrays, matching the DataArray helper."""
    np.testing.assert_array_equal(maybe_cast_number_to_values(3.0), np.array([3.0, 3.0]))
    np.testing.assert_array_equal(
        maybe_cast_number_to_values(np.float64(2.5)), np.array([2.5, 2.5])
    )


def test_maybe_cast_number_to_values_passes_arrays_through():
    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(maybe_cast_number_to_values(values), values)
    da = xr.DataArray(values, dims=("along",), coords={"along": np.linspace(0, 1, 3)})
    np.testing.assert_array_equal(maybe_cast_number_to_values(da), values)


def _data_array(values):
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values,
        dims=("along",),
        coords={"along": np.linspace(0.0, 1.0, values.size)},
    )


def test_align_along_track_values_matches_array_version():
    """Numpy alignment reproduces the xarray alignment element for element."""
    rng = np.random.default_rng(0)
    for sizes in [(3, 7, 5), (2, 2, 2), (11, 4, 23), (46, 13, 8)]:
        raw = [rng.normal(size=n) for n in sizes]
        from_values = align_along_track_values(*[np.asarray(r) for r in raw])
        from_arrays = align_along_track_arrays(*[_data_array(r) for r in raw])
        for got, want in zip(from_values, from_arrays):
            np.testing.assert_array_equal(got, want.values)


def test_power_and_hazard_values_match_reference_implementation():
    """The combined fast path equals the two xarray functions it replaces."""
    rng = np.random.default_rng(42)
    for n_cur, n_wind, n_wave in [(24, 47, 13), (5, 5, 5), (61, 9, 30)]:
        u_cur = _data_array(rng.normal(scale=0.5, size=n_cur))
        v_cur = _data_array(rng.normal(scale=0.5, size=n_cur))
        u_wind = _data_array(rng.normal(scale=8.0, size=n_wind))
        v_wind = _data_array(rng.normal(scale=8.0, size=n_wind))
        wave = _data_array(np.abs(rng.normal(scale=3.0, size=n_wave)))
        u_ship, v_ship = 4.2, -3.1

        kwargs = dict(
            u_current_ms=u_cur,
            v_current_ms=v_cur,
            u_wind_ms=u_wind,
            v_wind_ms=v_wind,
            w_wave_height=wave,
            u_ship_og_ms=u_ship,
            v_ship_og_ms=v_ship,
        )
        power, hazard = power_and_hazard_values(**kwargs)

        np.testing.assert_array_equal(power, power_maintain_speed(**kwargs).values)
        np.testing.assert_array_equal(
            hazard, hazard_conditions_wave_height(**kwargs).values
        )


def test_power_and_hazard_values_handles_missing_forcing():
    """Scalar zeros stand in for absent datasets, as in Leg.cost_through."""
    wave = _data_array([1.0, 2.0, 3.0, 4.0])
    power, hazard = power_and_hazard_values(
        u_current_ms=0,
        v_current_ms=0,
        u_wind_ms=0,
        v_wind_ms=0,
        w_wave_height=wave,
        u_ship_og_ms=5.0,
        v_ship_og_ms=0.0,
    )
    assert power.shape == (4,)
    assert hazard.shape == (4,)
    assert np.all(np.isfinite(power))


def test_power_and_hazard_values_propagates_nan():
    """NaN forcing must survive into the power array (Leg.cost_through relies on it)."""
    wave = _data_array([1.0, np.nan, 3.0])
    power, _ = power_and_hazard_values(
        w_wave_height=wave, u_ship_og_ms=5.0, v_ship_og_ms=0.0
    )
    assert np.isnan(power).any()
