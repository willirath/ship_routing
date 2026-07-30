"""Tests for the block-cached lazy grid in ``ship_routing.core.lookup``.

:class:`BlockCachedForcingGrid` exists to sample a lazily opened dataset without
materialising it. The property that matters is that it is indistinguishable from
the dense :class:`ForcingGrid` -- same values, same dtypes, same shapes -- no
matter how the domain is tiled into blocks or how hard the cache is thrashed.

These run on the packaged test grids, opened with ``load_eagerly=False``.
"""

import numpy as np
import pytest

from conftest import TEST_DATA_DIR

from ship_routing.core.data import load_currents, load_waves, load_winds, select_data_for_leg
from ship_routing.core.hashable_dataset import make_hashable
from ship_routing.core.lookup import (
    BlockCachedForcingGrid,
    ForcingGrid,
    dataset_is_in_memory,
    grid_for,
    leg_indices,
)


def _paths():
    (currents,) = sorted(TEST_DATA_DIR.glob("currents/*1deg_5day.nc"))
    (winds,) = sorted(TEST_DATA_DIR.glob("winds/cmems_*.nc"))
    (waves,) = sorted(TEST_DATA_DIR.glob("waves/cmems_*.nc"))
    return currents, winds, waves


@pytest.fixture(scope="module")
def eager():
    currents, winds, waves = _paths()
    return {
        "currents": load_currents(data_file=currents),
        "winds": load_winds(data_file=winds),
        "waves": load_waves(data_file=waves),
    }


@pytest.fixture(scope="module")
def lazy():
    currents, winds, waves = _paths()
    return {
        "currents": load_currents(data_file=currents, load_eagerly=False),
        "winds": load_winds(data_file=winds, load_eagerly=False),
        "waves": load_waves(data_file=waves, load_eagerly=False),
    }


def _legs(ds, n=40, seed=0):
    """Random legs spanning the dataset's domain."""
    rng = np.random.default_rng(seed)
    lon, lat, time = ds.lon.values, ds.lat.values, ds.time.values
    out = []
    for _ in range(n):
        out.append(
            dict(
                lon_start=float(rng.uniform(lon.min(), lon.max())),
                lat_start=float(rng.uniform(lat.min(), lat.max())),
                lon_end=float(rng.uniform(lon.min(), lon.max())),
                lat_end=float(rng.uniform(lat.min(), lat.max())),
                time_start=time[rng.integers(0, time.size)],
                time_end=time[rng.integers(0, time.size)],
            )
        )
    return out


def _assert_same(dense_sample, lazy_sample):
    assert dense_sample.keys() == lazy_sample.keys()
    for name in dense_sample:
        expected, got = dense_sample[name], lazy_sample[name]
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        assert np.array_equal(got, expected, equal_nan=True)


# -- dispatch -------------------------------------------------------------


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_eager_datasets_are_reported_in_memory(eager, store):
    assert dataset_is_in_memory(eager[store])


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_lazy_datasets_are_not_reported_in_memory(lazy, store):
    assert not dataset_is_in_memory(lazy[store])


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_grid_for_dispatches_on_whether_data_is_in_memory(eager, lazy, store):
    assert isinstance(grid_for(eager[store]), ForcingGrid)
    assert isinstance(grid_for(lazy[store]), BlockCachedForcingGrid)


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_grid_for_caches_per_dataset(lazy, store):
    assert grid_for(lazy[store]) is grid_for(lazy[store])


# -- equivalence with the dense grid --------------------------------------


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
@pytest.mark.parametrize("block_shape", [(8, 32, 32), (4, 16, 16), (2, 3, 5), (1, 1, 1)])
def test_matches_dense_grid_for_any_block_shape(eager, lazy, store, block_shape):
    """Tiling is an implementation detail; the samples must not depend on it."""
    dense = ForcingGrid.from_dataset(eager[store])
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=block_shape)
    for leg in _legs(eager[store]):
        _assert_same(dense.sample_leg(**leg), block_cached.sample_leg(**leg))


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_matches_dense_grid_when_block_is_larger_than_domain(eager, lazy, store):
    """A block bigger than the cube collapses to a single block covering it."""
    dense = ForcingGrid.from_dataset(eager[store])
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=(10_000,) * 3)
    assert block_cached.block_shape == dense.values[0].shape
    for leg in _legs(eager[store]):
        _assert_same(dense.sample_leg(**leg), block_cached.sample_leg(**leg))


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_matches_select_data_for_leg(eager, lazy, store):
    """End-to-end against the original xarray implementation."""
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=(2, 4, 4))
    for leg in _legs(eager[store], n=15):
        reference = select_data_for_leg(ds=eager[store], **leg)
        sampled = block_cached.sample_leg(**leg)
        for name in reference.data_vars:
            assert np.array_equal(
                sampled[name], reference[name].values, equal_nan=True
            )


# -- cache behaviour -------------------------------------------------------


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_eviction_does_not_change_results(eager, lazy, store):
    """A budget too small to hold even one leg's blocks must still be correct."""
    dense = ForcingGrid.from_dataset(eager[store])
    block_cached = BlockCachedForcingGrid(
        lazy[store], block_shape=(1, 2, 2), max_bytes=1
    )
    for leg in _legs(eager[store], n=20):
        _assert_same(dense.sample_leg(**leg), block_cached.sample_leg(**leg))
    assert block_cached.n_blocks_resident == 1
    assert block_cached.misses > 0


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_repeated_sampling_is_served_from_cache(lazy, store):
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=(4, 8, 8))
    (leg,) = _legs(lazy[store], n=1)
    block_cached.sample_leg(**leg)
    misses_after_first = block_cached.misses
    for _ in range(5):
        block_cached.sample_leg(**leg)
    assert block_cached.misses == misses_after_first
    assert block_cached.hits > 0


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_cache_stays_within_budget(eager, lazy, store):
    budget = 64 * 1024
    block_cached = BlockCachedForcingGrid(
        lazy[store], block_shape=(2, 8, 8), max_bytes=budget
    )
    for leg in _legs(eager[store], n=30):
        block_cached.sample_leg(**leg)
        # One block is always kept, so the budget can be exceeded by at most that.
        assert block_cached.cache_bytes <= budget or block_cached.n_blocks_resident == 1


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_threaded_and_sequential_fetching_agree(eager, lazy, store):
    dense = ForcingGrid.from_dataset(eager[store])
    sequential = BlockCachedForcingGrid(
        lazy[store], block_shape=(2, 4, 4), max_workers=1
    )
    threaded = BlockCachedForcingGrid(lazy[store], block_shape=(2, 4, 4), max_workers=8)
    for leg in _legs(eager[store], n=20):
        expected = dense.sample_leg(**leg)
        _assert_same(expected, sequential.sample_leg(**leg))
        _assert_same(expected, threaded.sample_leg(**leg))


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_samples_are_readonly(lazy, store):
    """Samples are shared between callers, so they must not be mutable."""
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=(2, 4, 4))
    for leg in _legs(lazy[store], n=5):
        for values in block_cached.sample_leg(**leg).values():
            assert not values.flags.writeable


# -- index-keyed sample cache ---------------------------------------------


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_cached_sampling_matches_uncached(eager, lazy, store):
    dense = ForcingGrid.from_dataset(eager[store])
    block_cached = BlockCachedForcingGrid(lazy[store], block_shape=(2, 4, 4))
    for leg in _legs(eager[store], n=25):
        expected = dense.sample_leg(**leg)
        _assert_same(expected, dense.sample_leg_cached(**leg))
        _assert_same(expected, block_cached.sample_leg_cached(**leg))


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_legs_snapping_to_the_same_cells_share_a_cache_entry(eager, store):
    """The point of keying on indices: nearby endpoints are the same lookup."""
    ds = eager[store]
    grid = ForcingGrid.from_dataset(ds)
    lon, lat, time = ds.lon.values, ds.lat.values, ds.time.values
    # Nudge the endpoints by a fraction of a cell, well inside the snap radius.
    d_lon = abs(lon[1] - lon[0]) / 8.0
    d_lat = abs(lat[1] - lat[0]) / 8.0
    leg = dict(
        lon_start=float(lon[2]),
        lat_start=float(lat[2]),
        time_start=time[0],
        lon_end=float(lon[5]),
        lat_end=float(lat[4]),
        time_end=time[-1],
    )
    nudged = dict(leg)
    nudged["lon_start"] += d_lon
    nudged["lat_start"] -= d_lat
    nudged["lon_end"] -= d_lon
    nudged["lat_end"] += d_lat

    first = grid.sample_leg_cached(**leg)
    assert len(grid._sample_cache) == 1
    second = grid.sample_leg_cached(**nudged)
    assert len(grid._sample_cache) == 1, "nudged endpoints should reuse the entry"
    for name in first:
        assert second[name] is first[name]


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_distinct_cells_do_not_share_a_cache_entry(eager, store):
    ds = eager[store]
    grid = ForcingGrid.from_dataset(ds)
    lon, lat, time = ds.lon.values, ds.lat.values, ds.time.values
    base = dict(
        lon_start=float(lon[2]),
        lat_start=float(lat[2]),
        time_start=time[0],
        lon_end=float(lon[5]),
        lat_end=float(lat[4]),
        time_end=time[-1],
    )
    grid.sample_leg_cached(**base)
    moved = dict(base, lon_end=float(lon[6]))
    grid.sample_leg_cached(**moved)
    assert len(grid._sample_cache) == 2


def test_sample_cache_is_not_part_of_grid_equality(eager):
    """The cache is incidental state; two grids over the same data are equal."""
    first = ForcingGrid.from_dataset(eager["currents"])
    second = ForcingGrid.from_dataset(eager["currents"])
    (leg,) = _legs(eager["currents"], n=1)
    first.sample_leg_cached(**leg)
    assert len(first._sample_cache) == 1
    assert len(second._sample_cache) == 0
    assert first.names == second.names


# -- shared index arithmetic ----------------------------------------------


@pytest.mark.parametrize("store", ["currents", "winds", "waves"])
def test_both_grids_use_the_same_indices(eager, lazy, store):
    """The two grids must not be able to drift apart on index arithmetic."""
    ds = eager[store]
    block_cached = BlockCachedForcingGrid(lazy[store])
    for leg in _legs(ds, n=20):
        l, j, i = leg_indices(
            lon=ds.lon.values, lat=ds.lat.values, time=ds.time.values, **leg
        )
        assert l.shape == j.shape == i.shape
        sampled = block_cached.sample_leg(**leg)
        for values in sampled.values():
            assert values.shape == l.shape


def test_block_codes_round_trip_through_decode(lazy):
    """Flat block codes must unpack back to the block they came from."""
    block_cached = BlockCachedForcingGrid(lazy["currents"], block_shape=(2, 3, 4))
    n_time, n_lat, n_lon = block_cached._n_blocks
    for b_time in range(n_time):
        for b_lat in range(n_lat):
            for b_lon in range(n_lon):
                code = (b_time * n_lat + b_lat) * n_lon + b_lon
                assert block_cached._decode(code) == (b_time, b_lat, b_lon)
