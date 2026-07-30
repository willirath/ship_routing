"""Along-track sampling of forcing fields, above the xarray ingest layer.

:mod:`ship_routing.core.data` loads and crops forcing datasets with xarray,
which is what gives us zarr support, coordinate handling and time selection for
free. That layer stays exactly as it is. What this module adds is a thin
indexing layer on top of it: once a dataset is loaded, its variables and
coordinate axes are extracted to plain numpy once, and every subsequent
per-leg sample is pure index arithmetic.

This matters because a leg samples only a few tens of points, so going through
``Dataset.sel`` / ``Dataset.isel`` / ``Dataset.compute`` per leg spends
essentially all of its time constructing xarray objects rather than reading
data.

There are two grids behind one interface, picked by :func:`grid_for` according
to whether the dataset is already in memory:

:class:`ForcingGrid`
    Dense. Unpacks the whole cropped cube to numpy once. Fastest per sample,
    but its footprint is the whole crop whether the routes visit it or not.
:class:`BlockCachedForcingGrid`
    Lazy. Tiles the domain into blocks and fetches only the blocks that routes
    actually land in. A representative population touches well under 1 % of the
    cells, so the resident set is one to two orders of magnitude smaller. This
    is what makes remote stores usable, and what leaves enough headroom to run
    several worker processes on one machine.

Both produce bit-identical samples -- they share :func:`leg_indices` and read
the same store.

The sampling reproduces :func:`ship_routing.core.data.select_data_for_leg`
exactly, including nearest-neighbour tie-breaking; see
``tests/core/test_lookup.py`` and ``tests/core/test_lookup_lazy.py``, which
check them against each other on real grids.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import xarray as xr

from .config import (
    BLOCK_CACHE_MAX_BYTES,
    BLOCK_FETCH_MAX_WORKERS,
    BLOCK_SHAPE_DEFAULT,
    MAX_CACHE_SIZE,
)

# Fallback for @profile decorator when not using line_profiler
try:
    profile
except NameError:

    def profile(func):
        return func


def nearest_index(axis: np.ndarray, value) -> int:
    """Index of the entry in ``axis`` nearest to ``value``.

    Reproduces ``DataArray.sel(dim=value, method="nearest")`` for a strictly
    increasing axis, including its tie-breaking: an exact midpoint resolves to
    the higher index.

    Parameters
    ----------
    axis : np.ndarray
        Strictly increasing coordinate values.
    value : scalar
        Value to look up. May be a float or a ``np.datetime64``.

    Returns
    -------
    int
        Index into ``axis``.
    """
    n = axis.size
    pos = int(np.searchsorted(axis, value))
    if pos == 0:
        return 0
    if pos >= n:
        return n - 1
    # Strict `<` puts exact ties on the higher index, matching pandas/xarray.
    return pos - 1 if (value - axis[pos - 1]) < (axis[pos] - value) else pos


def leg_indices(
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    lon_start: float,
    lat_start: float,
    time_start,
    lon_end: float,
    lat_end: float,
    time_end,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index triples ``(l, j, i)`` of the cells one leg samples.

    Mirrors :func:`ship_routing.core.data.select_data_for_leg`: the number of
    samples is set by the larger of the longitude and latitude index spans, and
    the time axis is walked in step with them.

    Shared by :class:`ForcingGrid` and :class:`BlockCachedForcingGrid` so the two
    cannot drift apart.

    Returns
    -------
    tuple of np.ndarray
        Time, latitude and longitude indices, all of the same length.
    """
    i_start = nearest_index(lon, lon_start)
    i_end = nearest_index(lon, lon_end)
    j_start = nearest_index(lat, lat_start)
    j_end = nearest_index(lat, lat_end)

    n = max(abs(i_end - i_start), abs(j_start - j_end)) + 1

    i = np.round(np.linspace(i_start, i_end, n)).astype(int)
    j = np.round(np.linspace(j_start, j_end, n)).astype(int)

    l_start = nearest_index(time, time_start)
    l_end = nearest_index(time, time_end)
    l = np.round(np.linspace(l_start, l_end, n)).astype(int)

    return l, j, i


class SampleCacheMixin:
    """Per-leg sample cache keyed on grid *indices* rather than coordinates.

    The samples a leg produces depend only on the cell indices it snaps to, so
    the indices are the canonical key: two legs whose endpoints differ slightly
    but round to the same cells are the same lookup. Keying on the raw floats
    misses those -- measured on a representative run, 30-37 % of the distinct
    float keys collapse to a shared index key.

    Canonicalising is free here because the indices have to be computed anyway
    to do the sampling. It also keeps the dataset out of the key: the cache
    lives on the grid, and there is one grid per dataset.
    """

    def _sample_key(self, l: np.ndarray, j: np.ndarray, i: np.ndarray) -> tuple:
        """Key identifying a sample. The endpoints determine the whole path."""
        return (
            int(l[0]),
            int(l[-1]),
            int(j[0]),
            int(j[-1]),
            int(i[0]),
            int(i[-1]),
        )

    def sample_leg_cached(
        self,
        lon_start: float,
        lat_start: float,
        time_start,
        lon_end: float,
        lat_end: float,
        time_end,
    ) -> dict[str, np.ndarray]:
        """Cached :meth:`sample_leg`. Returns the same arrays, never copies."""
        l, j, i = leg_indices(
            lon=self.lon,
            lat=self.lat,
            time=self.time,
            lon_start=lon_start,
            lat_start=lat_start,
            time_start=time_start,
            lon_end=lon_end,
            lat_end=lat_end,
            time_end=time_end,
        )
        key = self._sample_key(l, j, i)
        cache = self._sample_cache
        sampled = cache.get(key)
        if sampled is not None:
            cache.move_to_end(key)
            return sampled
        sampled = self._gather(l, j, i)
        cache[key] = sampled
        if len(cache) > MAX_CACHE_SIZE:
            cache.popitem(last=False)
        return sampled


@dataclass(frozen=True)
class ForcingGrid(SampleCacheMixin):
    """Raw numpy view of a forcing dataset, for fast along-track sampling.

    Parameters
    ----------
    names : tuple[str, ...]
        Data variable names, in the order they appear in ``values``.
    values : tuple[np.ndarray, ...]
        One ``(time, lat, lon)`` array per data variable.
    lon, lat, time : np.ndarray
        Strictly increasing coordinate axes.
    """

    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    lon: np.ndarray
    lat: np.ndarray
    time: np.ndarray
    _sample_cache: "OrderedDict[tuple, dict]" = field(
        default_factory=OrderedDict, compare=False, repr=False
    )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "ForcingGrid":
        """Extract a grid from a loaded dataset.

        The dataset keeps ownership of the data; this is a view onto it, so it
        is only valid as long as the dataset is unchanged. Datasets are frozen
        once loaded, which is what makes caching on dataset identity sound.
        """
        names = tuple(ds.data_vars)
        values = tuple(
            np.ascontiguousarray(ds[name].transpose("time", "lat", "lon").values)
            for name in names
        )
        return cls(
            names=names,
            values=values,
            lon=ds.lon.values,
            lat=ds.lat.values,
            time=ds.time.values,
        )

    def sample_leg(
        self,
        lon_start: float,
        lat_start: float,
        time_start,
        lon_end: float,
        lat_end: float,
        time_end,
    ) -> dict[str, np.ndarray]:
        """Sample all variables along one leg.

        Mirrors :func:`ship_routing.core.data.select_data_for_leg`: the number
        of samples is set by the larger of the longitude and latitude index
        spans, and the time axis is walked in step with them.

        Returns
        -------
        dict[str, np.ndarray]
            One 1-D array per data variable, all of the same length.
        """
        l, j, i = leg_indices(
            lon=self.lon,
            lat=self.lat,
            time=self.time,
            lon_start=lon_start,
            lat_start=lat_start,
            time_start=time_start,
            lon_end=lon_end,
            lat_end=lat_end,
            time_end=time_end,
        )
        return self._gather(l, j, i)

    def _gather(
        self, l: np.ndarray, j: np.ndarray, i: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Pull the values at the given index triples out of the dense arrays."""
        sampled = {}
        for name, values in zip(self.names, self.values):
            out = values[l, j, i]
            # Results are cached and shared between callers, so freeze them.
            out.flags.writeable = False
            sampled[name] = out
        return sampled


class BlockCachedForcingGrid(SampleCacheMixin):
    """Lazy counterpart of :class:`ForcingGrid`, backed by a block cache.

    Serves the same ``sample_leg`` interface, but never materialises the whole
    cube. Instead the domain is tiled into ``block_shape`` blocks; a sample
    fetches only the blocks it lands in and keeps them in an LRU cache.

    This is worthwhile because routes touch a very small fraction of the domain
    -- for a representative 32-member population, 0.02-0.23 % of the cells --
    so the resident working set is one to two orders of magnitude smaller than
    the eagerly cropped cube. It is also the only workable approach for remote
    stores, where eager loading is not an option.

    The blocks are *read* units, not storage chunks: a block may span several
    stored chunks. Because the cost of a read is dominated by a fixed per-call
    overhead plus decompression, a block larger than the stored chunk amortises
    the former at the price of the latter. Whatever the block shape, the store
    still has to decompress whole chunks, so the win only materialises if the
    data is stored in chunks no larger than the block.

    Parameters
    ----------
    ds : xr.Dataset
        Lazily opened dataset, as returned by :mod:`ship_routing.core.data` with
        ``load_eagerly=False``.
    block_shape : tuple of int, optional
        ``(time, lat, lon)`` extent of one cache block. Defaults to
        :data:`ship_routing.core.config.BLOCK_SHAPE_DEFAULT`.
    max_bytes : int, optional
        Byte budget for the cache; least-recently-used blocks are evicted past
        it. Defaults to :data:`ship_routing.core.config.BLOCK_CACHE_MAX_BYTES`.
    max_workers : int, optional
        Threads used to fetch several missing blocks at once. Reads release the
        GIL while decompressing and while waiting on the network, so this helps
        both locally and -- much more so -- against remote stores.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        block_shape: tuple[int, int, int] = None,
        max_bytes: int = None,
        max_workers: int = None,
    ):
        self._ds = ds.transpose("time", "lat", "lon")
        self.names = tuple(ds.data_vars)
        self.lon = ds.lon.values
        self.lat = ds.lat.values
        self.time = ds.time.values
        self.shape = (self.time.size, self.lat.size, self.lon.size)
        self._dtypes = tuple(ds[name].dtype for name in self.names)

        self.block_shape = tuple(
            int(min(b, s))
            for b, s in zip(block_shape or BLOCK_SHAPE_DEFAULT, self.shape)
        )
        self._n_blocks = tuple(
            -(-s // b) for s, b in zip(self.shape, self.block_shape)
        )
        self._max_bytes = (
            BLOCK_CACHE_MAX_BYTES if max_bytes is None else int(max_bytes)
        )
        self._max_workers = (
            BLOCK_FETCH_MAX_WORKERS if max_workers is None else int(max_workers)
        )

        self._cache: OrderedDict[int, tuple[np.ndarray, ...]] = OrderedDict()
        self._sample_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._cache_bytes = 0
        self._executor = None
        self.hits = 0
        self.misses = 0
        LIVE_BLOCK_CACHED_GRIDS.add(self)

    # -- block bookkeeping -------------------------------------------------

    def _decode(self, code: int) -> tuple[int, int, int]:
        """Unpack a flat block code into ``(time, lat, lon)`` block indices."""
        n_lat, n_lon = self._n_blocks[1], self._n_blocks[2]
        b_lon = code % n_lon
        rest = code // n_lon
        return rest // n_lat, rest % n_lat, b_lon

    def _fetch(self, code: int) -> tuple[np.ndarray, ...]:
        """Read one block, all variables at once, from the lazy dataset."""
        b_time, b_lat, b_lon = self._decode(code)
        bt, by, bx = self.block_shape
        nt, ny, nx = self.shape
        sub = self._ds.isel(
            time=slice(b_time * bt, min((b_time + 1) * bt, nt)),
            lat=slice(b_lat * by, min((b_lat + 1) * by, ny)),
            lon=slice(b_lon * bx, min((b_lon + 1) * bx, nx)),
        ).load()
        return tuple(np.ascontiguousarray(sub[name].values) for name in self.names)

    def _store(self, code: int, block: tuple[np.ndarray, ...]) -> None:
        self._cache[code] = block
        self._cache_bytes += sum(a.nbytes for a in block)
        while self._cache_bytes > self._max_bytes and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= sum(a.nbytes for a in evicted)

    def _ensure(self, codes: list[int]) -> dict[int, tuple[np.ndarray, ...]]:
        """Fetch whatever is missing and return every requested block.

        Returns the blocks rather than leaving the caller to read them back out
        of the cache, so a leg that needs more blocks than the byte budget holds
        still gets consistent data instead of a ``KeyError``.
        """
        missing = [c for c in codes if c not in self._cache]
        self.hits += len(codes) - len(missing)
        self.misses += len(missing)

        resident = {}
        for code in codes:
            block = self._cache.get(code)
            if block is not None:
                self._cache.move_to_end(code)
                resident[code] = block
        if not missing:
            return resident

        if len(missing) > 1 and self._max_workers > 1:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            blocks = list(self._executor.map(self._fetch, missing))
        else:
            blocks = [self._fetch(c) for c in missing]
        for code, block in zip(missing, blocks):
            self._store(code, block)
            resident[code] = block
        return resident

    # -- sampling ----------------------------------------------------------

    def sample_leg(
        self,
        lon_start: float,
        lat_start: float,
        time_start,
        lon_end: float,
        lat_end: float,
        time_end,
    ) -> dict[str, np.ndarray]:
        """Sample all variables along one leg.

        Bit-identical to :meth:`ForcingGrid.sample_leg` on the same dataset --
        the index arithmetic is shared and the values come from the same store.

        Returns
        -------
        dict[str, np.ndarray]
            One 1-D array per data variable, all of the same length.
        """
        l, j, i = leg_indices(
            lon=self.lon,
            lat=self.lat,
            time=self.time,
            lon_start=lon_start,
            lat_start=lat_start,
            time_start=time_start,
            lon_end=lon_end,
            lat_end=lat_end,
            time_end=time_end,
        )
        return self._gather(l, j, i)

    def _gather(
        self, l: np.ndarray, j: np.ndarray, i: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Fetch whichever blocks the index triples land in, then read them."""
        bt, by, bx = self.block_shape
        n_lat, n_lon = self._n_blocks[1], self._n_blocks[2]
        codes = ((l // bt) * n_lat + (j // by)) * n_lon + (i // bx)

        # Legs are short, so they usually land in a single block; skip the
        # grouping machinery in that common case.
        first = int(codes[0])
        if bool((codes == first).all()):
            block = self._ensure([first])[first]
            b_time, b_lat, b_lon = self._decode(first)
            dl = l - b_time * bt
            dj = j - b_lat * by
            di = i - b_lon * bx
            sampled = {}
            for name, values in zip(self.names, block):
                out = values[dl, dj, di]
                out.flags.writeable = False
                sampled[name] = out
            return sampled

        unique_codes, inverse = np.unique(codes, return_inverse=True)
        code_list = [int(c) for c in unique_codes]
        blocks = self._ensure(code_list)

        out = [np.empty(l.size, dtype=dtype) for dtype in self._dtypes]
        for position, code in enumerate(code_list):
            block = blocks[code]
            b_time, b_lat, b_lon = self._decode(code)
            mask = inverse == position
            dl = l[mask] - b_time * bt
            dj = j[mask] - b_lat * by
            di = i[mask] - b_lon * bx
            for k in range(len(self.names)):
                out[k][mask] = block[k][dl, dj, di]

        sampled = {}
        for name, values in zip(self.names, out):
            # Results are cached and shared between callers, so freeze them.
            values.flags.writeable = False
            sampled[name] = values
        return sampled

    # -- introspection -----------------------------------------------------

    @property
    def cache_bytes(self) -> int:
        """Bytes currently held in the block cache."""
        return self._cache_bytes

    @property
    def n_blocks_resident(self) -> int:
        """Number of blocks currently held in the block cache."""
        return len(self._cache)


#: Live block-cached grids, for diagnostics (see ``dev/profiling/profile_run.py``).
#: Weak, so holding this registry never keeps a dataset alive.
LIVE_BLOCK_CACHED_GRIDS: "weakref.WeakSet[BlockCachedForcingGrid]" = weakref.WeakSet()


def dataset_is_in_memory(ds: xr.Dataset) -> bool:
    """Whether every data variable of ``ds`` is already backed by numpy.

    Lazily opened datasets -- dask-backed or wrapped in xarray's lazy indexing
    adapters -- return ``False``, which is what selects the block-cached grid.
    """
    return all(ds[name].variable._in_memory for name in ds.data_vars)


@lru_cache(maxsize=32)
def grid_for(ds: xr.Dataset):
    """Return the cached grid for a dataset.

    Returns a :class:`ForcingGrid` for datasets already in memory and a
    :class:`BlockCachedForcingGrid` for lazily opened ones, so callers get the
    same ``sample_leg`` interface either way.

    Keyed on dataset identity via
    :class:`ship_routing.core.hashable_dataset.HashableDataset`, so each loaded
    forcing dataset is unpacked -- or given a block cache -- exactly once.
    """
    if dataset_is_in_memory(ds):
        return ForcingGrid.from_dataset(ds)
    return BlockCachedForcingGrid(ds)


@profile
def sample_values_for_leg(
    ds: xr.Dataset = None,
    lon_start=None,
    lon_end=None,
    lat_start=None,
    lat_end=None,
    time_start=None,
    time_end=None,
) -> dict[str, np.ndarray]:
    """Sample forcing variables along a leg, as plain numpy arrays.

    Fast-path equivalent of
    :func:`ship_routing.core.data.select_data_for_leg`, returning raw arrays
    instead of a :class:`xarray.Dataset`.

    Caching happens one level down, on the grid, keyed on the cell indices the
    leg snaps to -- see :class:`SampleCacheMixin`. Caching here instead would
    key on the raw coordinates, which is both a worse key and one that has to
    carry the dataset around with it.

    Parameters
    ----------
    ds : xr.Dataset
        Environmental dataset, as loaded by :mod:`ship_routing.core.data`.
    lon_start, lat_start : float
        Start position.
    lon_end, lat_end : float
        End position.
    time_start, time_end : datetime-like
        Start and end times.

    Returns
    -------
    dict[str, np.ndarray]
        One 1-D array per data variable.
    """
    return grid_for(ds).sample_leg_cached(
        lon_start=lon_start,
        lat_start=lat_start,
        time_start=time_start,
        lon_end=lon_end,
        lat_end=lat_end,
        time_end=time_end,
    )
