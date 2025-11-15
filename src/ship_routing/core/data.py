import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Callable

from functools import lru_cache
from .config import MAX_CACHE_SIZE
from .hashable_dataset import HashableDataset, make_hashable


def load_currents(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uo_name: str = "uo",
    vo_name: str = "vo",
    **kwargs,
) -> HashableDataset:
    ds = xr.open_dataset(data_file, **kwargs)
    ds = ds.rename(
        {
            lon_name: "lon",
            lat_name: "lat",
            time_name: "time",
            uo_name: "uo",
            vo_name: "vo",
        }
    )
    return make_hashable(ds)


def load_winds(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uw_name: str = "eastward_wind",
    vw_name: str = "northward_wind",
    **kwargs,
) -> HashableDataset:
    ds = xr.open_dataset(data_file, **kwargs)
    ds = ds.rename(
        {
            lon_name: "lon",
            lat_name: "lat",
            time_name: "time",
            uw_name: "uw",
            vw_name: "vw",
        }
    )
    return make_hashable(ds)


def load_waves(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    wh_name: str = "VHM0",
    **kwargs,
) -> HashableDataset:
    ds = xr.open_dataset(data_file, **kwargs)
    ds = ds.rename(
        {
            lon_name: "lon",
            lat_name: "lat",
            time_name: "time",
            wh_name: "wh",
        }
    )
    return make_hashable(ds)


def load_and_filter_forcing(
    path: str | None,
    loader: Callable[..., HashableDataset],
    time_start: np.datetime64,
    time_end: np.datetime64,
    engine: str | None = None,
    chunks: dict | None = None,
    load_eagerly: bool = False,
) -> HashableDataset | None:
    """Load dataset and filter to time period of interest.

    Parameters
    ----------
    path : str | None
        Path to the data file. If None, returns None.
    loader : Callable
        Function to load the dataset (e.g., load_currents, load_waves, load_winds).
    time_start : np.datetime64
        Start time for filtering.
    time_end : np.datetime64
        End time for filtering.
    engine : str | None
        Engine to use for opening the dataset (passed to xr.open_dataset).
    chunks : dict | None
        Chunking specification (passed to xr.open_dataset).
    load_eagerly : bool
        If True, load the dataset into memory. Default is False.

    Returns
    -------
    HashableDataset | None
        Filtered dataset, or None if path is None.
    """
    if not path:
        return None

    # Build kwargs for the loader
    kwargs = {}
    if engine is not None:
        kwargs["engine"] = engine
    if chunks is not None:
        kwargs["chunks"] = chunks

    ds = loader(data_file=path, **kwargs)

    # Calculate maximum time step for buffer
    max_time_step = ds.time.diff("time").max().load().data[()]

    # Create time mask with buffer
    time_mask = ((ds.time - max_time_step) >= time_start) & (
        (ds.time + max_time_step) <= time_end
    )

    # Filter time axis
    # Note: ds.where returns standard xr.Dataset instead of HashableDataset,
    # so we use this workaround to maintain the hashable type
    time_axis = ds.time.where(time_mask, drop=True).compute()
    ds = ds.sel(time=time_axis)

    if load_eagerly:
        ds = ds.load()

    return ds


@lru_cache(maxsize=MAX_CACHE_SIZE)
def _select_ij(
    ds: xr.Dataset = None,
    lon_start=None,
    lon_end=None,
    lat_start=None,
    lat_end=None,
):
    ds = ds.assign_coords(
        i=(("lon",), np.arange(ds.sizes["lon"])),
        j=(("lat",), np.arange(ds.sizes["lat"])),
    )
    i_start = ds.i.sel(lon=lon_start, method="nearest").data[()]
    j_start = ds.j.sel(lat=lat_start, method="nearest").data[()]
    i_end = ds.i.sel(lon=lon_end, method="nearest").data[()]
    j_end = ds.j.sel(lat=lat_end, method="nearest").data[()]

    n = max(abs(i_end - i_start), abs(j_start - j_end)) + 1
    i = xr.DataArray(
        np.round(np.linspace(i_start, i_end, n)).astype(int),
        name="i",
        dims=("along",),
        coords={"along": np.linspace(0, 1, n)},
    )
    j = xr.DataArray(
        np.round(np.linspace(j_start, j_end, n)).astype(int),
        name="j",
        dims=("along",),
        coords={"along": np.linspace(0, 1, n)},
    )

    return ds.isel(lon=i, lat=j)  # .compute()


@lru_cache(maxsize=MAX_CACHE_SIZE)
def _select_l(
    ds: xr.Dataset = None,
    time_start=None,
    time_end=None,
):
    ds = ds.assign_coords(
        l=(("time",), np.arange(ds.sizes["time"])),
    )
    l_start = ds.l.sel(time=time_start, method="nearest").data[()]
    l_end = ds.l.sel(time=time_end, method="nearest").data[()]
    n = ds.sizes["along"]
    l = xr.DataArray(
        np.round(np.linspace(l_start, l_end, n)).astype(int),
        name="l",
        dims=("along",),
        coords={"along": np.linspace(0, 1, n)},
    )
    return ds.isel(time=l)  # .compute()


@lru_cache(maxsize=MAX_CACHE_SIZE)
def select_data_for_leg(
    ds: xr.Dataset = None,
    lon_start=None,
    lon_end=None,
    lat_start=None,
    lat_end=None,
    time_start=None,
    time_end=None,
):
    return _select_l(
        ds=_select_ij(
            ds=ds,
            lon_start=lon_start,
            lat_start=lat_start,
            lon_end=lon_end,
            lat_end=lat_end,
        ),
        time_start=time_start,
        time_end=time_end,
    ).compute()
