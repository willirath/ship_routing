import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Callable

from functools import lru_cache
from .config import MAX_CACHE_SIZE
from .hashable_dataset import HashableDataset, make_hashable

# Fallback for @profile decorator when not using line_profiler
try:
    profile
except NameError:

    def profile(func):
        return func


@profile
def load_currents(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uo_name: str = "uo",
    vo_name: str = "vo",
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
    load_eagerly: bool = True,
    spatial_bounds: tuple = None,
    **kwargs,
) -> HashableDataset:
    """Load ocean current data from netCDF file or other formats.

    Loads current velocity data, renames variables to standard names,
    filters to time period of interest, and returns a hashable dataset.

    Parameters
    ----------
    data_file : Path
        Path to file containing current data (netCDF by default; other formats
        supported via engine kwarg in **kwargs, e.g., zarr, grib, hdf5)
    lon_name : str, default="longitude"
        Name of longitude variable in source file
    lat_name : str, default="latitude"
        Name of latitude variable in source file
    time_name : str, default="time"
        Name of time variable in source file
    uo_name : str, default="uo"
        Name of eastward current velocity variable in source file
    vo_name : str, default="vo"
        Name of northward current velocity variable in source file
    time_start : np.datetime64, optional
        Start time for filtering data
    time_end : np.datetime64, optional
        End time for filtering data
    load_eagerly : bool, default=True
        If True, load data into memory immediately
    spatial_bounds : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) for spatial cropping
    **kwargs
        Additional arguments passed to xr.open_dataset

    Returns
    -------
    HashableDataset
        Dataset with standardized variable names (lon, lat, time, uo, vo)
    """
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
    ds = _filter_times(ds, time_start=time_start, time_end=time_end)
    if spatial_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = spatial_bounds
        ds = _apply_spatial_selection(ds, lon_min, lon_max, lat_min, lat_max)
    if load_eagerly:
        ds = ds.load()
    return make_hashable(ds)


@profile
def load_winds(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uw_name: str = "eastward_wind",
    vw_name: str = "northward_wind",
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
    load_eagerly: bool = True,
    spatial_bounds: tuple = None,
    **kwargs,
) -> HashableDataset:
    """Load wind data from netCDF file or other formats.

    Loads wind velocity data, renames variables to standard names,
    filters to time period of interest, and returns a hashable dataset.

    Parameters
    ----------
    data_file : Path
        Path to file containing wind data (netCDF by default; other formats
        supported via engine kwarg in **kwargs, e.g., zarr, grib, hdf5)
    lon_name : str, default="longitude"
        Name of longitude variable in source file
    lat_name : str, default="latitude"
        Name of latitude variable in source file
    time_name : str, default="time"
        Name of time variable in source file
    uw_name : str, default="eastward_wind"
        Name of eastward wind velocity variable in source file
    vw_name : str, default="northward_wind"
        Name of northward wind velocity variable in source file
    time_start : np.datetime64, optional
        Start time for filtering data
    time_end : np.datetime64, optional
        End time for filtering data
    load_eagerly : bool, default=True
        If True, load data into memory immediately
    spatial_bounds : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) for spatial cropping
    **kwargs
        Additional arguments passed to xr.open_dataset

    Returns
    -------
    HashableDataset
        Dataset with standardized variable names (lon, lat, time, uw, vw)
    """
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
    ds = _filter_times(ds, time_start=time_start, time_end=time_end)
    if spatial_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = spatial_bounds
        ds = _apply_spatial_selection(ds, lon_min, lon_max, lat_min, lat_max)
    if load_eagerly:
        ds = ds.load()
    return make_hashable(ds)


@profile
def load_waves(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    wh_name: str = "VHM0",
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
    load_eagerly: bool = True,
    spatial_bounds: tuple = None,
    **kwargs,
) -> HashableDataset:
    """Load wave data from netCDF file or other formats.

    Loads significant wave height data, renames variables to standard names,
    filters to time period of interest, and returns a hashable dataset.

    Parameters
    ----------
    data_file : Path
        Path to file containing wave data (netCDF by default; other formats
        supported via engine kwarg in **kwargs, e.g., zarr, grib, hdf5)
    lon_name : str, default="longitude"
        Name of longitude variable in source file
    lat_name : str, default="latitude"
        Name of latitude variable in source file
    time_name : str, default="time"
        Name of time variable in source file
    wh_name : str, default="VHM0"
        Name of significant wave height variable in source file
    time_start : np.datetime64, optional
        Start time for filtering data
    time_end : np.datetime64, optional
        End time for filtering data
    load_eagerly : bool, default=True
        If True, load data into memory immediately
    spatial_bounds : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) for spatial cropping
    **kwargs
        Additional arguments passed to xr.open_dataset

    Returns
    -------
    HashableDataset
        Dataset with standardized variable names (lon, lat, time, wh)
    """
    ds = xr.open_dataset(data_file, **kwargs)
    ds = ds.rename(
        {
            lon_name: "lon",
            lat_name: "lat",
            time_name: "time",
            wh_name: "wh",
        }
    )
    ds = _filter_times(ds, time_start=time_start, time_end=time_end)
    if spatial_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = spatial_bounds
        ds = _apply_spatial_selection(ds, lon_min, lon_max, lat_min, lat_max)
    if load_eagerly:
        ds = ds.load()
    return make_hashable(ds)


@profile
def _filter_times(
    ds: xr.Dataset = None,
    time_start: np.datetime64 = None,
    time_end: np.datetime64 = None,
) -> xr.Dataset:
    """Filter dataset to time period of interest.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    time_start : np.datetime64
        Start time for filtering.
    time_end : np.datetime64
        End time for filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """
    # If we don't filter at all, just fall through
    if time_end is None and time_start is None:
        return ds

    # Calculate maximum time step for buffer
    time_buffer = ds.time.diff("time").max().load().data[()]
    if time_start is not None:
        time_sel_start = time_start - time_buffer
    if time_end is not None:
        time_sel_end = time_end + time_buffer
    # Filter time axis
    ds = ds.sel(time=slice(time_sel_start, time_sel_end))

    return ds


@profile
@lru_cache(maxsize=MAX_CACHE_SIZE)
def _select_ij(
    ds: xr.Dataset = None,
    lon_start=None,
    lon_end=None,
    lat_start=None,
    lat_end=None,
):
    """Select dataset along spatial dimensions between start and end points."""
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


@profile
@lru_cache(maxsize=MAX_CACHE_SIZE)
def _select_l(
    ds: xr.Dataset = None,
    time_start=None,
    time_end=None,
):
    """Select dataset along time dimension between start and end times."""
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


def _apply_spatial_selection(
    ds: xr.Dataset = None,
    lon_min: float = None,
    lon_max: float = None,
    lat_min: float = None,
    lat_max: float = None,
) -> xr.Dataset:
    """Apply spatial selection to dataset using bounding box.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with lon, lat dimensions
    lon_min : float
        Minimum longitude for selection
    lon_max : float
        Maximum longitude for selection
    lat_min : float
        Minimum latitude for selection
    lat_max : float
        Maximum latitude for selection

    Returns
    -------
    xr.Dataset
        Dataset cropped to specified bounding box
    """
    return ds.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )


@profile
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
    """Select environmental data for a route leg.

    Extracts subset of dataset along spatial and temporal dimensions
    covering the specified leg.

    Parameters
    ----------
    ds : xr.Dataset
        Environmental dataset
    lon_start : float
        Starting longitude
    lon_end : float
        Ending longitude
    lat_start : float
        Starting latitude
    lat_end : float
        Ending latitude
    time_start : datetime-like
        Starting time
    time_end : datetime-like
        Ending time

    Returns
    -------
    xr.Dataset
        Subset of data along leg path
    """
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
