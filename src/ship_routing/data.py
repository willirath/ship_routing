import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path

from functools import lru_cache
from .config import MAX_CACHE_SIZE


class HashableDataset(xr.Dataset):
    def __hash__(self):
        # Note that there's a _lot_ of assumptions going into this...
        return hash(id(self))


def make_hashable(ds):
    return HashableDataset(ds)


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
