import xarray as xr
import pandas as pd

from pathlib import Path


def load_currents(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uo_name: str = "uo",
    vo_name: str = "vo",
) -> xr.Dataset:
    ds = xr.open_dataset(data_file)
    ds = ds.rename(
        {
            lon_name: "lon",
            lat_name: "lat",
            time_name: "time",
            uo_name: "uo",
            vo_name: "vo",
        }
    )
    return ds


def load_currents_time_average(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    uo_name: str = "uo",
    vo_name: str = "vo",
) -> xr.Dataset:
    ds = load_currents(
        data_file=data_file,
        lon_name=lon_name,
        lat_name=lat_name,
        time_name=time_name,
        uo_name=uo_name,
        vo_name=vo_name,
    )
    ds = ds.mean("time")
    return ds


def select_currents_along_traj(
    ds: xr.Dataset = None, ship_positions: pd.DataFrame = None
):
    ship_pos_ds = ship_positions.to_xarray()
    return ds.sel(
        lon=ship_pos_ds.lon,
        lat=ship_pos_ds.lat,
        method="nearest",
    )
