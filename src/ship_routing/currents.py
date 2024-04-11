import xarray as xr

from pathlib import Path


def load_currents(
    data_file: Path = None,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    time_name: str = "time",
    u_name: str = "uo",
    v_name: str = "vo",
) -> xr.Dataset:
    ds = xr.open_dataset(data_file)
    ds = ds.rename(
        {lon_name: "lon", lat_name: "lat", time_name: "time", u_name: "u", v_name: "v"}
    )
    return ds
