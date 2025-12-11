#!/usr/bin/env python

import copernicusmarine
import numpy as np


def download_30days_6hourly_regional_wave_data():
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
    )

    ds = ds[["eastward_wind", "northward_wind"]]
    ds = ds.sel(longitude=slice(-100, 20), latitude=slice(10, 65))
    ds = ds.isel(
        longitude=slice(None, None, 4),
        latitude=slice(None, None, 4),
    )
    ds = ds.sel(
        time=slice(
            "2024-01-01T00:00:00",
            "2024-02-01T00:00:00",
        )
    )
    ds = ds.isel(time=slice(None, None, 6))
    ds

    ds = ds.drop_encoding()
    for vname in ds.data_vars.keys():
        ds[vname].encoding["zlib"] = True
        ds[vname].encoding["complevel"] = 2
    ds

    ds.to_netcdf(
        "../winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2024-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )


if __name__ == "__main__":
    download_30days_6hourly_regional_wave_data()
