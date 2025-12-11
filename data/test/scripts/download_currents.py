#!/usr/bin/env python

import copernicusmarine


def download_30days_1deg_5daily_global_current_data():
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    )

    ds = (
        ds.isel(depth=0)
        .sel(time=slice("2024-01-01", "2024-02-01"))
        .isel(time=slice(None, 30))
    )

    ds = ds.coarsen(
        longitude=12,
        latitude=12,
        time=5,
        boundary="pad",
    ).mean()

    ds.drop_encoding().to_netcdf(
        "../currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2024-01_1deg_5day.nc"
    )


def download_1day_112deg_regional_current_data():

    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
    )

    ds = (
        ds.sel(time=slice("2024-01-01", None))
        .isel(time=slice(0, 1))
        .isel(depth=0)
        .sel(longitude=slice(-100, 20))
        .sel(latitude=slice(10, 65))
    )

    ds = ds.drop_encoding()

    ds.uo.encoding["complevel"] = 1
    ds.uo.encoding["contiguous"] = False
    ds.uo.encoding["zlib"] = True
    ds.uo.encoding["shuffle"] = True

    ds.vo.encoding["complevel"] = 1
    ds.vo.encoding["contiguous"] = False
    ds.vo.encoding["zlib"] = True
    ds.vo.encoding["shuffle"] = True

    ds.to_netcdf(
        "../currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2024-01_100W-020E_10N-65N.nc"
    )


if __name__ == "__main__":
    download_30days_1deg_5daily_global_current_data()
    download_1day_112deg_regional_current_data()
