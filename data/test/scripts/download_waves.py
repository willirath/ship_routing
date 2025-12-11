#!/usr/bin/env python

import copernicusmarine
import numpy as np


def downlaoad_30days_regional_wave_data():
    ds = copernicusmarine.open_dataset(dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i")

    wave_height = ds.VHM0
    wave_height = wave_height.sel(
        time=slice(
            "2024-01-01T00:00:00",
            "2024-02-01T00:00:00",
        )
    )
    wave_height = wave_height.sel(longitude=slice(-100, 20), latitude=slice(10, 65))
    wave_height = wave_height.resample(time="1D").max()
    wave_height = wave_height.assign_coords(
        time=wave_height.time + np.timedelta64(12, "h")
    )
    wave_height = wave_height.drop_encoding()
    wave_height.encoding["zlib"] = True
    wave_height.encoding["complevel"] = 2
    wave_height.to_dataset(name="VHM0").to_netcdf(
        "../waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2024-01_1d-max_100W-020E_10N-65N.nc"
    )


if __name__ == "__main__":
    downlaoad_30days_regional_wave_data()
