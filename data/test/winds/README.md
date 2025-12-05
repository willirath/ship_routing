Test data was created with:
```python
import copernicusmarine
import numpy as np

ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
)

ds = ds[["eastward_wind", "northward_wind"]]
ds = ds.sel(
    longitude=slice(-100, -20), latitude=slice(10, 65)
)
ds = ds.isel(
    longitude=slice(None, None, 4),
    latitude=slice(None, None, 4),
)
ds = ds.sel(time=slice(
    "2021-01-01T00:00:00", 
    "2021-02-01T00:00:00", 
))
ds = ds.isel(time=slice(None, None, 6))
ds

ds = ds.drop_encoding()
for vname in ds.data_vars.keys():
    ds[vname].encoding["zlib"] = True
    ds[vname].encoding["complevel"] = 2
ds

ds.to_netcdf("cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2021-01_6hours_0.5deg_100W-020E_10N-65N.nc")
```