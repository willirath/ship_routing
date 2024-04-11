Test data was created with:
```python
import copernicusmarine

ds = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
)

ds = ds.isel(depth=0).sel(time=slice("2021-01-01", "2021-02-01")).isel(time=slice(None, 30))

ds = ds.coarsen(
    longitude=12, latitude=12,
    time=5,
    boundary="pad",
).mean()

ds.drop_encoding().to_netcdf("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc")
```