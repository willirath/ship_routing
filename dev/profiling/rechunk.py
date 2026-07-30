"""Write a rechunked (small-chunk, sharded) copy of the profiling case's forcing crop.

Reads lazily, crops to the production ellipse bbox + journey window, writes zarr v3
with small inner chunks inside larger shards.
"""
import sys, time, shutil
from pathlib import Path
import numpy as np
import xarray as xr
from shapely.geometry import LineString

from ship_routing.core.geodesics import compute_ellipse_bbox, knots_to_ms, get_length_meters

LON = (-80.5, -11.0); LAT = (30.0, 50.0)
T0 = np.datetime64("2021-01-01T00:00:00")
duration_s = get_length_meters(LineString(zip(LON, LAT))) / knots_to_ms(12.0)
T1 = T0 + np.timedelta64(int(duration_s), "s")
lon_min, lon_max, lat_min, lat_max = compute_ellipse_bbox(
    lon_start=LON[0], lat_start=LAT[0], lon_end=LON[1], lat_end=LAT[1],
    length_multiplier=1.5, buffer_degrees=0.5)

ROOT = Path("/Users/wrath/src/git.geomar.de/willi-rath/ship_routing_integrated/ship_routing")
SRC = ROOT / "data/large"
DST = ROOT / "data/rechunked"
DST.mkdir(parents=True, exist_ok=True)

STORES = {
    "currents": "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr",
    "winds": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr",
    "waves": "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr",
}

CHUNK = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (4, 16, 16)
# How many chunks go into one shard, per axis. Sharding keeps the file count sane
# when chunks are small, at the cost of an extra indirection on every read; pass
# "none" to write plain chunks instead.
SHARD_ARG = sys.argv[2] if len(sys.argv) > 2 else "4,16,16"
SHARD_MULT = None if SHARD_ARG == "none" else tuple(int(x) for x in SHARD_ARG.split(","))

for name, fname in STORES.items():
    suffix = "x".join(map(str, CHUNK)) + ("" if SHARD_MULT else "_noshard")
    out = DST / f"{name}_c{suffix}.zarr"
    if out.exists():
        shutil.rmtree(out)
    ds = xr.open_zarr(SRC / fname).rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    tb = ds.time.diff("time").max().values
    ds = ds.sel(time=slice(T0 - tb, T1 + tb)).transpose("time", "lat", "lon")

    nt, ny, nx = ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]
    chunk = tuple(min(c, s) for c, s in zip(CHUNK, (nt, ny, nx)))
    if SHARD_MULT is None:
        shard = None
    else:
        shard = tuple(min(c * m, s) for c, m, s in zip(chunk, SHARD_MULT, (nt, ny, nx)))
        # shards must be whole multiples of chunks
        shard = tuple(int(np.ceil(s / c)) * c for s, c in zip(shard, chunk))

    # Source is zarr v2 (numcodecs Blosc); clear inherited encoding so v3 codecs apply.
    for v in list(ds.variables):
        ds[v].encoding = {}
    spec = {"chunks": chunk, "dtype": "float32"}
    if shard is not None:
        spec["shards"] = shard
    enc = {v: spec for v in ds.data_vars}
    write_chunk = shard if shard is not None else tuple(
        c * m for c, m in zip(chunk, (4, 16, 16))
    )
    ds = ds.chunk(
        {"time": write_chunk[0], "lat": write_chunk[1], "lon": write_chunk[2]}
    )
    # Restore the source coordinate names so the result is a drop-in replacement
    # for ship_routing.core.data.load_* .
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    enc = {v: dict(e) for v, e in enc.items()}
    t = time.perf_counter()
    ds.to_zarr(out, mode="w", zarr_format=3, encoding=enc, consolidated=True)
    dt = time.perf_counter() - t
    nbytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    raw = nt * ny * nx * 4 * len(ds.data_vars)
    print(f"{name:9s} {nt}x{ny}x{nx} chunk={chunk} shard={shard} "
          f"-> {nbytes/2**20:7.1f} MiB on disk (raw {raw/2**20:7.1f} MiB, "
          f"ratio {raw/nbytes:.2f}x) in {dt:.1f}s")
