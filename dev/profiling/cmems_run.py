#!/usr/bin/env python
"""Route against Copernicus Marine (CMEMS) forcing.

Two things are worth separating here, because they behave very differently.

**Sampling straight off the ARCO stores does not work well.** The block cache in
:class:`ship_routing.core.lookup.BlockCachedForcingGrid` reads correctly from
them -- ``--direct`` demonstrates that -- but CMEMS chunks are laid out for
time-series or map extraction, not for the diagonal a route traces through
``(time, lat, lon)``. Measured chunk shapes are ``(50, 2, 512, 2048)`` for
currents, ``(50, 1024, 1024)`` for winds and ``(17720, 16, 200)`` for waves, so
one small block pulls a few hundred MB over the network and discards almost all
of it: about 11 s per cold block, which is hours for a real run.

**Subsetting first does.** ``copernicusmarine`` subsets server-side, and the
ellipse bbox plus journey window is small. So the workable path is to mirror the
subset into a local store with route-friendly chunks (see ``rechunk.py``) and let
the block cache work against that -- which is what this script does by default.

Usage
-----
    # mirror the subset locally (once), then route against it
    pixi run python dev/profiling/cmems_run.py

    # show what sampling the ARCO store directly costs
    pixi run python dev/profiling/cmems_run.py --direct
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import click
import numpy as np
import xarray as xr
from shapely.geometry import LineString

from ship_routing.core.geodesics import compute_ellipse_bbox, get_length_meters, knots_to_ms
from ship_routing.core.lookup import BlockCachedForcingGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data" / "cmems"

#: dataset id, variables, and any extra subsetting needed to reach a 3-D cube.
DATASETS = {
    "currents": dict(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        extra=dict(minimum_depth=0.0, maximum_depth=1.0),
    ),
    "winds": dict(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        variables=["eastward_wind", "northward_wind"],
        extra={},
    ),
    "waves": dict(
        dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
        variables=["VHM0", "VMDR"],
        extra={},
    ),
}

LON = (-80.5, -11.0)
LAT = (30.0, 50.0)


def journey_window(days: float):
    """Ellipse bbox and time window for the representative Atlantic journey."""
    time_start = np.datetime64("2021-01-01T00:00:00")
    duration_s = get_length_meters(LineString(zip(LON, LAT))) / knots_to_ms(12.0)
    time_end = time_start + np.timedelta64(int(min(duration_s, days * 86400)), "s")
    bbox = compute_ellipse_bbox(
        lon_start=LON[0],
        lat_start=LAT[0],
        lon_end=LON[1],
        lat_end=LAT[1],
        length_multiplier=1.5,
        buffer_degrees=0.5,
    )
    return bbox, time_start, time_end


def open_cmems(name: str, bbox, time_start, time_end) -> xr.Dataset:
    """Lazily open one CMEMS dataset, subset server-side to the journey."""
    import copernicusmarine as cm

    spec = DATASETS[name]
    lon_min, lon_max, lat_min, lat_max = bbox
    ds = cm.open_dataset(
        dataset_id=spec["dataset_id"],
        variables=spec["variables"],
        minimum_longitude=float(lon_min),
        maximum_longitude=float(lon_max),
        minimum_latitude=float(lat_min),
        maximum_latitude=float(lat_max),
        start_datetime=str(time_start),
        end_datetime=str(time_end),
        **spec["extra"],
    )
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    return ds.transpose("time", "latitude", "longitude")


def mirror(name: str, bbox, time_start, time_end, chunk, force: bool) -> Path:
    """Materialise the CMEMS subset into a local, route-friendly zarr store."""
    out = CACHE / f"{name}_c{'x'.join(map(str, chunk))}.zarr"
    if out.exists() and not force:
        print(f"  {name:9s} cached at {out.relative_to(REPO_ROOT)}")
        return out
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    ds = open_cmems(name, bbox, time_start, time_end)
    sizes = tuple(ds.sizes[d] for d in ("time", "latitude", "longitude"))
    ds = ds.astype("float32").load()
    fetch_s = time.perf_counter() - t

    for var in list(ds.variables):
        ds[var].encoding = {}
    block = tuple(min(c, s) for c, s in zip(chunk, sizes))
    ds = ds.chunk(
        {"time": block[0], "latitude": block[1], "longitude": block[2]}
    )
    ds.to_zarr(
        out,
        mode="w",
        zarr_format=3,
        consolidated=True,
        encoding={v: {"chunks": block, "dtype": "float32"} for v in ds.data_vars},
    )
    on_disk = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(
        f"  {name:9s} {sizes} fetched in {fetch_s:6.1f}s, "
        f"chunk={block} -> {on_disk / 2**20:6.1f} MiB"
    )
    return out


@click.command()
@click.option("--days", type=float, default=12.0, help="Journey length to cover.")
@click.option("--chunk", type=str, default="8,32,32", help="Chunk/block shape.")
@click.option("--force", is_flag=True, help="Re-download even if cached.")
@click.option(
    "--direct",
    is_flag=True,
    help="Instead of mirroring, sample the ARCO store directly and time it.",
)
def main(days, chunk, force, direct):
    chunk = tuple(int(v) for v in chunk.split(","))
    bbox, time_start, time_end = journey_window(days)
    print(f"journey window {time_start} .. {time_end}")
    print(f"ellipse bbox lon {bbox[0]:.2f}..{bbox[1]:.2f} lat {bbox[2]:.2f}..{bbox[3]:.2f}")

    if direct:
        print("\n=== sampling the CMEMS ARCO store directly ===")
        for name in DATASETS:
            ds = open_cmems(name, bbox, time_start, time_end)
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
            grid = BlockCachedForcingGrid(ds, block_shape=chunk, max_workers=4)
            lon, lat, tim = grid.lon, grid.lat, grid.time
            t = time.perf_counter()
            grid.sample_leg(
                lon_start=float(lon[len(lon) // 3]),
                lat_start=float(lat[len(lat) // 3]),
                time_start=tim[len(tim) // 3],
                lon_end=float(lon[len(lon) // 3 + 20]),
                lat_end=float(lat[len(lat) // 3 + 5]),
                time_end=tim[min(len(tim) // 3 + 2, len(tim) - 1)],
            )
            cold = time.perf_counter() - t
            t = time.perf_counter()
            grid.sample_leg(
                lon_start=float(lon[len(lon) // 3]),
                lat_start=float(lat[len(lat) // 3]),
                time_start=tim[len(tim) // 3],
                lon_end=float(lon[len(lon) // 3 + 5]),
                lat_end=float(lat[len(lat) // 3 + 2]),
                time_end=tim[min(len(tim) // 3 + 1, len(tim) - 1)],
            )
            warm = time.perf_counter() - t
            print(
                f"  {name:9s} cold leg {cold:7.2f}s ({grid.misses} blocks), "
                f"warm leg {1e6 * warm:8.0f} us"
            )
        print(
            "\nCold reads are dominated by CMEMS chunk shapes, not by our block size:\n"
            "one small block pulls a whole ARCO chunk. Mirror the subset instead."
        )
        return

    print(f"\n=== mirroring CMEMS subset to {CACHE.relative_to(REPO_ROOT)} ===")
    paths = {
        name: mirror(name, bbox, time_start, time_end, chunk, force)
        for name in DATASETS
    }

    print("\n=== routing against the mirrored subset ===")
    from ship_routing.app.config import ForcingConfig, HyperParams, JourneyConfig, RoutingConfig
    from ship_routing.app.routing import RoutingApp

    config = RoutingConfig(
        journey=JourneyConfig(
            name="Atlantic_forward_cmems",
            lon_waypoints=LON,
            lat_waypoints=LAT,
            time_start=str(time_start),
            speed_knots=12.0,
            time_resolution_hours=6.0,
        ),
        forcing=ForcingConfig(
            currents_path=paths["currents"],
            winds_path=paths["winds"],
            waves_path=paths["waves"],
            engine="zarr",
            load_eagerly=False,
            chunks=None,
        ),
        hyper=HyperParams(
            population_size=32,
            offspring_size=32,
            generations=2,
            gd_iterations=1,
            random_seed=345,
            executor_type="sequential",
            num_workers=1,
        ),
    )
    t = time.perf_counter()
    result = RoutingApp(config).run()
    print(f"\ntotal wall clock: {time.perf_counter() - t:.2f} s")
    print(f"seed cost:  {result.seed_member.cost:.6e}")
    print(f"elite cost: {min(m.cost for m in result.elite_population.members):.6e}")

    from ship_routing.core.lookup import LIVE_BLOCK_CACHED_GRIDS

    total = 0
    for grid in LIVE_BLOCK_CACHED_GRIDS:
        total += grid.cache_bytes
        looked_up = grid.hits + grid.misses
        print(
            f"  {'/'.join(grid.names):12s} {grid.n_blocks_resident:5d} blocks "
            f"{grid.cache_bytes / 2**20:6.1f} MiB  "
            f"hit rate {100 * grid.hits / max(looked_up, 1):5.1f}%"
        )
    print(f"  block cache total {total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
