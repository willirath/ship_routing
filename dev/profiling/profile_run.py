#!/usr/bin/env python
"""Profile a scaled-down production routing run.

Runs one representative case (Atlantic forward, January, 12 kn) with the
production algorithm structure but reduced population / generations so the
whole run finishes in ~1-2 minutes on a laptop.

Usage
-----
    # plain timing + per-stage breakdown
    pixi run python dev/profiling/profile_run.py

    # with cProfile, dumping stats to dev/profiling/out/
    pixi run python dev/profiling/profile_run.py --cprofile

    # scale knobs
    pixi run python dev/profiling/profile_run.py --population-size 64 --generations 2
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from ship_routing.app.config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)
from ship_routing.app.routing import RoutingApp

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "large"

LARGE_STORES = {
    "currents": "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr",
    "waves": "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr",
    "winds": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr",
}


def build_forcing(
    data_dir: Path, load_eagerly: bool, chunks, pattern: str = "{name}_*.zarr"
) -> ForcingConfig:
    """Point at either the production store or a rechunked copy of it.

    The rechunked copies (``dev/profiling/rechunk.py``) carry small chunks, which
    is what makes lazy block-cached sampling pay off -- see
    :class:`ship_routing.core.lookup.BlockCachedForcingGrid`.
    """
    if data_dir.name == "large":
        paths = {k: data_dir / v for k, v in LARGE_STORES.items()}
    else:
        paths = {}
        for name in LARGE_STORES:
            matches = sorted(data_dir.glob(pattern.format(name=name)))
            if not matches:
                raise SystemExit(
                    f"no store matching {pattern.format(name=name)!r} in {data_dir}"
                )
            paths[name] = matches[0]
    return ForcingConfig(
        currents_path=paths["currents"],
        waves_path=paths["waves"],
        winds_path=paths["winds"],
        engine="zarr",
        load_eagerly=load_eagerly,
        chunks=chunks,
    )

# Representative case: Atlantic forward, January, 12 kn (production journey def)
JOURNEY = JourneyConfig(
    name="Atlantic_forward",
    lon_waypoints=(-80.5, -11.0),
    lat_waypoints=(30.0, 50.0),
    time_start="2021-01-01T00:00:00",
    speed_knots=12.0,
    time_resolution_hours=6.0,
)


def build_config(
    population_size: int,
    offspring_size: int,
    generations: int,
    gd_iterations: int,
    seed: int,
    forcing: ForcingConfig,
) -> RoutingConfig:
    """Production hyperparameters, scaled down in population and depth."""
    return RoutingConfig(
        journey=JOURNEY,
        forcing=forcing,
        hyper=HyperParams(
            population_size=population_size,
            offspring_size=offspring_size,
            generations=generations,
            gd_iterations=gd_iterations,
            random_seed=seed,
            executor_type="sequential",
            num_workers=1,
        ),
    )


def peak_rss_mib() -> float:
    """Peak resident set size of this process, in MiB."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS reports bytes.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def report_block_caches() -> None:
    """Print block cache statistics, if the run sampled lazily."""
    from ship_routing.core.lookup import LIVE_BLOCK_CACHED_GRIDS

    grids = list(LIVE_BLOCK_CACHED_GRIDS)
    if not grids:
        return
    print("\n=== block caches ===")
    total = 0
    for grid in grids:
        total += grid.cache_bytes
        looked_up = grid.hits + grid.misses
        print(
            f"  {'/'.join(grid.names):12s} block={grid.block_shape} "
            f"{grid.n_blocks_resident:6d} blocks resident "
            f"{grid.cache_bytes / 2**20:7.1f} MiB  "
            f"hit rate {100 * grid.hits / max(looked_up, 1):5.1f}% "
            f"({grid.misses} fetched)"
        )
    print(f"  {'TOTAL':12s} {total / 2**20:.1f} MiB resident")


def stage_breakdown(result) -> pd.DataFrame:
    """Wall-clock seconds between consecutive stage log entries."""
    df = result.logs.to_dataframe()
    if df.empty or "timestamp" not in df:
        return pd.DataFrame()
    t = pd.to_datetime(df["timestamp"])
    df = df.assign(dt_seconds=t.diff().dt.total_seconds().shift(-1))
    grouped = (
        df.groupby("stage")["dt_seconds"]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
    )
    grouped["percent"] = 100.0 * grouped["sum"] / grouped["sum"].sum()
    return grouped


@click.command()
@click.option("--population-size", type=int, default=64)
@click.option("--offspring-size", type=int, default=64)
@click.option("--generations", type=int, default=2)
@click.option("--gd-iterations", type=int, default=1)
@click.option("--seed", type=int, default=345)
@click.option("--cprofile", is_flag=True, help="Run under cProfile.")
@click.option(
    "--lazy",
    is_flag=True,
    help="Open forcing lazily and sample it through the block cache.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DATA,
    help="Forcing store directory (data/large, or a rechunked copy).",
)
@click.option(
    "--store-pattern",
    type=str,
    default="{name}_*.zarr",
    help="Glob picking one store per variable group inside --data-dir.",
)
@click.option(
    "--chunks",
    type=str,
    default=None,
    help="xarray chunks. 'auto' uses dask; unset opens without it (faster per block).",
)
@click.option(
    "--block-shape",
    type=str,
    default=None,
    help="Block cache tile as 'time,lat,lon'. Only meaningful with --lazy.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path(__file__).resolve().parent / "out",
)
@click.option("--tag", type=str, default="baseline", help="Label for output files.")
def main(
    population_size,
    offspring_size,
    generations,
    gd_iterations,
    seed,
    cprofile,
    lazy,
    data_dir,
    store_pattern,
    chunks,
    block_shape,
    out,
    tag,
):
    if block_shape is not None:
        from ship_routing.core import config as core_config

        core_config.BLOCK_SHAPE_DEFAULT = tuple(
            int(v) for v in block_shape.split(",")
        )
        # lookup.py binds the default at import time.
        from ship_routing.core import lookup

        lookup.BLOCK_SHAPE_DEFAULT = core_config.BLOCK_SHAPE_DEFAULT

    out.mkdir(parents=True, exist_ok=True)
    config = build_config(
        population_size=population_size,
        offspring_size=offspring_size,
        generations=generations,
        gd_iterations=gd_iterations,
        seed=seed,
        forcing=build_forcing(
            data_dir=data_dir,
            load_eagerly=not lazy,
            chunks=chunks,
            pattern=store_pattern,
        ),
    )
    app = RoutingApp(config)

    t0 = time.perf_counter()
    if cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        result = app.run()
        profiler.disable()
        stats_path = out / f"{tag}.pstats"
        profiler.dump_stats(stats_path)
        print(f"\nprofile written to {stats_path}")
    else:
        result = app.run()
    elapsed = time.perf_counter() - t0

    print(f"\n=== total wall clock: {elapsed:.2f} s ===")
    print(f"peak RSS: {peak_rss_mib():.0f} MiB")
    print(f"seed cost:  {result.seed_member.cost:.6e}")
    print(f"elite cost: {min(m.cost for m in result.elite_population.members):.6e}")

    report_block_caches()

    breakdown = stage_breakdown(result)
    if not breakdown.empty:
        print("\n=== per-stage wall clock ===")
        print(breakdown.to_string(float_format=lambda v: f"{v:10.3f}"))
        breakdown.to_csv(out / f"{tag}_stages.csv")

    if cprofile:
        stats = pstats.Stats(str(out / f"{tag}.pstats"))
        print("\n=== top 30 by cumulative time ===")
        stats.sort_stats("cumulative").print_stats(30)
        print("\n=== top 30 by total (self) time ===")
        stats.sort_stats("tottime").print_stats(30)

    # Record the elite cost so optimisation work can check for regressions.
    np.save(out / f"{tag}_costs.npy", np.array([m.cost for m in result.elite_population.members]))


if __name__ == "__main__":
    main()
