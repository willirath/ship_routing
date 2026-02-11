#!/usr/bin/env python
"""Compute decomposed costs for a single baseline msgpack file.

Decomposes route costs into 3 additive components that sum exactly:
- cost_calm: Hull resistance (depends on speed through water)
- cost_waves: Wave added resistance (depends on speed through water and wave height)
- cost_wind: Wind resistance (depends on speed through wind)

Also computes "no current" ablation for calm and wave components to isolate
current effects.

Output: ../results/decomposed_costs_<stem>.parquet
    Plain parquet (no geometry) joinable with results_prelim.geoparquet
    on (filename, n_elite).
"""

from pathlib import Path

import click
import geopandas as gpd
import msgpack
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment_params import FORCING_SCENARIOS
from ship_routing.app.routing import RoutingResult
from ship_routing.core.config import SHIP_DEFAULT, PHYSICS_DEFAULT
from ship_routing.core.data import load_currents, load_waves, load_winds
from ship_routing.core.routes import Route, WayPoint
from load_tuning_results import filter_suspicious_routes, add_derived_features

import warnings

warnings.filterwarnings("ignore")


def route_from_routing_result(rr, elite_idx=0):
    """Extract a Route with ns-precision timestamps from a RoutingResult."""
    route = rr.elite_population.members[elite_idx].route
    fixed_waypoints = [
        WayPoint(
            lon=wp.lon,
            lat=wp.lat,
            time=np.datetime64(wp.time, "ns"),
        )
        for wp in route.way_points
    ]
    return Route(way_points=tuple(fixed_waypoints))


def decompose_route(route, currents, waves, winds):
    """Compute decomposed costs and hazard info for a route."""
    costs = route.cost_through_decomposed(
        current_data_set=currents,
        wind_data_set=winds,
        wave_data_set=waves,
        ship=SHIP_DEFAULT,
        physics=PHYSICS_DEFAULT,
    )

    wave_heights = []
    for wp in route.way_points:
        try:
            wh = float(
                waves.wh.sel(
                    lon=wp.lon, lat=wp.lat, time=wp.time, method="nearest"
                ).values
            )
            if not np.isnan(wh):
                wave_heights.append(wh)
        except Exception:
            pass

    hazard_threshold_m = SHIP_DEFAULT.waterline_length_m / 40.0
    costs["max_wave_height_m"] = max(wave_heights) if wave_heights else np.nan
    costs["is_hazardous"] = costs["max_wave_height_m"] > hazard_threshold_m
    return costs


@click.command()
@click.argument("msgpack_path", type=click.Path(exists=True, path_type=Path))
def main(msgpack_path):
    """Compute decomposed costs for routes in a single MSGPACK_PATH file."""
    stem = msgpack_path.stem
    out_path = Path("../results") / f"decomposed_costs_{stem}.parquet"
    click.echo(f"Worker: {msgpack_path.name} -> {out_path.name}")

    # -----------------------------------------------------------------
    # Identify baseline routes from prelim metadata
    # -----------------------------------------------------------------
    click.echo("Loading results metadata...")
    gdf = gpd.read_parquet("../results/results_prelim.geoparquet")
    gdf = add_derived_features(gdf)
    gdf = filter_suspicious_routes(gdf)
    gdf = gdf[gdf.forcing_scenario_name == "baseline"].copy()
    gdf = gdf.reset_index()
    baseline_keys = set(zip(gdf.filename, gdf.n_elite.astype(int)))
    click.echo(f"  Baseline routes total: {len(baseline_keys)}")

    # -----------------------------------------------------------------
    # Load forcing data
    # -----------------------------------------------------------------
    bounds = gdf.total_bounds
    spatial_bounds = (bounds[0] - 5, bounds[2] + 5, bounds[1] - 5, bounds[3] + 5)

    baseline = FORCING_SCENARIOS["baseline"]
    time_start = np.datetime64("2021-01-01")
    time_end = np.datetime64("2021-12-31T23:59:59")
    data_prefix = Path("..")

    click.echo("Loading currents...")
    currents = load_currents(
        data_prefix / baseline["currents_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )

    click.echo("Loading waves...")
    waves = load_waves(
        data_prefix / baseline["waves_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )

    click.echo("Loading winds...")
    winds = load_winds(
        data_prefix / baseline["winds_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )

    # -----------------------------------------------------------------
    # Process single msgpack file
    # -----------------------------------------------------------------
    click.echo(f"Loading {msgpack_path.name}...")
    with open(msgpack_path, "rb") as f:
        raw = msgpack.unpack(f, raw=False)

    results = []
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for key, value in tqdm(raw.items(), desc=stem):
        rr = RoutingResult.from_msgpack(value)
        n_elites = len(rr.elite_population.members)

        for elite_idx in range(n_elites):
            if (key, elite_idx) not in baseline_keys:
                n_skipped += 1
                continue

            try:
                route = route_from_routing_result(rr, elite_idx)
                costs = decompose_route(route, currents, waves, winds)
                costs["filename"] = key
                costs["n_elite"] = elite_idx
                results.append(costs)
                n_processed += 1
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    click.echo(f"  Failed: {key} elite={elite_idx}: {e}")

    del raw

    click.echo(f"  Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(out_path, index=False)
        click.echo(f"  Saved {len(df)} rows to {out_path}")
    else:
        click.echo("  No results to save.")


if __name__ == "__main__":
    main()
