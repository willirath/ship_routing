#!/usr/bin/env python
"""Compute exact decomposed costs for baseline elite routes.

Decomposes route costs into 3 additive components that sum exactly:
- cost_calm: Hull resistance (depends on speed through water)
- cost_waves: Wave added resistance (depends on speed through water and wave height)
- cost_wind: Wind resistance (depends on speed through wind)

Also computes "no current" ablation for calm and wave components to isolate
current effects.

Output: ../results/baseline_elites_decomposed.geoparquet
"""

from pathlib import Path

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


# =============================================================================
# Helper functions
# =============================================================================


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

    # Sample max wave height along route
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


# =============================================================================
# Main
# =============================================================================


def main():
    # -------------------------------------------------------------------------
    # Identify baseline routes
    # -------------------------------------------------------------------------
    print("Loading results metadata...")
    gdf = gpd.read_parquet("../results/results_prelim.geoparquet")
    gdf = add_derived_features(gdf)
    gdf = filter_suspicious_routes(gdf)

    gdf = gdf[gdf.forcing_scenario_name == "baseline"].copy()
    gdf = gdf.reset_index()

    baseline_keys = set(zip(gdf.filename, gdf.n_elite.astype(int)))
    print(f"Baseline routes to decompose: {len(baseline_keys)}")
    print(f"Unique experiments: {gdf.filename.nunique()}")

    # -------------------------------------------------------------------------
    # Load forcing data
    # -------------------------------------------------------------------------
    msgpack_files = sorted(Path("../results/").glob("results_ablation_baseline_2*.msgpack"))
    msgpack_files = [f for f in msgpack_files if "crosseval" not in f.name]
    print(f"Found {len(msgpack_files)} baseline msgpack files")

    bounds = gdf.total_bounds
    spatial_bounds = (bounds[0] - 5, bounds[2] + 5, bounds[1] - 5, bounds[3] + 5)
    print(f"Spatial bounds: {spatial_bounds}")

    baseline = FORCING_SCENARIOS["baseline"]
    time_start = np.datetime64("2021-01-01")
    time_end = np.datetime64("2021-12-31T23:59:59")
    data_prefix = Path("..")

    print("Loading currents...")
    currents = load_currents(
        data_prefix / baseline["currents_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )
    print(f"  Currents: {dict(currents.dims)}")

    print("Loading waves...")
    waves = load_waves(
        data_prefix / baseline["waves_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )
    print(f"  Waves: {dict(waves.dims)}")

    print("Loading winds...")
    winds = load_winds(
        data_prefix / baseline["winds_path"],
        time_start=time_start,
        time_end=time_end,
        engine=baseline["engine"],
        spatial_bounds=spatial_bounds,
        load_eagerly=True,
    )
    print(f"  Winds: {dict(winds.dims)}")

    # -------------------------------------------------------------------------
    # Compute decomposed costs
    # -------------------------------------------------------------------------
    print("\nComputing decomposed costs...")
    results = []
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for mf in tqdm(msgpack_files, desc="Files"):
        with open(mf, "rb") as f:
            raw = msgpack.unpack(f, raw=False)

        for key, value in tqdm(raw.items(), desc=mf.name, leave=False):
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
                        print(f"Failed: {key} elite={elite_idx}: {e}")

        del raw

    print(f"\nProcessed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")

    # -------------------------------------------------------------------------
    # Merge and verify
    # -------------------------------------------------------------------------
    df_costs = pd.DataFrame(results)
    print(f"Decomposed costs: {len(df_costs)} routes")

    gdf_out = gdf.merge(df_costs, on=["filename", "n_elite"], how="inner")
    print(f"Merged output: {len(gdf_out)} rows")

    # Verify decomposition
    sum_check = gdf_out.cost_calm + gdf_out.cost_waves + gdf_out.cost_wind
    max_error = (sum_check - gdf_out.cost_total).abs().max()
    print(f"Decomposition verification: max error = {max_error:.2e}")

    # Summary
    total = gdf_out.cost_total.mean()
    print(f"\n=== Component Fractions ===")
    print(f"  calm/total:  {gdf_out.cost_calm.mean() / total * 100:.1f}%")
    print(f"  waves/total: {gdf_out.cost_waves.mean() / total * 100:.1f}%")
    print(f"  wind/total:  {gdf_out.cost_wind.mean() / total * 100:.1f}%")

    print(f"\n=== Current Effects ===")
    for col in ["delta_current_on_calm", "delta_current_on_waves", "delta_current_total"]:
        val = gdf_out[col].mean()
        sign = "favorable" if val > 0 else "adverse"
        print(f"  {col}: {val:.4e} ({sign})")

    n_hazardous = gdf_out.is_hazardous.sum()
    n_total = len(gdf_out)
    print(f"\n=== Hazard Diagnosis ===")
    print(f"  Hazardous routes: {n_hazardous}/{n_total} ({100*n_hazardous/n_total:.1f}%)")
    print(f"  Max wave height range: {gdf_out.max_wave_height_m.min():.1f} - {gdf_out.max_wave_height_m.max():.1f} m")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out_path = "../results/baseline_elites_decomposed.geoparquet"
    gpd.GeoDataFrame(gdf_out).to_parquet(out_path)
    print(f"\nSaved {len(gdf_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
