#!/usr/bin/env python
"""Micro-benchmark for the route cost hot path.

Isolates `Route.cost_through` from the surrounding algorithm so optimisation
work gets a sub-second feedback loop instead of a 2-minute full run. Also
breaks the per-leg cost down into its constituent parts (data selection,
power ufunc, hazard check) and records reference costs for regression checks.

Usage
-----
    pixi run python dev/profiling/bench_leg_cost.py
    pixi run python dev/profiling/bench_leg_cost.py --n-routes 20 --check baseline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np

from ship_routing.core.routes import Leg, Route, WayPoint
from ship_routing.core.cost import (
    hazard_conditions_wave_height,
    power_maintain_speed,
)
from ship_routing.core.data import (
    load_currents,
    load_waves,
    load_winds,
    select_data_for_leg,
)
from ship_routing.core.geodesics import compute_ellipse_bbox

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "large"
OUT = Path(__file__).resolve().parent / "out"

CURRENTS = (
    DATA
    / "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
)
WAVES = (
    DATA
    / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
)
WINDS = (
    DATA
    / "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
)

# Production representative case: Atlantic forward, January, 12 kn, 6 h steps.
LON = (-80.5, -11.0)
LAT = (30.0, 50.0)
TIME_START = np.datetime64("2021-01-01T00:00:00")
SPEED_KNOTS = 12.0
TIME_RESOLUTION_HOURS = 6.0


def load_forcing():
    bbox = compute_ellipse_bbox(
        lon_start=LON[0],
        lat_start=LAT[0],
        lon_end=LON[-1],
        lat_end=LAT[-1],
        length_multiplier=1.5,
        buffer_degrees=0.5,
    )
    from ship_routing.core.geodesics import knots_to_ms, get_length_meters
    from shapely.geometry import LineString

    length_m = get_length_meters(LineString(zip(LON, LAT)))
    duration_s = length_m / knots_to_ms(SPEED_KNOTS)
    time_end = TIME_START + np.timedelta64(int(duration_s), "s")

    kw = dict(
        time_start=TIME_START,
        time_end=time_end,
        load_eagerly=True,
        engine="zarr",
        chunks="auto",
        spatial_bounds=bbox,
    )
    return (
        load_currents(data_file=CURRENTS, **kw),
        load_winds(data_file=WINDS, **kw),
        load_waves(data_file=WAVES, **kw),
        time_end,
    )


def build_route(time_end) -> Route:
    """Great-circle seed route at production time resolution."""
    route = Route(
        way_points=(
            WayPoint(lon=LON[0], lat=LAT[0], time=TIME_START),
            WayPoint(lon=LON[-1], lat=LAT[-1], time=time_end),
        )
    )
    duration_s = float(
        (time_end - TIME_START) / np.timedelta64(1, "s")
    )
    n_legs = max(1, int(round(duration_s / (TIME_RESOLUTION_HOURS * 3600.0))))
    return route.refine(distance_meters=route.length_meters / n_legs)


def jitter(route: Route, rng, meters=20_000.0) -> Route:
    """Perturb interior waypoints, mimicking a mutation step."""
    wps = list(route.way_points)
    for n in range(1, len(wps) - 1):
        wps[n] = wps[n].move_space(
            azimuth_degrees=float(rng.uniform(0, 360)),
            distance_meters=float(rng.uniform(0, meters)),
        )
    return Route(way_points=tuple(wps))


def clear_caches():
    for fn in (
        select_data_for_leg,
        Leg.cost_through,
        Route.cost_through,
        Route.cost_per_leg_through,
    ):
        try:
            fn.cache_clear()
        except AttributeError:
            pass


@click.command()
@click.option("--n-routes", type=int, default=20, help="Distinct routes to cost.")
@click.option("--tag", type=str, default="bench", help="Label for reference costs.")
@click.option(
    "--check",
    type=str,
    default=None,
    help="Compare costs against a previously saved tag.",
)
def main(n_routes, tag, check):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    currents, winds, waves, time_end = load_forcing()
    print(f"forcing loaded in {time.perf_counter() - t0:.2f} s")
    for name, ds in (("currents", currents), ("winds", winds), ("waves", waves)):
        print(f"  {name}: {dict(ds.sizes)}")

    route = build_route(time_end)
    print(f"route: {len(route.way_points)} waypoints, {len(route.legs)} legs")

    rng = np.random.default_rng(0)
    routes = [route] + [jitter(route, rng) for _ in range(n_routes - 1)]

    # --- whole-route cost ---
    clear_caches()
    t0 = time.perf_counter()
    costs = [
        r.cost_through(
            current_data_set=currents, wind_data_set=winds, wave_data_set=waves
        )
        for r in routes
    ]
    elapsed = time.perf_counter() - t0
    n_legs_total = sum(len(r.legs) for r in routes)
    print(
        f"\ncost_through: {elapsed:.3f} s for {n_routes} routes "
        f"({n_legs_total} legs) -> {1e6 * elapsed / n_legs_total:.1f} us/leg"
    )

    # --- component breakdown on one uncached route ---
    clear_caches()
    leg = routes[-1].legs[len(routes[-1].legs) // 2]
    reps = 200

    def timeit(fn, reps=reps):
        clear_caches()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return 1e6 * (time.perf_counter() - t0) / reps

    sel_kw = dict(
        lon_start=leg.way_point_start.lon,
        lat_start=leg.way_point_start.lat,
        time_start=leg.way_point_start.time,
        lon_end=leg.way_point_end.lon,
        lat_end=leg.way_point_end.lat,
        time_end=leg.way_point_end.time,
    )
    us_sel_cold = timeit(lambda: select_data_for_leg(ds=currents, **sel_kw))
    ds_c = select_data_for_leg(ds=currents, **sel_kw)
    ds_w = select_data_for_leg(ds=winds, **sel_kw)
    ds_v = select_data_for_leg(ds=waves, **sel_kw)
    u_og, v_og = leg.uv_over_ground_ms

    us_sel_warm = 1e6 * _warm_select(currents, sel_kw, reps)
    us_power = _plain_time(
        lambda: power_maintain_speed(
            u_current_ms=ds_c.uo,
            v_current_ms=ds_c.vo,
            u_wind_ms=ds_w.uw,
            v_wind_ms=ds_w.vw,
            w_wave_height=ds_v.wh,
            u_ship_og_ms=u_og,
            v_ship_og_ms=v_og,
        ),
        reps,
    )
    us_hazard = _plain_time(
        lambda: hazard_conditions_wave_height(
            u_current_ms=ds_c.uo,
            v_current_ms=ds_c.vo,
            u_wind_ms=ds_w.uw,
            v_wind_ms=ds_w.vw,
            w_wave_height=ds_v.wh,
            u_ship_og_ms=u_og,
            v_ship_og_ms=v_og,
        ),
        reps,
    )
    print("\n=== per-call component costs (microseconds) ===")
    print(f"  select_data_for_leg  (cold, 1 dataset): {us_sel_cold:9.1f}")
    print(f"  select_data_for_leg  (warm, 1 dataset): {us_sel_warm:9.1f}")
    print(f"  power_maintain_speed (7-way align)    : {us_power:9.1f}")
    print(f"  hazard_conditions    (7-way align)    : {us_hazard:9.1f}")
    print(
        f"  => modelled cold leg  (3x sel + power + hazard): "
        f"{3 * us_sel_cold + us_power + us_hazard:9.1f}"
    )

    # --- reference costs for regression checking ---
    ref_path = OUT / f"{tag}_route_costs.json"
    payload = {"n_routes": n_routes, "costs": [float(c) for c in costs]}
    ref_path.write_text(json.dumps(payload, indent=2))
    print(f"\nreference costs -> {ref_path}")

    if check:
        prev = json.loads((OUT / f"{check}_route_costs.json").read_text())
        a = np.array(prev["costs"])
        b = np.array(payload["costs"])
        if a.shape != b.shape:
            print(f"!! shape mismatch vs {check}: {a.shape} vs {b.shape}")
        else:
            rel = np.abs(b - a) / np.abs(a)
            print(
                f"\nvs {check}: max rel diff {rel.max():.3e}, "
                f"exact matches {int((a == b).sum())}/{len(a)}"
            )


def _warm_select(ds, sel_kw, reps):
    select_data_for_leg(ds=ds, **sel_kw)
    t0 = time.perf_counter()
    for _ in range(reps):
        select_data_for_leg(ds=ds, **sel_kw)
    return (time.perf_counter() - t0) / reps


def _plain_time(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1e6 * (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    main()
