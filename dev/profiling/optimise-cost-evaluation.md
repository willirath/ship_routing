# Optimise route cost evaluation (ship_routing, branch `perf-lookup`)

Status: **implemented and measured**. 130.17 s -> 4.65 s (**28x**), route costs bit-identical.
Awaiting review. Not merged to main.

## Benchmark definition

`dev/profiling/profile_run.py` — one representative production case (Atlantic forward,
January, 12 kn, 6 h time resolution, 47 waypoints / 46 legs) with production algorithm
structure scaled down to fit a laptop:

    pop=32, offspring=32, generations=2, gd_iterations=1, sequential

Baseline 130.17 s. Regression check: `dev/profiling/bench_leg_cost.py --check baseline`
compares 20 whole-route costs against `dev/profiling/out/baseline_route_costs.json`.

## Where the time actually went

cProfile of the 130 s run: the entire top-22 by self time was xarray internals —
`copy.deepcopy` (21.8M calls), `Variable.__init__` (10.9M), `DataArray.__init__` (1.9M),
`_parse_dimensions`, `merge_collected`. Neither the physics nor the file I/O appeared.

Cause: a leg samples only a few tens of along-track points, but every operation on that
handful of numbers went through xarray, which pays coordinate alignment, `Variable`
construction and attr deep-copies per operation. Measured per leg: 22.6 ms, of which
8.8 ms `power_maintain_speed`, 5.9 ms `hazard_conditions_wave_height`, and most of the
rest cold `select_data_for_leg`.

The initial hypothesis (data selection / lookup is the bottleneck) was **wrong** in
steady state — `select_data_for_leg` is lru_cached down to 0.4 us on a hit. It was
still worth fixing because the *cold* path cost 2.37 ms.

## What changed

### 1. numpy-native arithmetic — `core/cost.py`, `core/routes.py`

Added `power_and_hazard_values()`: pulls `.values` out once, does the alignment and the
~30 elementwise ops on raw numpy, returns plain arrays. `Leg.cost_through` uses it;
`np.isnan(pwr).any()` and `pwr.mean()` replace the xarray reductions.

The `DataArray` functions (`power_maintain_speed`, `hazard_conditions_wave_height`,
`align_along_track_arrays`) are untouched and still public — they are now the readable
reference implementation that the tests check the fast path against.

Also removed a duplicated alignment: `hazard_conditions_wave_height` re-ran the full
7-array alignment but consumed only wave height. Power and hazard now share one pass.

**Numerics gotcha, now pinned by a test.** Alignment is nearest-neighbour resampling onto
the longest array's axis. Since every `along` coord is `linspace(0, 1, n)`, the index map
depends only on `(n_src, n_ref)` and is cached. But closed-form `round(k*(ns-1)/(nr-1))`
does **not** reproduce xarray — `linspace` values do not sit exactly on the analytic
midpoints, so float noise decides ties. Swept 7744 size pairs: the rule that matches is
`searchsorted` with ties resolving to the **higher** index (`dl < dr`).
See `tests/core/test_cost_values.py`.

Result: 22.6 -> 7.19 ms/leg; 130.17 -> 48.58 s.

### 2. Lookup layer above ingest — new `core/lookup.py`

`ForcingGrid` extracts each loaded Dataset's variables and coordinate axes to numpy
**once** (cached on dataset identity); `sample_values_for_leg` then serves per-leg samples
by index arithmetic, replacing the per-leg `assign_coords` -> 4x `.sel(method="nearest")`
-> `.isel` -> `.compute()` chain.

xarray remains the ingest layer — this sits strictly above `core/data.py`, which is
unchanged. `select_data_for_leg` is still used by `cost_through_decomposed` and
`hazard_through`, so the analysis paths and notebooks are unaffected.

Note the coordinate axes are **not** uniformly spaced (float32 grids), so this uses
`searchsorted` on real coordinate values, not spacing arithmetic.

Verified differentially against `select_data_for_leg` on the real grids: 0 mismatches over
1800 arrays / 300 legs x 3 datasets, including deliberate exact-midpoint tie probes.
Cold path 2370 -> 27 us (87x). See `tests/core/test_lookup.py`.

Result: 7.19 ms -> 159 us/leg; 48.58 -> 7.10 s.

### 3. Geodesics — `core/geodesics.py`, `core/routes.py`

- Hoisted `pyproj.Geod` to a module-level `_GEOD_WGS84` (there was a `TODO` for exactly
  this). The profile showed **127,959** `Geod.__init__` calls in one run.
- `Leg.length_meters` was building 2 shapely `Point`s + a `LineString` just to measure a
  two-point distance, on the per-leg cost path via `speed_ms`. Now calls the geodesic
  inverse directly. Bit-identical over 20,000 random pairs; 9.24 -> 0.55 us.

Result: 7.10 -> 4.65 s.

## Verification

- Route costs bit-identical at every step: `max rel diff 0.000e+00, exact matches 20/20`.
- End-to-end elite cost unchanged from baseline: `1.327449e+13` (seed `5.388951e+14`),
  so the algorithm's trajectory is identical, not merely its endpoint.
- `pytest tests/` 321 passed (was 237 + 84 new). `tests/app/test_config_factory.py` is
  excluded — pre-existing breakage, imports `ship_routing.app.config_factory` which moved
  to `ship_routing.htc.config_factory`. Unrelated to this branch; worth a separate fix.

## Findings that did NOT need acting on

- **GD is not the bottleneck.** It was 14 s / 130 s (11%) originally and scaled down with
  everything else; now ~0.14 s per iteration, ~2% of the run.
- **Gradient probes do hit the leg cache.** Measured: `move_space` and `move_time` each
  change 2 of 46 legs (44 hits / 2 misses); a full across-track GD step runs at 95.7% leg-cost
  cache hit rate. Moving a waypoint does not re-time downstream waypoints — `move_space`
  preserves `self.time` and `replace_waypoint` swaps only index `n`.
- **numba not needed.** The arithmetic was never the cost; plain numpy suffices.

## Remaining opportunities (not done)

1. **GD probe deltas.** Each `cost_gradient_*(n)` calls full-route `cost_through()` on two
   perturbed routes — rebuilding 46 `Leg`s, hashing a 47-waypoint `Route`, summing 46
   cached floats — when only 2 legs changed. A delta formulation would cut GD's object
   churn ~20x, but for ~2% of current runtime. **Caveat:** it changes float summation
   order, so it would *not* be bit-identical. Only worth it if GD gets heavier (it scales
   as O(n_waypoints^2) in object overhead, so it matters at finer time resolution).
2. **Shapely in `Route.length_meters` / `refine`.** Still the largest remaining item in the
   profile. Called per member per mutation rather than per leg.
3. **Parallelism.** 6 performance cores idle; `executor_type="process"` already exists.
   Multiplies whatever single-thread cost remains — but note it changes RNG streams, so
   results would no longer be bit-comparable to sequential runs.

## USER COMMENTS:


