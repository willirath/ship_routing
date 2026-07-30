# Lazy forcing via a block cache (ship_routing, branch `perf-lookup`)

Status: **implemented and measured**. Routes only ever touch a sliver of the
forcing cube, so the cube no longer has to be resident. Peak RSS 2743 -> 334 MiB
(**8.2x**) at 1.36x wall clock, route costs bit-identical. Routing directly
against Copernicus Marine now works end to end. Not merged to main.

## The observation

For a representative 32-member population (46 legs each, built with the real
warmup mutation), the fraction of the cropped cube that any route actually
samples is:

| store    | cropped cube | cells touched | fraction |
|----------|--------------|---------------|----------|
| currents | 13x661x1215  | 24188         | 0.2317 % |
| winds    | 276x440x810  | 16339         | 0.0166 % |
| waves    | 93x276x507   | 9610          | 0.0738 % |

Eagerly cropped, those cubes are 929 MiB together. A route is a diagonal through
`(time, lat, lon)`; nearly everything else is loaded and never read.

## Why laziness alone does not help

Chunk granularity, not cell count, decides what a lazy read costs. On the
production stores the chunks are full spatial slabs -- currents `(1,661,1321)`,
waves `(40,276,551)`, winds `(120,440,880)` = 186 MB -- so sampling lazily
fetches *more* than the eager crop:

| store    | eager  | lazy, as stored |
|----------|--------|-----------------|
| currents | 80 MiB | 80 MiB (100 %)  |
| winds    | 750    | 1064 (142 %)    |
| waves    | 99     | 139 (140 %)     |

So rechunking is the enabling prerequisite, not an optimisation on top.

## Sizing the blocks

Two costs pull in opposite directions, and the second one was not obvious:

1. **Bytes.** Smaller blocks fetch less.
2. **Calls.** zarr-python 3 has a **~318 us per-call floor even from an
   in-memory store** -- the async `sync()` machinery, not I/O. Sharding adds
   ~365 us, decompression ~256 us.

Because the floor is per *call*, the byte-optimal shape is far from the
time-optimal one. Measured cold-fill of the whole working set for the population
above, reading from a `(4,16,16)`-chunked store:

| block shape   | resident | blocks | cold fill |
|---------------|----------|--------|-----------|
| (4, 16, 16)   | 25.2 MiB | 3226   | 5.14 s    |
| (8, 32, 32)   | 65.4     | 1047   | **3.98 s**|
| (12, 32, 32)  | 79.6     | 849    | 4.45 s    |
| (16, 64, 64)  | 148.0    | 296    | 6.53 s    |
| (24, 128, 128)| 315.0    | 105    | 11.59 s   |

`(8,32,32)` is the sweet spot and is the default. Note a block need not equal a
stored chunk -- a larger block just reads several chunks in one call -- but a
block *smaller* than the chunk is a pessimisation, since the store still
decompresses whole chunks. Measured: block `(4,16,16)` on a `(8,32,32)` store
runs 7.91 s vs 5.22 s for block = chunk.

**Sharding costs more than it saves here.** Same data, same block shape:
sharded 8.63 s vs unsharded 5.22 s. Sharding exists to keep file counts sane,
but at `(8,32,32)` the counts are already fine (2591 / 3129 / 25489 files) and
compression is marginally *better* than at `(4,16,16)` (1.96-2.91x). So the
recommendation is small chunks, no shards.

## What changed

### 1. `BlockCachedForcingGrid` -- new, in `core/lookup.py`

A lazy counterpart to `ForcingGrid` behind the identical `sample_leg` interface.
It tiles the domain, fetches only the blocks legs land in, and holds them in an
LRU cache with a byte budget. `grid_for()` dispatches on
`dataset_is_in_memory(ds)`, so nothing above this layer changes: `core/data.py`
and `Leg.cost_through` are untouched, and setting `load_eagerly=False` is the
whole switch.

Both grids share `leg_indices()`, so they cannot drift apart on index arithmetic
or nearest-neighbour tie-breaking.

Missing blocks are fetched together through a thread pool rather than one at a
time. Locally that recovers ~1.6x; against a remote store, where each read is a
network round trip, it is the difference between usable and not.

### 2. Index-keyed sample cache -- `SampleCacheMixin`

The per-leg cache used to sit on `sample_values_for_leg`, keyed on the raw
coordinates plus the dataset. Two problems:

- **It never hit.** Instrumenting a real run: 5850 calls per dataset, 5850
  distinct float keys. A 100 000-entry `lru_cache` with a **0 % hit rate**,
  because `Leg.cost_through` already absorbs every repeat above it.
- The dataset was in the key only because the function was free-standing, which
  is what forced `HashableDataset` into this path.

Samples depend only on the cells a leg snaps to, so the indices are the
canonical key -- and they are free, because `sample_leg` computes them anyway.
Moving the cache onto the grid (one grid per dataset) drops the dataset from the
key and collapses 30-37 % of the entries. Measured hit rate: **33.4 %**.

Wall-clock effect is ~2 %, inside run-to-run noise; the reason to do it is that
it replaces a cache that cost memory and returned nothing.

### 3. `core/data.py` accepts an open Dataset

`_open()` passes an `xr.Dataset` straight through instead of opening a path.
Three lines, and it is what lets `copernicusmarine.open_dataset` feed the
existing loaders without a detour through disk.

## Results

Benchmark as before: `dev/profiling/profile_run.py`, pop=32, offspring=32,
generations=2, gd=1, sequential.

| path                              | wall  | peak RSS | block cache | elite cost   |
|-----------------------------------|-------|----------|-------------|--------------|
| eager (production store)          | 4.00s | 2743 MiB | -           | 1.327449e+13 |
| lazy, rechunked `(8,32,32)`       | 5.43s | **334**  | 65.5 MiB    | 1.327449e+13 |

Bit-identical elite cost across every block shape and store variant tried
(`(4,16,16)`, `(8,32,32)`, `(12,32,32)`, `(16,64,64)`, `(8,64,64)`, sharded and
unsharded). `pytest tests/` 380 passed (321 before, 59 new).
`tests/app/test_config_factory.py` still excluded -- pre-existing breakage,
imports `ship_routing.app.config_factory`, which moved to `ship_routing.htc`.

The trade is 1.36x wall clock for 8.2x memory. That is worth taking mainly
because of what the memory buys: 12 worker processes at ~334 MiB fit on this
machine where 12 x 2743 MiB does not.

## Copernicus Marine

`dev/profiling/cmems_run.py`. Two findings, and they point in opposite
directions.

**Sampling the ARCO stores directly does not work.** The block cache reads them
correctly -- real values, warm legs at 147 us -- but CMEMS chunk shapes are laid
out for time-series or map extraction, not for a route's diagonal: currents
`(50,2,512,2048)`, winds `(50,1024,1024)`, waves `(17720,16,200)`. One small
block pulls a few hundred MB and discards over 99 % of it. Measured: **11.4 s
per cold block**, which is ~3.5 hours for a run of the size above.

**Mirroring the subset first does.** `copernicusmarine` subsets server-side, and
the ellipse bbox plus journey window is small. Mirroring all three products for
the full 12-day Atlantic journey took 152 s (currents 21.5 s, winds 118.2 s,
waves 12.8 s; 577 MiB on disk at `(8,32,32)`), after which a full routing run
took **5.49 s** with a 66.4 MiB block cache and 95-98 % block hit rates.

So CMEMS is usable as an *ingest* source, not a *sampling* source. The seam is
already in the right place -- the mirror step is `rechunk.py` pointed at a
remote dataset.

Note the CMEMS run's costs (seed 5.398289e+14, elite 1.327585e+13) differ
slightly from the local ones. That is expected and not a bug: `data/large` is
itself a subset (lat +10..+65, lon -100..+010) while the CMEMS mirror covers the
full ellipse bbox (lat 4.46..73.89), so grid extents differ and edge cells snap
differently.

## Decisions worth your input

### D1. Should the rechunked store replace the archive copies?

Right now `rechunk.py` writes a derived copy beside `data/large`, and both are
gitignored. Keeping both costs disk but leaves the eager path available as the
reference the tests check against.

**Suggestion: keep both.** The dense grid is the reference implementation for
the differential tests, and the eager path is still faster in wall clock.

Alternative: rechunk in place and drop the dense path. Simpler, but then there
is nothing to check the block cache against.

**USER FEEDBACK:**


### D2. Should `enable_spatial_cropping` go away?

With a block cache the ellipse bbox is no longer needed to bound memory -- the
cache becomes the corridor, discovered rather than guessed. Dropping it would
also remove the `route_length_multiplier=1.5` heuristic and the risk of a route
silently hitting a bbox edge.

**Suggestion: keep it for now, revisit after parallelism.** It still helps the
eager path, and changing it changes which cells exist, hence costs -- so it
should not ride along with a change that is otherwise bit-identical.

**USER FEEDBACK:**


### D3. Default `max_workers=4` for block fetching?

Threads help ~1.6x locally and much more remotely, but 12 worker processes x 4
threads oversubscribes. The threads are I/O- and decompression-bound and mostly
idle, so I do not expect trouble, but it is untested under process parallelism.

**Suggestion: leave at 4, revisit when process parallelism lands.**

**USER FEEDBACK:**


## Remaining opportunities (not done)

1. **Process parallelism.** This is the point of the memory reduction: 6
   performance cores idle, `executor_type="process"` already exists. Changes RNG
   streams, so results stop being bit-comparable to sequential runs.
2. **GD probe deltas** and **shapely in `Route.length_meters` / `refine`** --
   carried over from `optimise-cost-evaluation.md`, unchanged.
3. **`HashableDataset.__hash__` is `hash(id(ds))`**, which is not sound in
   general: ids are recycled after GC, and `xr.Dataset.__eq__` returns a
   *Dataset*, so a dict-key comparison on a hash collision would raise rather
   than compare. It holds today only because the datasets stay alive for the
   whole run. Now used in fewer places, but still used.

## USER COMMENTS:


