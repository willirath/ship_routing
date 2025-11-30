# Repository Reorganization Plan: Three-Layer Architecture

## Goal

Restructure the repository into three clear layers:

1. **Core Layer**: Routes, waypoints, legs, data structures, route-route interaction, cost calculation, route mutation methods, route-route segmentation
2. **Algorithm Building Blocks Layer**: Gradient descent, stochastic optimization, selection, and other optimization algorithms
3. **User-Facing/App Layer**: High-level route optimization implementations that assemble algorithm building blocks

---

## Current State Analysis

### Existing Files and Their Mapping

#### Layer 1: Core Functionality
- `core.py` - WayPoint, Leg, Route classes and their methods
- `geodesics.py` - Geodesic calculations on WGS84 ellipsoid
- `cost.py` + `cost_ufuncs.py` - Cost calculation functions and physics
- `remix.py` - Route geometry operations (segmentation)
- `data.py` - Environmental data loading and selection
- `hashable_dataset.py` - Caching utility
- `config.py` (partial) - Ship, Physics dataclasses, MAX_CACHE_SIZE

#### Layer 2: Algorithm Building Blocks
- `algorithms.py` - Gradient descent variants, crossover operations
- `convenience.py` (partial) - `stochastic_search()` function

#### Layer 3: User-Facing/App
- `app.py` - RoutingApp orchestrator and pipeline
- `convenience.py` (partial) - `create_route()`, `gradient_descent()` wrapper
- `config.py` (partial) - JourneyConfig, ForcingConfig, RoutingConfig, all GA configs

---

## Decision Points

### 1. Directory Structure

**Options:**

- [X] **Option A**: Subdirectories within ship_routing
  ```
  src/ship_routing/
  ├── core/
  ├── algorithms/
  └── app/
  ```

- [ ] **Option B**: Top-level separation
  ```
  src/
  ├── core/
  ├── algorithms/
  └── app/
  ```

- [ ] **Option C**: Other structure (describe below)

**Your decision:**

Option A: subdirectories / submodules within the ship_routing directory / module

**Rationale:**

 I want a ship_routing package with the respective submodules.

---

### 2. Config File Split

Current `config.py` contains:
- **Core configs**: `Ship`, `Physics`, `MAX_CACHE_SIZE`
- **App configs**: `JourneyConfig`, `ForcingConfig`, `ForcingData`, `PopulationConfig`, `StochasticStageConfig`, `CrossoverConfig`, `SelectionConfig`, `GradientConfig`, `RoutingConfig`

**Options:**

- [X] **Option A**: Split by layer
  - `core/config.py` (Ship, Physics, MAX_CACHE_SIZE)
  - `app/config.py` (all user-facing configs)

- [ ] **Option B**: More granular splitting
  - `core/physics.py` (Ship, Physics)
  - `core/constants.py` (MAX_CACHE_SIZE)
  - `app/journey.py` (JourneyConfig)
  - `app/routing_config.py` (RoutingConfig and all optimization configs)

- [ ] **Option C**: Other approach (describe below)

**Your decision:**

Option A: two layers (code config and user facing config)

**Rationale:**

Sufficient to separate out the user facing stuff.

---

### 3. Stochastic Search Placement

`stochastic_search()` is currently in `convenience.py` but implements an optimization algorithm using `Route.move_waypoints_left_nonlocal()`.

**Options:**

- [X] **Option A**: Move to `algorithms` layer (alongside gradient descent primitives)
- [ ] **Option B**: Keep in a convenience/utilities module
- [ ] **Option C**: Move to `app` layer (user-facing)

**Your decision:**

option A: stochastic search is an algorithm block

**Rationale:**

stochastic search should be where the gradient descent and selection logic is.

---

### 4. Gradient Descent Wrapper Placement

The high-level `gradient_descent()` function in `convenience.py` orchestrates multiple gradient descent variants (time_shift, along_track, across_track) from `algorithms.py`.

**Options:**

- [ ] **Option A**: Keep in convenience utilities
- [X] **Option B**: Move to `algorithms` layer (composition of algorithm primitives)
- [ ] **Option C**: Move to `app` layer (user-facing orchestration)

**Your decision:**

B: Move to algorithms.

**Rationale:**

It's a building block for the user-facing high-level algo in the app.

---

### 5. Convenience Module Fate

Current `convenience.py` contains:
- `create_route()` - Helper to build initial routes
- `stochastic_search()` - Optimization algorithm
- `gradient_descent()` - Orchestration of GD variants
- `Logs`, `LogsRoute` - Logging dataclasses

**Options:**

- [ ] **Option A**: Keep as utilities module, move only optimization functions
- [X] **Option B**: Split entirely across layers
- [ ] **Option C**: Rename to better reflect purpose (suggest name: _____________)
- [ ] **Option D**: Other approach (describe below)

**Your decision:**

B. And call it helpers?

**Rationale:**

There's elements from all layers and there's no inherent value to this convenience.py. Calling it helpers (or any other better idea?) is clearer.

---

### 6. Backwards Compatibility

**Options:**

- [ ] **Option A**: Maintain full backwards compatibility
  - Use `__init__.py` re-exports to preserve existing imports
  - Example: `from ship_routing import Route` still works

- [X] **Option B**: Break compatibility, require explicit layer imports
  - Example: `from ship_routing.core import Route`
  - Update all examples and tests

- [ ] **Option C**: Hybrid approach (describe below)

**Your decision:**

B: Break but adapt existing code.

**Rationale:**

This is still in the dev phase and there's no users in need of backwards compat.

---

## Additional Considerations

### Import Dependencies

Current dependency flow:
```
app.py → algorithms.py, convenience.py, core.py, data.py, config.py
convenience.py → core.py, algorithms.py, config.py
algorithms.py → core.py, config.py, remix.py
core.py → geodesics.py, remix.py, data.py, cost.py, config.py
cost.py → config.py, cost_ufuncs.py
data.py → config.py, hashable_dataset.py
```

Target clean hierarchy:
- **Layer 1 (core)** imports nothing from layers 2 or 3
- **Layer 2 (algorithms)** imports only from layer 1
- **Layer 3 (app)** imports from layers 1 and 2

### Test Organization

Current structure:
```
tests/
├── test_core.py
├── test_algorithms.py
├── test_geodesics.py
├── test_cost.py
├── test_cost_ufuncs.py
├── test_data.py
└── test_remix.py
```

Should tests mirror the new layer structure?

**Your thoughts:**

Yes, the test should mirror these layers. Go for three directories as well? I aim at _full_ coverage for core, at decent coverage for algorithms and I'm not sure about coverage requirements for app for now.

---

## Proposed File Organization

Based on your decisions above, fill in the proposed structure:

```
src/ship_routing/
├── core/
│   ├── __init__.py
│   ├── [list files here]
│   │
│   │
│   │
│   └──
├── algorithms/
│   ├── __init__.py
│   ├── [list files here]
│   │
│   │
│   └──
└── app/
    ├── __init__.py
    ├── [list files here]
    │
    │
    └──
```

**Your proposed organization:**

Sounds good.

---

## Migration Steps

Once structure is decided, the implementation will involve:

1. Create new directory structure
2. Move/split files according to plan
3. Update all internal imports
4. Update `__init__.py` files for backwards compatibility (if desired)
5. Update tests
6. Update examples and documentation
7. Run full test suite to verify
8. Update pyproject.toml if needed

---

## Questions and Notes

Add any additional thoughts, concerns, or questions here:
