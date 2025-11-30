# Repository Reorganization Summary

## Completed: Three-Layer Architecture

The repository has been successfully reorganized into three clear layers as planned.

### New Structure

```
src/ship_routing/
├── core/               # Layer 1: Core functionality
│   ├── __init__.py
│   ├── routes.py       # WayPoint, Leg, Route (formerly core.py)
│   ├── geodesics.py
│   ├── cost.py
│   ├── cost_ufuncs.py
│   ├── remix.py
│   ├── data.py
│   ├── hashable_dataset.py
│   └── config.py       # Ship, Physics, MAX_CACHE_SIZE
├── algorithms/         # Layer 2: Algorithm building blocks
│   ├── __init__.py
│   ├── optimization.py # All gradient descent, crossover, stochastic search
│   └── logging.py      # Logs, LogsRoute (temporary, to be removed later)
└── app/                # Layer 3: User-facing application
    ├── __init__.py
    ├── routing.py      # RoutingApp (formerly app.py)
    └── config.py       # All user-facing configs

tests/
├── core/               # Core layer tests
├── algorithms/         # Algorithm layer tests
└── app/                # App layer tests
```

### Key Changes

1. **Config Split**: `config.py` split into `core/config.py` and `app/config.py`
   - Core: Ship, Physics, MAX_CACHE_SIZE
   - App: All user-facing configs (JourneyConfig, RoutingConfig, etc.)

2. **core.py → core/routes.py**:
   - Renamed for clarity
   - Added `Route.create_route()` classmethod (moved from `convenience.py`)

3. **algorithms/optimization.py**: Merged content from:
   - `algorithms.py` (gradient descent primitives, crossover)
   - `convenience.py` (stochastic_search, gradient_descent wrapper)

4. **app.py → app/routing.py**: Renamed for clarity

5. **Removed Files**:
   - `convenience.py` - split across layers
   - Old `core.py`, `algorithms.py`, `app.py`, `config.py` - replaced by new structure

### Import Changes

**Before:**
```python
from ship_routing.core import Route
from ship_routing.data import load_currents
from ship_routing.algorithms import gradient_descent_time_shift
from ship_routing.app import RoutingApp
from ship_routing.config import RoutingConfig
```

**After:**
```python
from ship_routing.core import Route
from ship_routing.core.data import load_currents
from ship_routing.algorithms import gradient_descent_time_shift
from ship_routing.app import RoutingApp, RoutingConfig
```

### Test Results

All 185 tests passing ✓

### Migration Notes

1. **create_route()** is now a classmethod: `Route.create_route(...)`
2. **Logs/LogsRoute** moved to `algorithms.logging` (temporary, will be removed)
3. **No backwards compatibility** - explicit layer imports required
4. **Test structure mirrors code** - tests organized into core/, algorithms/, app/

### Next Steps

- Consider removing `Logs`/`LogsRoute` from algorithms layer
- May refactor selection logic from app to algorithms layer
- Full test coverage maintained for core, decent for algorithms