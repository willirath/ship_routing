from functools import lru_cache

from .config import (
    Ship,
    Physics,
    SHIP_DEFAULT,
    PHYSICS_DEFAULT,
)  # TODO: replace globals with explicit config wiring
from .cost_ufuncs import (
    power_maintain_speed_ufunc,
    power_maintain_speed_decomposed_ufunc,
    hazard_conditions_wave_height_ufunc,
)


import numpy as np
import xarray as xr

# Fallback for @profile decorator when not using line_profiler
try:
    profile
except NameError:

    def profile(func):
        return func


@profile
def align_along_track_arrays(*argv) -> tuple:
    """Align all fields on their `along` dimension."""
    # find longest array
    along_sizes = [a.sizes["along"] for a in argv]
    i_of_longest = np.argmax(along_sizes)
    a_ref = argv[i_of_longest]

    # interpolate shorter ones on the longer
    return tuple(
        (
            a.sel(along=a_ref.along, method="nearest").assign_coords(along=a_ref.along)
            for a in argv
        )
    )


# --- numpy-native fast path -------------------------------------------------
#
# The array-level helpers above are the readable reference implementation, but
# they carry xarray/pandas overhead on every call. Along-track arrays are only
# a few tens of elements long, so that overhead dominates the actual arithmetic
# by two orders of magnitude. The helpers below do the same thing on raw numpy
# and are what the per-leg cost evaluation uses.
#
# All `along` coordinates are `np.linspace(0, 1, n)` (see core.data), so the
# nearest-neighbour index map depends only on the pair of sizes and can be
# cached. The tie-breaking rule (`dl < dr`, i.e. ties resolve to the higher
# index) reproduces `DataArray.sel(along=..., method="nearest")` exactly; this
# is verified for all size pairs up to 90 in tests/core/test_cost_values.py.


@lru_cache(maxsize=4096)
def nearest_along_index(n_src: int, n_ref: int) -> np.ndarray:
    """Index into a length-``n_src`` along-axis nearest each length-``n_ref`` point.

    Both axes are assumed to be ``np.linspace(0, 1, n)``, matching how the
    ``along`` coordinate is constructed during data selection.

    Parameters
    ----------
    n_src : int
        Length of the source along-axis.
    n_ref : int
        Length of the reference along-axis to resample onto.

    Returns
    -------
    np.ndarray
        Integer index array of length ``n_ref``. Read-only, because it is
        shared between callers via the cache.
    """
    if n_src == n_ref:
        idx = np.arange(n_ref)
        idx.flags.writeable = False
        return idx

    src = np.linspace(0.0, 1.0, n_src)
    ref = np.linspace(0.0, 1.0, n_ref)
    pos = np.searchsorted(src, ref)
    left = np.clip(pos - 1, 0, n_src - 1)
    right = np.clip(pos, 0, n_src - 1)
    # Strict `<` puts exact ties on the higher index, matching pandas/xarray.
    idx = np.where(np.abs(ref - src[left]) < np.abs(src[right] - ref), left, right)
    idx.flags.writeable = False
    return idx


def maybe_cast_number_to_values(obj) -> np.ndarray:
    """Return ``obj`` as a 1-D numpy array along track.

    Scalars become two-element arrays, mirroring
    :func:`maybe_cast_number_to_data_array`, so that the choice of reference
    axis in :func:`align_along_track_values` is unchanged.
    """
    values = getattr(obj, "values", obj)
    values = np.asarray(values)
    if values.shape == ():
        return np.full(2, float(values))
    return values


@profile
def align_along_track_values(*argv) -> tuple:
    """Align 1-D numpy arrays on their along-track axis.

    Numpy equivalent of :func:`align_along_track_arrays`: every array is
    resampled with nearest-neighbour onto the longest array's axis.
    """
    sizes = [a.size for a in argv]
    n_ref = sizes[int(np.argmax(sizes))]
    return tuple(
        a if a.size == n_ref else a[nearest_along_index(a.size, n_ref)] for a in argv
    )


@profile
def power_and_hazard_values(
    u_ship_og_ms=0.0,
    v_ship_og_ms=0.0,
    u_current_ms=0.0,
    v_current_ms=0.0,
    u_wind_ms=0.0,
    v_wind_ms=0.0,
    w_wave_height=0.0,
    physics: Physics = PHYSICS_DEFAULT,
    ship: Ship = SHIP_DEFAULT,
) -> tuple:
    """Along-track power and hazard flags for one leg, on raw numpy arrays.

    Combines :func:`power_maintain_speed` and
    :func:`hazard_conditions_wave_height` so the along-track alignment is paid
    for once instead of twice. Inputs may be xarray DataArrays, numpy arrays or
    scalars.

    Parameters
    ----------
    (same as power_maintain_speed)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(power_w, hazard)`` along track: power in W and a boolean array
        flagging where stability thresholds are violated.
    """
    (
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    ) = align_along_track_values(
        maybe_cast_number_to_values(u_ship_og_ms),
        maybe_cast_number_to_values(v_ship_og_ms),
        maybe_cast_number_to_values(u_current_ms),
        maybe_cast_number_to_values(v_current_ms),
        maybe_cast_number_to_values(u_wind_ms),
        maybe_cast_number_to_values(v_wind_ms),
        maybe_cast_number_to_values(w_wave_height),
    )

    power = power_maintain_speed_ufunc(
        u_ship_og_ms=u_ship_og_ms,
        v_ship_og_ms=v_ship_og_ms,
        u_current_ms=u_current_ms,
        v_current_ms=v_current_ms,
        u_wind_ms=u_wind_ms,
        v_wind_ms=v_wind_ms,
        w_wave_height=w_wave_height,
        physics=physics,
        ship=ship,
    )
    hazard = hazard_conditions_wave_height_ufunc(
        w_wave_height_m=w_wave_height,
        ship=ship,
    )
    return power, hazard


@profile
def maybe_cast_number_to_data_array(obj):
    """Make obj a data array with one along-track point."""
    if np.array(obj).shape == ():
        obj = xr.DataArray(
            [float(obj), float(obj)],
            dims=("along",),
            coords={"along": [0.0, 1.0]},
        )
    return obj


@profile
def power_maintain_speed(
    u_ship_og_ms: xr.DataArray = 0.0,
    v_ship_og_ms: xr.DataArray = 0.0,
    u_current_ms: xr.DataArray = 0.0,
    v_current_ms: xr.DataArray = 0.0,
    u_wind_ms: xr.DataArray = 0.0,
    v_wind_ms: xr.DataArray = 0.0,
    w_wave_height: xr.DataArray = 0.0,
    physics: Physics = PHYSICS_DEFAULT,
    ship: Ship = SHIP_DEFAULT,
):
    """Calculate power needed to maintain speed over ground along a route.

    Wrapper for power_maintain_speed_ufunc that handles xarray DataArrays with
    along-track dimensions. Aligns all input arrays and applies the power
    calculation along the route.

    Parameters
    ----------
    u_ship_og_ms : xr.DataArray, default=0.0
        Ship eastward speed over ground in m/s
    v_ship_og_ms : xr.DataArray, default=0.0
        Ship northward speed over ground in m/s
    u_current_ms : xr.DataArray, default=0.0
        Ocean currents eastward speed in m/s
    v_current_ms : xr.DataArray, default=0.0
        Ocean currents northward speed in m/s
    u_wind_ms : xr.DataArray, default=0.0
        Eastward 10 m wind in m/s
    v_wind_ms : xr.DataArray, default=0.0
        Northward 10 m wind in m/s
    w_wave_height : xr.DataArray, default=0.0
        Spectral significant wave height in m
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters
    ship : Ship, default=SHIP_DEFAULT
        Ship parameters

    Returns
    -------
    xr.DataArray
        Power in W (=kg*m2/s3) needed to maintain speed over ground along track
    """
    # cast all to arrays
    u_ship_og_ms = maybe_cast_number_to_data_array(u_ship_og_ms)
    v_ship_og_ms = maybe_cast_number_to_data_array(v_ship_og_ms)
    u_current_ms = maybe_cast_number_to_data_array(u_current_ms)
    v_current_ms = maybe_cast_number_to_data_array(v_current_ms)
    u_wind_ms = maybe_cast_number_to_data_array(u_wind_ms)
    v_wind_ms = maybe_cast_number_to_data_array(v_wind_ms)
    w_wave_height = maybe_cast_number_to_data_array(w_wave_height)

    # align all
    (
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    ) = align_along_track_arrays(
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    )

    # calc power
    return power_maintain_speed_ufunc(
        u_ship_og_ms=u_ship_og_ms,
        v_ship_og_ms=v_ship_og_ms,
        u_current_ms=u_current_ms,
        v_current_ms=v_current_ms,
        u_wind_ms=u_wind_ms,
        v_wind_ms=v_wind_ms,
        w_wave_height=w_wave_height,
        physics=physics,
        ship=ship,
    )


@profile
def power_maintain_speed_decomposed(
    u_ship_og_ms: xr.DataArray = 0.0,
    v_ship_og_ms: xr.DataArray = 0.0,
    u_current_ms: xr.DataArray = 0.0,
    v_current_ms: xr.DataArray = 0.0,
    u_wind_ms: xr.DataArray = 0.0,
    v_wind_ms: xr.DataArray = 0.0,
    w_wave_height: xr.DataArray = 0.0,
    physics: Physics = PHYSICS_DEFAULT,
    ship: Ship = SHIP_DEFAULT,
) -> tuple:
    """Calculate decomposed power components along a route.

    Wrapper for power_maintain_speed_decomposed_ufunc that handles xarray
    DataArrays. Components sum exactly to power_maintain_speed output.

    Parameters
    ----------
    (same as power_maintain_speed)

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        (power_calm, power_waves, power_wind) along track in W
    """
    # cast all to arrays
    u_ship_og_ms = maybe_cast_number_to_data_array(u_ship_og_ms)
    v_ship_og_ms = maybe_cast_number_to_data_array(v_ship_og_ms)
    u_current_ms = maybe_cast_number_to_data_array(u_current_ms)
    v_current_ms = maybe_cast_number_to_data_array(v_current_ms)
    u_wind_ms = maybe_cast_number_to_data_array(u_wind_ms)
    v_wind_ms = maybe_cast_number_to_data_array(v_wind_ms)
    w_wave_height = maybe_cast_number_to_data_array(w_wave_height)

    # align all
    (
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    ) = align_along_track_arrays(
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    )

    return power_maintain_speed_decomposed_ufunc(
        u_ship_og_ms=u_ship_og_ms,
        v_ship_og_ms=v_ship_og_ms,
        u_current_ms=u_current_ms,
        v_current_ms=v_current_ms,
        u_wind_ms=u_wind_ms,
        v_wind_ms=v_wind_ms,
        w_wave_height=w_wave_height,
        physics=physics,
        ship=ship,
    )


@profile
def hazard_conditions_wave_height(
    u_ship_og_ms: xr.DataArray = 0.0,
    v_ship_og_ms: xr.DataArray = 0.0,
    u_current_ms: xr.DataArray = 0.0,
    v_current_ms: xr.DataArray = 0.0,
    u_wind_ms: xr.DataArray = 0.0,
    v_wind_ms: xr.DataArray = 0.0,
    w_wave_height: xr.DataArray = 0.0,
    physics: Physics = PHYSICS_DEFAULT,
    ship: Ship = SHIP_DEFAULT,
):
    """Check stability thresholds for wave heights along a route.

    Wrapper for hazard_conditions_wave_height_ufunc that handles xarray DataArrays
    with along-track dimensions. Aligns all input arrays and checks wave height
    stability criteria along the route.

    Parameters
    ----------
    u_ship_og_ms : xr.DataArray, default=0.0
        Ship eastward speed over ground in m/s (unused but kept for consistency)
    v_ship_og_ms : xr.DataArray, default=0.0
        Ship northward speed over ground in m/s (unused but kept for consistency)
    u_current_ms : xr.DataArray, default=0.0
        Ocean currents eastward speed in m/s (unused but kept for consistency)
    v_current_ms : xr.DataArray, default=0.0
        Ocean currents northward speed in m/s (unused but kept for consistency)
    u_wind_ms : xr.DataArray, default=0.0
        Eastward 10 m wind in m/s (unused but kept for consistency)
    v_wind_ms : xr.DataArray, default=0.0
        Northward 10 m wind in m/s (unused but kept for consistency)
    w_wave_height : xr.DataArray, default=0.0
        Spectral significant wave height in m
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters (unused but kept for consistency)
    ship : Ship, default=SHIP_DEFAULT
        Ship parameters

    Returns
    -------
    xr.DataArray
        Boolean array indicating whether stability thresholds are violated along track
    """
    # cast all to arrays
    u_ship_og_ms = maybe_cast_number_to_data_array(u_ship_og_ms)
    v_ship_og_ms = maybe_cast_number_to_data_array(v_ship_og_ms)
    u_current_ms = maybe_cast_number_to_data_array(u_current_ms)
    v_current_ms = maybe_cast_number_to_data_array(v_current_ms)
    u_wind_ms = maybe_cast_number_to_data_array(u_wind_ms)
    v_wind_ms = maybe_cast_number_to_data_array(v_wind_ms)
    w_wave_height = maybe_cast_number_to_data_array(w_wave_height)

    # align all
    (
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    ) = align_along_track_arrays(
        u_ship_og_ms,
        v_ship_og_ms,
        u_current_ms,
        v_current_ms,
        u_wind_ms,
        v_wind_ms,
        w_wave_height,
    )

    return hazard_conditions_wave_height_ufunc(
        w_wave_height_m=w_wave_height,
        ship=ship,
    )
