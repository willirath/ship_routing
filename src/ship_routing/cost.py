from .config import Ship, Physics, SHIP_DEFAULT, PHYSICS_DEFAULT
from .cost_ufuncs import power_maintain_speed_ufunc, hazard_conditions_wave_height_ufunc


import numpy as np
import xarray as xr


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


def maybe_cast_number_to_data_array(obj):
    """Make obj a data array with one along-track point."""
    if np.array(obj).shape == ():
        obj = xr.DataArray(
            [float(obj), float(obj)],
            dims=("along",),
            coords={"along": [0.0, 1.0]},
        )
    return obj


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
