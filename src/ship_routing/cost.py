from .currents import select_data_for_leg

import numpy as np


from dataclasses import dataclass


@dataclass(frozen=True)
class Physics:
    gravity_acceleration_ms2: float = 9.80665
    sea_water_density_kgm3: float = 1029.0
    air_density_kgm3: float = 1.225


@dataclass(frozen=True)
class Ship:
    waterline_width_m: float = 30.0
    waterline_length_m: float = 210.0
    total_propulsive_efficiency: float = 0.7
    reference_engine_power_W: float = 14296344.0
    referemce_speed_calm_water_ms: float = 9.259
    draught_m: float = 11.5
    supersurface_area_m2: float = 345.0
    wind_resistance_coefficient: float = 0.4
    reference_froede_number: float = 0.25


def power_maintain_speed(
    u_ship_og_ms: float = None,
    v_ship_og_ms: float = None,
    u_current_ms: float = 0.0,
    v_current_ms: float = 0.0,
    u_wind_ms: float = 0.0,
    v_wind_ms: float = 0.0,
    physics: Physics = Physics(),
    ship: Ship = Ship(),
) -> float:
    """Calculate power needed to maintain speed over ground.

    Parameters
    ----------
    u_ship_og: array
        Ship eastward speed over ground in meters per second.
    v_ship_og: array
        Ship northward speed over ground in meters per second.
        Needs same shape as u_ship_og.
    u_current: array
        Ocean currents eastward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    v_current: array
        Ocean currents northward speed over ground in meters per second.
        Needs shape that can be broadcast to shape of u_ship_og and v_ship_og.
    u_wind: array
        Eastward 10 m wind in m/s
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
    v_wind: array
        Northward 10 m wind in m/s
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
    w_wave_height: array
        Spectral significant wave height (Hm0), meters
        Needs shape that can be broadcst to shape of u_ship_og and v_ship_og
    physics: Physics
        Physics parameters.
    ship: Ship
        Ship parameters.

    -------
    array:
        Power in W (=kg*m2/s3) needed to maintain speed over ground for each
        element of u_ship_og and v_ship_og. Shape will be identical to
        u_ship_og and v_ship_og.
    """

    raise NotImplementedError()
