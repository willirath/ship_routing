from .data import select_data_for_leg

import numpy as np


from dataclasses import dataclass


@dataclass(frozen=True)
class Physics:
    """Physical constants used in power estimation."""

    gravity_acceleration_ms2: float = 9.80665
    sea_water_density_kgm3: float = 1029.0
    air_density_kgm3: float = 1.225


PHYSICS_DEFAULT = Physics()


@dataclass(frozen=True)
class Ship:
    """Ship dimensions, resistance coefficients, engine characteristics."""

    waterline_width_m: float = 30.0
    waterline_length_m: float = 210.0
    total_propulsive_efficiency: float = 0.7
    reference_engine_power_W: float = 14296344.0
    reference_speed_calm_water_ms: float = 9.259
    draught_m: float = 11.5
    projected_frontal_area_above_waterline_m2: float = 690.0
    wind_resistance_coefficient: float = 0.4


SHIP_DEFAULT = Ship()


def power_maintain_speed(
    u_ship_og_ms: float = None,
    v_ship_og_ms: float = None,
    u_current_ms: float = 0.0,
    v_current_ms: float = 0.0,
    u_wind_ms: float = 0.0,
    v_wind_ms: float = 0.0,
    w_wave_height: float = 0.0,
    physics: Physics = PHYSICS_DEFAULT,
    ship: Ship = SHIP_DEFAULT,
) -> float:
    """Calculate power needed to maintain speed over ground.

    This largely implements the resistance estimates due to calm water and sea
    state as outlined in
        Mannarini, G., Pinardi, N., Coppini, G., Oddo, P., and Iafrati, A.:
        VISIR-I: small vessels – least-time nautical routes using wave forecasts,
        Geosci. Model Dev., 9, 1597–1625, 2016. https://doi.org/10.5194/gmd-9-1597-2016
    and wind resistance as outlined in
        Kim K-S, Roh M-I. ISO 15016:2015-Based Method for Estimating the Fuel Oil Consumption
        of a Ship. Journal of Marine Science and Engineering. 2020; 8(10):791.
        https://doi.org/10.3390/jmse8100791

    Note that for zero wave height, there is no additional resistance. For zero winds and
    currents, however, resistance from relative movement in, respectively, water and air
    will be accounted for. Hence, to turn off wind resistance, we need to set either the projected
    frontal area or the wind resistance coefficient to zero.

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

    # speed through water
    speed_through_water_ms = (
        (u_ship_og_ms - u_current_ms) ** 2 + (v_ship_og_ms - v_current_ms) ** 2
    ) ** 0.5
    speed_through_wind_ms = (
        (u_ship_og_ms - u_wind_ms) ** 2 + (v_ship_og_ms - v_wind_ms) ** 2
    ) ** 0.5

    # resistance through calm water (Note that calm only refers to the sea state
    # and does not imply absence of currents.)
    reference_resistance_calm = (
        ship.total_propulsive_efficiency
        * ship.reference_engine_power_W
        / ship.reference_speed_calm_water_ms**3
    )
    power_through_water = reference_resistance_calm * speed_through_water_ms**3

    # resistance through sea waves
    spectral_average = 0.5
    nondimensional_resistance_reference = (
        20.0
        * (ship.waterline_width_m / ship.waterline_length_m) ** (-1.20)
        * (ship.draught_m / ship.waterline_length_m) ** 0.62
    )
    froude_number_reference = (
        ship.reference_speed_calm_water_ms
        / (physics.gravity_acceleration_ms2 * ship.waterline_length_m) ** 0.5
    )
    froude_number = (
        speed_through_water_ms
        / (physics.gravity_acceleration_ms2 * ship.waterline_length_m) ** 0.5
    )
    nondimensional_resistance = (
        nondimensional_resistance_reference / froude_number_reference * froude_number
    )
    resistance_through_sea_waves = (
        nondimensional_resistance
        * physics.sea_water_density_kgm3
        * physics.gravity_acceleration_ms2
        * (w_wave_height / 2.0) ** 2  # wave amplitude = height / 2
        * ship.waterline_width_m**2
        / ship.waterline_length_m
        * spectral_average
    )
    power_through_waves = resistance_through_sea_waves * speed_through_water_ms

    # resistance through wind. (Note simplify this to the case where the apparent
    # wind comes from the front.)
    resistance_through_wind = (
        0.5
        * ship.wind_resistance_coefficient
        * ship.projected_frontal_area_above_waterline_m2
        * physics.air_density_kgm3
        * speed_through_wind_ms**2
    )
    power_through_wind = resistance_through_wind * speed_through_wind_ms

    return power_through_water + power_through_waves + power_through_wind


def hazard_conditions_wave_height(
    ship: Ship = Ship(),
    w_wave_height_m: float = 0,
) -> bool:
    """Check stability thresholds for wave heights.

    Follows
        Mannarini, G., Pinardi, N., Coppini, G., Oddo, P., and Iafrati, A.:
        VISIR-I: small vessels – least-time nautical routes using wave forecasts,
        Geosci. Model Dev., 9, 1597–1625, 2016. https://doi.org/10.5194/gmd-9-1597-2016
    but only accounts for ration of wave height and ship length. Disregarding wave period
    and angle of attack, this amounts to cheching wether wave height / ship length < 1/40.

    Parameters
    ----------
    ship: Ship
        Ship parameters.
    w_wave_height_m: float
       Significant wave height in meters. Defaults to: 0.0

    Returns
    -------
    bool:
        Whether any of the stability thresholds is violated.
    """
    return (w_wave_height_m / ship.waterline_length_m) < 1 / 40.0
