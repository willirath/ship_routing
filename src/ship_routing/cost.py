from .currents import select_data_for_leg

import numpy as np


def power_maintain_speed(
    uo: float = None,
    vo: float = None,
    us: float = None,
    vs: float = None,
    coeff: float = 1.0,
) -> float:
    u_through_water = us - uo
    v_through_water = vs - vo
    speed_through_water = (u_through_water**2 + v_through_water**2) ** 0.5
    power_needed = coeff * (speed_through_water**3)
    return power_needed


def power_maintain_speed_realistic(
    u_ship_og: float = None,
    v_ship_og: float = None,
    u_current: float = 0.0,
    v_current: float = 0.0,
    u_wind: float = 0.0,
    v_wind: float = 0.0,
    w_wave_height: float = 0.0,
    vessel_waterline_width: float = 30.0,
    vessel_waterline_length: float = 210.0,
    vessel_total_propulsive_efficiency: float = 0.7,
    vessel_maximum_engine_power: float = 14296344.0,
    vessel_speed_calm_water: float = 9.259,
    vessel_draught: float = 11.5,
    vessel_supersurface_area: float = 345.0,
    physics_air_mass_density: float = 1.225,
    vessel_wind_resistance_coefficient: float = 0.4,
    vessel_reference_froede_number: float = 17.6,
    physics_spectral_average: float = 0.5,
    physics_surface_water_density: float = 1029.0,
    physics_acceleration_gravity: float = 9.80665,
    **kwargs,
) -> float:
    """Calculate quadratic drag law power needed to maintain speed over ground.
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
    vessel_supersurface_area: float
        area of the above water vessel structure exposed to wind [m ** 2]. Defaults to 345
    vessel_subsurface_area: float
        an average area of the lateral projection of underwater vessel structure [m ** 2]. Defaults to 245
    vessel_waterline_width: float
        width of vessel at the waterline in [m]. Defaults to 30
    vessel_waterline_length: float
        length of vessel at the waterline in [m]. Defaults to 210
    vessel_total_propulsive_efficiency: float
        total propulsive engine efficiency. Defaults to 0.7
    vessel_maximum_engine_power: float
        vessel maximu engine power in [W]. Defaults to 14296344.0,
    vessel_speed_calm_water: float
        vessel speed maximum in calm water [m/s]. Defaults 9.259
    vessel_draught: float
        vessel draught in [m]. Defaults to 11.5
    physics_air_mass_density: float
       mass density of air [kg/m**3]. Defaults to 1.225
    vessel_wind_resistance_coefficient: float
       wind resistance coefficent, typically in rages [0.4-1]. Defaults to 0.4
    physics_spectral_average: float
       spectral and angular dependency factor, dimensionless. Defaults to 0.5
    physics_surface_water_density: float
       density of surface water [kg/m**3]. Defaults to 1029
    physics_acceleration_gravity: float
       the Earth gravity accleration [m/s**2]. Defaults to 9.80665
    All other keyword arguments will be ignored.
    Returns
    -------
    array:
        Power in W (=kg*m2/s3) needed to maintain speed over ground for each
        element of u_ship_og and v_ship_og. Shape will be identical to
        u_ship_og and v_ship_og.
    """

    # ensure shapes of u_ship_og and v_ship_og agree
    if np.array(u_ship_og).shape != np.array(v_ship_og).shape:
        raise ValueError("Shape of u_ship_og and v_ship_og need to agree.")

    # calc velocities through water
    u_ship_tw = u_ship_og - u_current
    v_ship_tw = v_ship_og - v_current

    # calc speeds (over ground, through water, relative to wind)
    # speed_og = (u_ship_og ** 2 + v_ship_og ** 2) ** 0.5
    speed_tw = (u_ship_tw**2 + v_ship_tw**2) ** 0.5
    speed_rel_to_wind = ((u_ship_og - u_wind) ** 2 + (v_ship_og - v_wind) ** 2) ** 0.5

    # calc speeds relative to wind
    u_speed_rel_to_wind = u_ship_og - u_wind
    v_speed_rel_to_wind = v_ship_og - v_wind

    # drag coefficients
    coeff_water_drag = vessel_maximum_engine_power / vessel_speed_calm_water**3

    coeff_wind_drag = (
        0.5
        * physics_air_mass_density
        * vessel_wind_resistance_coefficient
        * vessel_supersurface_area
        / vessel_total_propulsive_efficiency
    )

    coeff_wave_drag = (
        20.0
        * (vessel_waterline_width / vessel_waterline_length) ** (-1.2)
        * (1 / vessel_waterline_length) ** 0.62
        / vessel_total_propulsive_efficiency
        / vessel_reference_froede_number
        * physics_spectral_average
        * physics_surface_water_density
        * w_wave_height**2
        / 4
        * vessel_waterline_width**2
        * (physics_acceleration_gravity / vessel_waterline_length**3) ** 0.5
        * vessel_draught**0.62
        * 0.25
    )

    # Reference frame is relative to water
    wave_resistance_x = coeff_wave_drag * u_ship_tw
    water_resistance_x = coeff_water_drag * speed_tw * u_ship_tw
    wind_resistance_x = coeff_wind_drag * speed_rel_to_wind * u_speed_rel_to_wind

    wave_resistance_y = coeff_wave_drag * v_ship_tw
    water_resistance_y = coeff_water_drag * speed_tw * v_ship_tw
    wind_resistance_y = coeff_wind_drag * speed_rel_to_wind * v_speed_rel_to_wind

    power_needed = (
        wave_resistance_x + water_resistance_x + wind_resistance_x
    ) * u_ship_tw + (
        wave_resistance_y + water_resistance_y + wind_resistance_y
    ) * v_ship_tw

    return power_needed
