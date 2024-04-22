from .geodesics import get_directions
from .currents import select_currents_along_traj


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


def power_for_traj_in_ocean(ship_positions=None, speed=None, ocean_data=None):
    lon = ship_positions.lon
    lat = ship_positions.lat
    uhat, vhat = get_directions(lon=lon, lat=lat)
    us = uhat * speed
    vs = vhat * speed
    ds_uovo = select_currents_along_traj(ds=ocean_data, ship_positions=ship_positions)
    return power_maintain_speed(us=us, vs=vs, uo=ds_uovo.uo, vo=ds_uovo.vo)
