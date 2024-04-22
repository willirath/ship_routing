from .geodesics import get_directions
from .currents import select_currents_for_leg


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


def power_for_leg_in_ocean(leg_pos=None, leg_speed=None, ocean_data=None):
    lon = (leg_pos[0][0], leg_pos[1][0])
    lat = (leg_pos[0][1], leg_pos[1][1])
    uhat, vhat = get_directions(lon=lon, lat=lat)
    us, vs = sum(uhat) / 2.0 * leg_speed, sum(vhat) / 2.0 * leg_speed
    ds_uovo = select_currents_for_leg(
        ds=ocean_data,
        lon_start=leg_pos[0][0],
        lon_end=leg_pos[1][0],
        lat_start=leg_pos[0][1],
        lat_end=leg_pos[1][1],
    )
    return (
        power_maintain_speed(us=us, vs=vs, uo=ds_uovo.uo, vo=ds_uovo.vo).mean().data[()]
    )
