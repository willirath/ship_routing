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


def power_for_traj_in_ocean(trajectory=None, ocean_data=None):
    pass
