from .core import Route, WayPoint

import numpy as np
import xarray as xr
import tqdm


def create_route(
    lon_waypoints: list = None,
    lat_waypoints: list = None,
    time_start: str | np.datetime64 = None,
    time_end: str | np.datetime64 = None,
    time_resolution_hours: float = 6.0,
) -> Route:
    dt = (time_end - time_start) / (len(lon_waypoints) - 1)
    time_waypoints = [time_start + n * dt for n in range(len(lon_waypoints))]
    route_gc = Route(
        way_points=tuple(
            WayPoint(lon=lon, lat=lat, time=time)
            for lon, lat, time in zip(
                lon_waypoints,
                lat_waypoints,
                time_waypoints,
            )
        )
    )
    refine_to_dist = (
        np.mean([l.speed_ms for l in route_gc.legs]) * time_resolution_hours * 3600.0
    )
    return route_gc.refine(distance_meters=refine_to_dist)


def stochastic_search(
    route: Route = None,
    number_of_iterations: int = 10,
    acceptance_rate_target: float = 0.05,
    acceptance_rate_for_increase_cost: float = 0.0,
    refinement_factor: float = 0.5,
    mod_width: float = None,
    max_move_meters: float = None,
    include_logging: bool = True,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> Route:
    if include_logging:
        cost_steps = []
    else:
        cost_steps = None

    cost = route.cost_through(
        current_data_set=current_data_set,
        wave_data_set=wave_data_set,
        wind_data_set=wind_data_set,
    )
    if include_logging:
        cost_steps.append(cost)

    accepted = 0
    n_reset = 0
    for n in tqdm.tqdm(range(number_of_iterations)):
        route_ = route.move_waypoints_left_nonlocal(
            center_distance_meters=np.random.uniform(
                mod_width / 2.0, route.length_meters - mod_width / 2.0
            ),
            width_meters=mod_width,
            max_move_meters=max_move_meters * np.random.uniform(-1, 1),
        )
        cost_ = route_.cost_through(
            current_data_set=current_data_set,
            wave_data_set=wave_data_set,
            wind_data_set=wind_data_set,
        )
        if not np.isnan(cost_) and (
            (cost_ < cost)
            or (np.random.uniform(0, 1) < acceptance_rate_for_increase_cost)
        ):
            route = route_
            cost = cost_
            accepted += 1
            if include_logging:
                cost_steps.append(cost)
        if (accepted + 1) / (n_reset + 1) < acceptance_rate_target:
            n_reset = 0
            accepted = 0
            mod_width *= refinement_factor
            max_move_meters *= refinement_factor

        n_reset += 1

    return route, cost_steps
