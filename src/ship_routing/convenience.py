from .core import Route, WayPoint, Leg

from .algorithms import (
    gradient_descent_across_track_left,
    gradient_descent_along_track,
    gradient_descent_time_shift,
    InvalidGradientError,
    LargeIncrementError,
    ZeroGradientsError,
)

import numpy as np
import pandas as pd
import xarray as xr
import tqdm


from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Logs:
    iteration: int = 0
    cost: float = 0.0
    method: str = "initial"

    @property
    def data_frame(self):
        return pd.DataFrame(
            asdict(self),
            index=[
                0,
            ],
        )


@dataclass(frozen=True)
class LogsRoute:
    logs: Logs
    route: Route

    @property
    def data_frame(self):
        return self.logs.data_frame.join(self.route.data_frame, how="cross")


def create_route(
    lon_waypoints: list = None,
    lat_waypoints: list = None,
    time_start: str | np.datetime64 = None,
    time_end: str | np.datetime64 = None,
    time_resolution_hours: float = 6.0,
) -> Route:
    # construct intermediate time points from leg lengths
    leg_lengths = [
        Leg(
            WayPoint(lon=lon0, lat=lat0, time=0), WayPoint(lon=lon1, lat=lat1, time=0)
        ).length_meters
        for lon0, lat0, lon1, lat1 in zip(
            lon_waypoints[:-1], lat_waypoints[:-1], lon_waypoints[1:], lat_waypoints[1:]
        )
    ]
    route_length = sum(leg_lengths)
    relative_time_offsets = [0] + [ll / route_length for ll in leg_lengths]
    time_waypoints = time_start + np.timedelta64(1, "s") * (
        (time_end - time_start) / np.timedelta64(1, "s")
    ) * np.cumsum(relative_time_offsets)
    # create GC route
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
    # resample to desired resolution
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
    include_logs_routes: bool = True,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> Route:
    if include_logs_routes:
        logs_routes = []
    else:
        logs_routes = None

    cost = route.cost_through(
        current_data_set=current_data_set,
        wave_data_set=wave_data_set,
        wind_data_set=wind_data_set,
    )
    if include_logs_routes:
        logs_routes.append(
            LogsRoute(
                logs=Logs(iteration=-1, cost=cost, method="initial"),
                route=route,
            )
        )

    accepted = 0
    n_reset = 0
    for iteration in tqdm.tqdm(range(number_of_iterations)):
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
            if include_logs_routes:
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration, cost=cost, method="stochastic_search"
                        ),
                        route=route,
                    )
                )
        if (accepted + 1) / (n_reset + 1) < acceptance_rate_target:
            n_reset = 0
            accepted = 0
            mod_width *= refinement_factor
            max_move_meters *= refinement_factor

        n_reset += 1

    return route, logs_routes


def gradient_descent(
    route: Route = None,
    num_iterations: int = 1,
    learning_rate_percent_time: float = 0.5,
    time_increment: float = 1_200,
    learning_rate_percent_along: float = 0.5,
    dist_shift_along: float = 10_000,
    learning_rate_percent_across: float = 0.5,
    dist_shift_across: float = 10_000,
    include_logs_routes: bool = True,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> tuple[Route, Logs]:
    if include_logs_routes:
        logs_routes = []
    else:
        logs_routes = None

    for iteration in tqdm.tqdm(range(num_iterations)):
        try:
            route = gradient_descent_time_shift(
                route=route,
                current_data_set=current_data_set,
                wave_data_set=wave_data_set,
                wind_data_set=wind_data_set,
                time_shift_seconds=time_increment,
                learning_rate_percent=learning_rate_percent_time,
            )
            if include_logs_routes:
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            method="gradient_descent_time_shift",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
            pass
        except InvalidGradientError:
            time_increment /= 2.0
            learning_rate_percent_time /= 2.0
        except LargeIncrementError:
            learning_rate_percent_time /= 2.0

        try:
            route = gradient_descent_across_track_left(
                route=route,
                current_data_set=current_data_set,
                wave_data_set=wave_data_set,
                wind_data_set=wind_data_set,
                distance_meters=dist_shift_across,
                learning_rate_percent=learning_rate_percent_across,
            )
            if include_logs_routes:
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            method="gradient_descent_across_track_left",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
            pass
        except InvalidGradientError:
            dist_shift_across /= 2.0
            learning_rate_percent_across /= 2.0
        except LargeIncrementError:
            learning_rate_percent_across /= 2.0

        try:
            route = gradient_descent_along_track(
                route=route,
                current_data_set=current_data_set,
                wave_data_set=wave_data_set,
                wind_data_set=wind_data_set,
                distance_meters=dist_shift_along,
                learning_rate_percent=learning_rate_percent_along,
            )
            if include_logs_routes:
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            method="gradient_descent_across_track_left",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
            pass
        except InvalidGradientError:
            dist_shift_along /= 2.0
            learning_rate_percent_along /= 2.0
        except LargeIncrementError:
            learning_rate_percent_along /= 2.0

    return route, logs_routes
