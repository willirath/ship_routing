from .core import Route
from .config import (
    SHIP_DEFAULT,
    PHYSICS_DEFAULT,
    Ship,
    Physics,
)  # TODO: replace globals with explicit config wiring

import xarray as xr
import numpy as np


class InvalidGradientError(RuntimeError):
    """Raised if gradients are invalid, e.g., because modified route on land."""

    pass


class ZeroGradientsError(RuntimeError):
    """Raised if gradients are zero, indicating that route is an extremum."""

    pass


class LargeIncrementError(RuntimeError):
    """Raised if shifts adjusted for learning rate are larger than the test increment."""

    pass


def gradient_descent_time_shift(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    time_shift_seconds: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent by shifting times.

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    time_shift_seconds: float
        Time shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated time stamps but unchanged locations of way points.
    """
    gradients = np.array(
        [
            route.cost_gradient_time_shift(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                time_shift_seconds=time_shift_seconds,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    time_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    # Note that there's no invalid gradients here, because time shifts won't
    # make a formerly valid route touch land.
    if np.any(abs(time_shifts) > time_shift_seconds):
        raise LargeIncrementError
    for n in range(1, len(route) - 1):
        ts = time_shifts[n - 1]
        route = route.replace_waypoint(
            n=n,
            new_way_point=route.way_points[n].move_time(
                time_diff=ts * 1000 * np.timedelta64(1, "ms")
            ),
        )
    return route


def gradient_descent_along_track(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    distance_meters: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent with along-track shifts.

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    distance_meters: float
        Spatial shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated locations but unchanged times of way points.
    """
    gradients = np.array(
        [
            route.cost_gradient_along_track(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                distance_meters=distance_meters,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    dist_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    if np.isnan(gradients_squared_sum):
        raise InvalidGradientError
    if np.any(abs(dist_shifts) > distance_meters):
        raise LargeIncrementError
    # keep a copy of the original route, because we need the original way point azimuths
    _route = route
    for n in range(1, len(route) - 1):
        ds = dist_shifts[n - 1]
        _route = _route.move_waypoint(
            n=n,
            distance_meters=ds,
            azimuth_degrees=route.waypoint_azimuth(n=n),
        )
    return _route


def gradient_descent_across_track_left(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    distance_meters: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent with across-track shifts.

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    distance_meters: float
        Spatial shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated locations but unchanged times of way points.
    """
    gradients = np.array(
        [
            route.cost_gradient_across_track_left(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                distance_meters=distance_meters,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    dist_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    if np.isnan(gradients_squared_sum):
        raise InvalidGradientError
    if np.any(abs(dist_shifts) > distance_meters):
        raise LargeIncrementError
    # keep a copy of the original route, because we need the original way point azimuths
    _route = route
    for n in range(1, len(route) - 1):
        _route = _route.move_waypoint(
            n=n,
            distance_meters=dist_shifts[n - 1],
            azimuth_degrees=route.waypoint_azimuth(n=n) - 90.0,
        )
    return _route


def crossover_routes_random(
    route_0: Route = None,
    route_1: Route = None,
) -> Route:
    """Randomly cross over routes.

    Parameters
    ----------
    route_0: Route
        First route.
    route_1: Route
        Second Route.

    Returns
    -------
    Route:
        Route of randomly selected segments.
    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    segments_mix = [
        s0s1[np.random.randint(0, 2)] for s0s1 in zip(segments_0, segments_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    ref_timestep_seconds = route_0.data_frame.time.diff().mean() / np.timedelta64(
        1, "s"
    )
    route_mix = route_mix.remove_consecutive_duplicate_timesteps(
        min_time_diff_seconds=ref_timestep_seconds / 5
    )
    return route_mix


def crossover_routes_minimal_cost(
    route_0: Route = None,
    route_1: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    ship: Ship = SHIP_DEFAULT,
    physics: Physics = PHYSICS_DEFAULT,
) -> Route:
    """Cross over routes to minimy cost.

    Parameters
    ----------
    route_0: Route
        First route.
    route_1: Route
        Second Route.
    current_data_set: xr.Dataset
        Current data set.
    wind_data_set: xr.Dataset
        Wind data set.
    wave_data_set: xr.Dataset
        Wave data set.
    ship: Ship
        Ship parameters. Defaults to: SHIP_DEFAULT
    physics: Physics
        Physics parameters. Defaults to: PHYSICS_DEFAULT

    Returns
    -------
    Route:
        Route with segments selected such that the cost is minimised.

    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    cost_0 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_0
    ]
    cost_1 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_1
    ]
    segments_mix = [
        s0s1c0c1[int(s0s1c0c1[-1] < s0s1c0c1[-2])]
        for s0s1c0c1 in zip(segments_0, segments_1, cost_0, cost_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    route_mix = route_mix.remove_consecutive_duplicate_timesteps()
    return route_mix
