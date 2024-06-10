from .core import Route

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
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
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
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
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
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
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
