from dataclasses import dataclass

import numpy as np
import pandas as pd
import pint
import xarray as xr
from shapely.geometry import LineString, Point
from shapely import union_all as shp_union_all
from shapely.ops import snap
from scipy.interpolate import interp1d
from scipy.signal.windows import hann

from typing import Iterable, Tuple, Any

from functools import lru_cache
from .config import MAX_CACHE_SIZE

# Fallback for @profile decorator when not using line_profiler
try:
    profile
except NameError:

    def profile(func):
        return func


from .geodesics import (
    get_length_meters,
    get_distance_meters,
    move_fwd,
    refine_along_great_circle,
    get_leg_azimuth,
    knots_to_ms,
)

from .remix import (
    segment_lines_with_each_other,
    SHAPELY_RESOLUTION,
)

from .data import select_data_for_leg

from .cost import (
    power_maintain_speed,
    hazard_conditions_wave_height,
)

from .config import Ship, Physics, SHIP_DEFAULT, PHYSICS_DEFAULT


@dataclass(frozen=True)
class WayPoint:
    """Way point."""

    lon: float
    lat: float
    time: np.datetime64

    @property
    def data_frame(self):
        """Single-row data frame with cols lon, lat, time."""
        return pd.DataFrame(
            {"lon": self.lon, "lat": self.lat, "time": self.time},
            index=[
                0,
            ],
        )

    @classmethod
    def from_data_frame(cls, data_frame: pd.DataFrame = None):
        """Construct way point from data frame.

        This will use the first row only.
        """
        return cls(
            lon=data_frame.iloc[0]["lon"],
            lat=data_frame.iloc[0]["lat"],
            time=data_frame.iloc[0]["time"],
        )

    @property
    def point(self):
        """Point geometry with x=lon and y=lat."""
        return Point(self.lon, self.lat)

    @classmethod
    def from_point(cls, point: Point = None, time: np.datetime64 = None):
        """Construct from Point with x=lon and y=lat and from time."""
        return cls(lon=point.x, lat=point.y, time=time)

    def move_space(self, azimuth_degrees: float = None, distance_meters: float = None):
        """Move in space.

        Parameters
        ----------
        azimuth_degrees: float
            Azimuth in degrees.
        distance_meters: float
            Distance in meters.

        Returns
        -------
        WayPoint

        """
        lon_new, lat_new = move_fwd(
            lon=self.lon,
            lat=self.lat,
            azimuth_degrees=azimuth_degrees,
            distance_meters=distance_meters,
        )
        return WayPoint(lon=lon_new, lat=lat_new, time=self.time)

    def move_time(self, time_diff: np.timedelta64):
        """Move in time by time_diff."""
        return WayPoint(lon=self.lon, lat=self.lat, time=self.time + time_diff)


@dataclass(frozen=True)
class Leg:
    """A leg connecting two waypoints."""

    way_point_start: WayPoint
    way_point_end: WayPoint

    @property
    def data_frame(self):
        """Two-row data frame with cols lon, lat, time."""
        return pd.concat(
            (self.way_point_start.data_frame, self.way_point_end.data_frame),
            ignore_index=True,
        )

    @classmethod
    def from_data_frame(cls, data_frame: pd.DataFrame = None):
        """Construct leg from data frame.

        This will use the first two rows only.
        """
        return cls(
            way_point_start=WayPoint(
                lon=data_frame.iloc[0]["lon"],
                lat=data_frame.iloc[0]["lat"],
                time=data_frame.iloc[0]["time"],
            ),
            way_point_end=WayPoint(
                lon=data_frame.iloc[1]["lon"],
                lat=data_frame.iloc[1]["lat"],
                time=data_frame.iloc[1]["time"],
            ),
        )

    @property
    def line_string(self):
        """LineString geometry with x=lon and y=lat."""
        return LineString((self.way_point_start.point, self.way_point_end.point))

    @classmethod
    def from_line_string(cls, line_string: LineString = None, time: Iterable = None):
        """Construct from LineString with x=lon and y=lat and from time vector."""
        point_start, point_end = map(Point, line_string.coords)
        time_start, time_end = time
        return cls(
            way_point_start=WayPoint.from_point(point=point_start, time=time_start),
            way_point_end=WayPoint.from_point(point=point_end, time=time_end),
        )

    @property
    def length_meters(self):
        """Length of the leg in meters."""
        return get_length_meters(self.line_string)

    @property
    def duration_seconds(self):
        """Duration in seconds excluding sign.

        Always positive irrespective of order or way points.
        """
        return (
            abs(
                (self.way_point_end.time - self.way_point_start.time)
                / np.timedelta64(1, "ms")
            )
            / 1000.0
        )

    @property
    def speed_ms(self):
        """Speed in meters per second."""
        return self.length_meters / self.duration_seconds

    def time_at_distance(self, distance_meters: float = None):
        """Time after distance travelled along this leg."""
        return (
            self.way_point_start.time
            + np.timedelta64(1, "ms") * 1000.0 * distance_meters / self.speed_ms
        )

    def refine(self, distance_meters: float = None):
        """Refine in distance.

        If the new distance is shorted than the length, there will be at least two new legs.

        """
        lons = (self.way_point_start.lon, self.way_point_end.lon)
        lats = (self.way_point_start.lat, self.way_point_end.lat)
        lon_refined, lat_refined = refine_along_great_circle(
            lon=lons, lat=lats, new_dist=distance_meters
        )
        dist_refined = (
            [
                0,
            ]
            + [
                get_distance_meters(
                    lon_start=self.way_point_start.lon,
                    lat_start=self.way_point_start.lat,
                    lon_end=_lon,
                    lat_end=_lat,
                )
                for _lon, _lat in zip(lon_refined[1:-1], lat_refined[1:-1])
            ]
            + [
                self.length_meters,
            ]
        )
        time_refined = (
            [
                self.way_point_start.time,
            ]
            + [
                self.time_at_distance(distance_meters=_dist)
                for _dist in dist_refined[1:-1]
            ]
            + [
                self.way_point_end.time,
            ]
        )
        return tuple(
            (
                Leg(
                    way_point_start=WayPoint(
                        lon=lon_start, lat=lat_start, time=time_start
                    ),
                    way_point_end=WayPoint(lon=lon_end, lat=lat_end, time=time_end),
                )
                for lon_start, lat_start, time_start, lon_end, lat_end, time_end in zip(
                    lon_refined[:-1],
                    lat_refined[:-1],
                    time_refined[:-1],
                    lon_refined[1:],
                    lat_refined[1:],
                    time_refined[1:],
                )
            )
        )

    def overlaps_time(self, time: np.datetime64 = None):
        """Whether time is withing leg."""
        times = (self.way_point_start.time, self.way_point_end.time)
        return not ((time < min(times)) or (time > max(times)))

    @property
    def fw_azimuth_degrees(self):
        """Forward azimuth from the start waypoint in degrees."""
        _, fw_az_deg, _ = get_leg_azimuth(
            lon_start=self.way_point_start.lon,
            lat_start=self.way_point_start.lat,
            lon_end=self.way_point_end.lon,
            lat_end=self.way_point_end.lat,
        )
        return fw_az_deg

    @property
    def bw_azimuth_degrees(self):
        """Backward azimuth from the start waypoint in degrees."""
        _, _, bw_az_deg = get_leg_azimuth(
            lon_start=self.way_point_start.lon,
            lat_start=self.way_point_start.lat,
            lon_end=self.way_point_end.lon,
            lat_end=self.way_point_end.lat,
        )
        return bw_az_deg

    @property
    def azimuth_degrees(self):
        """Azimuth of the leg in degrees.

        This averages the forward azimuth of the first and backward azimuth of the last point.
        """
        az_deg, _, _ = get_leg_azimuth(
            lon_start=self.way_point_start.lon,
            lat_start=self.way_point_start.lat,
            lon_end=self.way_point_end.lon,
            lat_end=self.way_point_end.lat,
        )
        return az_deg

    @property
    def uv_over_ground_ms(self):
        """Speed over ground in meters per second."""
        spd = self.speed_ms
        az_rad = np.deg2rad(self.azimuth_degrees)
        return spd * np.sin(az_rad), spd * np.cos(az_rad)

    @profile
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def cost_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        u_ship_og, v_ship_og = self.uv_over_ground_ms
        if current_data_set is not None:
            ds_current = select_data_for_leg(
                ds=current_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            u_current, v_current = ds_current.uo, ds_current.vo
        else:
            u_current, v_current = 0, 0
        if wind_data_set is not None:
            ds_wind = select_data_for_leg(
                ds=wind_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            u_wind, v_wind = ds_wind.uw, ds_wind.vw
        else:
            u_wind, v_wind = 0, 0
        if wave_data_set is not None:
            ds_wave = select_data_for_leg(
                ds=wave_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            w_wave_height = ds_wave.wh
        else:
            w_wave_height = 0
        pwr = power_maintain_speed(
            u_current_ms=u_current,
            v_current_ms=v_current,
            u_wind_ms=u_wind,
            v_wind_ms=v_wind,
            w_wave_height=w_wave_height,
            u_ship_og_ms=u_ship_og,
            v_ship_og_ms=v_ship_og,
            ship=ship,
            physics=physics,
        )
        if pwr.isnull().sum() > 0:
            return np.nan
        else:
            return pwr.mean().data[()] * self.duration_seconds

    @profile
    def hazard_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        u_ship_og, v_ship_og = self.uv_over_ground_ms
        if current_data_set is not None:
            ds_current = select_data_for_leg(
                ds=current_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            u_current, v_current = ds_current.uo, ds_current.vo
        else:
            u_current, v_current = 0, 0
        if wind_data_set is not None:
            ds_wind = select_data_for_leg(
                ds=wind_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            u_wind, v_wind = ds_wind.uw, ds_wind.vw
        else:
            u_wind, v_wind = 0, 0
        if wave_data_set is not None:
            ds_wave = select_data_for_leg(
                ds=wave_data_set,
                lon_start=self.way_point_start.lon,
                lat_start=self.way_point_start.lat,
                time_start=self.way_point_start.time,
                lon_end=self.way_point_end.lon,
                lat_end=self.way_point_end.lat,
                time_end=self.way_point_end.time,
            )
            w_wave_height = ds_wave.wh
        else:
            w_wave_height = 0
        return any(
            hazard_conditions_wave_height(
                u_current_ms=u_current,
                v_current_ms=v_current,
                u_wind_ms=u_wind,
                v_wind_ms=v_wind,
                w_wave_height=w_wave_height,
                u_ship_og_ms=u_ship_og,
                v_ship_og_ms=v_ship_og,
                ship=ship,
                physics=physics,
            )
        )

    def speed_through_water_ms(self, current_data_set: xr.Dataset = None):
        # normalized velocity over ground
        az_rad = np.deg2rad(self.azimuth_degrees)
        uhat, vhat = np.sin(az_rad), np.cos(az_rad)
        # parallel component of currents
        ds_uovo = select_data_for_leg(
            ds=current_data_set,
            lon_start=self.way_point_start.lon,
            lat_start=self.way_point_start.lat,
            time_start=self.way_point_start.time,
            lon_end=self.way_point_end.lon,
            lat_end=self.way_point_end.lat,
            time_end=self.way_point_end.time,
        )
        uo_along = (ds_uovo.uo * uhat + ds_uovo.vo * vhat).mean().data[()]
        return self.speed_ms - uo_along

    def split_at_distance(self, distance_meters: float = None):
        """Split leg at given distance (relto start waypoint)."""
        fw_az = self.fw_azimuth_degrees
        new_time = self.time_at_distance(distance_meters=distance_meters)
        split_wp = self.way_point_start.move_space(
            azimuth_degrees=fw_az, distance_meters=distance_meters
        ).move_time(time_diff=new_time - self.way_point_start.time)
        return (
            Leg(way_point_start=self.way_point_start, way_point_end=split_wp),
            Leg(way_point_start=split_wp, way_point_end=self.way_point_end),
        )


@dataclass(frozen=True)
class Route:
    """A route containing of multiple waypoints."""

    way_points: Tuple

    def __post_init__(self):
        if not isinstance(self.way_points, tuple):
            raise ValueError("Way_points need to be a tuple.")
        if len(self.way_points) < 2:
            raise ValueError(
                "A Route needs at least two way points which may be identical."
            )

    def __len__(self):
        """Length is determined by numer of way points."""
        return len(self.way_points)

    def __getitem__(self, key):
        try:
            return Route(way_points=self.way_points[key])
        except ValueError as valerr:
            raise ValueError("Slicing needs at least two way points.")

    def __add__(self, other):
        """Concatenate route with other route.

        If last own / first other way points are identical, drop first wp of other.
        """
        if other.way_points[0] == self.way_points[-1]:
            other_wps = other.way_points[1:]
        else:
            other_wps = other.way_points
        return Route(way_points=self.way_points + other_wps)

    @property
    def legs(self):
        """Tuple of legs pairing all consecutive way points."""
        return tuple(
            (
                Leg(way_point_start=w0, way_point_end=w1)
                for w0, w1 in zip(self.way_points[:-1], self.way_points[1:])
            )
        )

    @classmethod
    def from_legs(cls, legs: Iterable = None):
        """Construct route from sequence of legs.

        Note that no ordering is done before constructing the route.
        """
        legs = list(legs)
        return cls(
            way_points=tuple((l.way_point_start for l in legs))
            + tuple((legs[-1].way_point_end,))
        )

    @property
    def data_frame(self):
        """Data frame with columns lon, lat, time."""
        return pd.concat((wp.data_frame for wp in self.way_points), ignore_index=True)

    def to_dict(self) -> dict[str, Any]:
        """Simple dict representation of the route."""
        return {"way_points": self.data_frame.to_dict(orient="records")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Construct route from dict."""
        frame = pd.DataFrame(data["way_points"])
        return cls.from_data_frame(frame)

    @classmethod
    def from_data_frame(cls, data_frame: pd.DataFrame = None):
        """Construct route from data frame."""
        return cls(
            way_points=tuple(
                (
                    WayPoint.from_data_frame(data_frame=data_frame.iloc[n : n + 1])
                    for n in range(len(data_frame))
                )
            )
        )

    @property
    def line_string(self):
        """LineString geometry with x=lon and y=lat."""
        return LineString((w.point for w in self.way_points))

    @classmethod
    def from_line_string(cls, line_string: LineString = None, time: Iterable = None):
        """Construct from LineString with x=lon and y=lat and from time vector."""
        return cls(
            way_points=tuple(
                WayPoint.from_point(point=_p, time=_t)
                for _p, _t in zip(map(Point, line_string.coords), time)
            )
        )

    @classmethod
    def create_route(
        cls,
        lon_waypoints: list = None,
        lat_waypoints: list = None,
        time_start: str | np.datetime64 = None,
        time_end: str | np.datetime64 = None,
        speed_knots: float = None,
        time_resolution_hours: float = 6.0,
    ):
        """Create a route from waypoints, times, and speed.

        Parameters
        ----------
        lon_waypoints : list
            Longitudes of waypoints
        lat_waypoints : list
            Latitudes of waypoints
        time_start : str | np.datetime64
            Start time
        time_end : str | np.datetime64, optional
            End time
        speed_knots : float, optional
            Speed in knots
        time_resolution_hours : float
            Time resolution in hours (default 6.0)

        Returns
        -------
        Route
            Created and refined route

        Notes
        -----
        Only two of time_start, time_end, speed_knots can be specified.
        """
        # construct prelim leg and route lengths (no time known yet)
        leg_lengths = [
            Leg(
                WayPoint(lon=lon0, lat=lat0, time=0),
                WayPoint(lon=lon1, lat=lat1, time=0),
            ).length_meters
            for lon0, lat0, lon1, lat1 in zip(
                lon_waypoints[:-1],
                lat_waypoints[:-1],
                lon_waypoints[1:],
                lat_waypoints[1:],
            )
        ]
        route_length = sum(leg_lengths)

        # find duration
        if time_start is not None and time_end is not None:
            if speed_knots is not None:
                raise ValueError(
                    "Only two of time_start, time_end, speed_knots can be given."
                )
        if isinstance(time_start, str):
            time_start = np.datetime64(time_start)
        if isinstance(time_end, str):
            time_end = np.datetime64(time_end)
        if time_start is not None and speed_knots is not None:
            speed_ms = knots_to_ms(speed_knots)
            time_end = time_start + np.timedelta64(1, "s") * route_length / speed_ms
        if time_end is not None and speed_knots is not None:
            speed_ms = knots_to_ms(speed_knots)
            time_start = time_end - np.timedelta64(1, "s") * route_length / speed_ms

        # construct times of waypoints
        relative_time_offsets = [0] + [ll / route_length for ll in leg_lengths]
        time_waypoints = time_start + np.timedelta64(1, "s") * (
            (time_end - time_start) / np.timedelta64(1, "s")
        ) * np.cumsum(relative_time_offsets)

        # create GC segment route
        route_gc = cls(
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
            np.mean([l.speed_ms for l in route_gc.legs])
            * time_resolution_hours
            * 3600.0
        )
        return route_gc.refine(distance_meters=refine_to_dist)

    @property
    def length_meters(self):
        """Length of the route in meters."""
        return get_length_meters(self.line_string)

    @property
    def strictly_monotonic_time(self):
        """True if strictly monotonic in all time steps."""
        return all(
            (
                w1.time > w0.time
                for w0, w1 in zip(self.way_points[:-1], self.way_points[1:])
            )
        )

    def sort_in_time(self):
        """Return route with waypoints sorted in time in ascending order."""
        return Route(way_points=tuple(sorted(self.way_points, key=lambda w: w.time)))

    def remove_consecutive_duplicate_timesteps(self, min_time_diff_seconds=600):
        """Route with the first of each 2 consecutive way points having the same time stamp."""

        def generate_non_dupe_wps(wps):
            current_wp = wps[0]
            yield current_wp
            n = 0
            while n < len(wps) - 1:
                n += 1
                candidate_wp = wps[n]
                if (
                    candidate_wp.time - current_wp.time
                ) >= min_time_diff_seconds * np.timedelta64(1, "s"):
                    current_wp = candidate_wp
                    yield current_wp

        return Route(way_points=tuple(generate_non_dupe_wps(self.way_points)))

    @property
    def distance_meters(self):
        """Along-track distance in meters."""
        return (0,) + tuple(np.cumsum([l.length_meters for l in self.legs]))

    def refine(self, distance_meters: float = None):
        """Refine with new distance.

        Refinement is done per leg.
        """
        refined_legs = tuple(
            sum(
                [l.refine(distance_meters=distance_meters) for l in self.legs], start=()
            )
        )
        return Route.from_legs(legs=refined_legs)

    def replace_waypoint(
        self,
        n: int = None,
        new_way_point: WayPoint = None,
    ):
        """Replace nth way point with."""
        return Route(
            way_points=(
                self.way_points[:n] + (new_way_point,) + self.way_points[n + 1 :]
            )
        )

    def move_waypoint(
        self,
        n: int = None,
        azimuth_degrees: float = None,  # abs azimuth !!!!
        distance_meters: float = None,
    ):
        """Move nth waypoint."""
        wp_moved = self.way_points[n].move_space(
            azimuth_degrees=azimuth_degrees,
            distance_meters=distance_meters,
        )
        return self.replace_waypoint(n=n, new_way_point=wp_moved)

    @profile
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def cost_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        """Cost along whole route."""
        return sum(
            self.cost_per_leg_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
        )

    @profile
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def cost_per_leg_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        """Cost along each leg."""
        return tuple(
            (
                l.cost_through(
                    current_data_set=current_data_set,
                    wind_data_set=wind_data_set,
                    wave_data_set=wave_data_set,
                    ship=ship,
                    physics=physics,
                )
                for l in self.legs
            )
        )

    def hazard_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        return any(
            self.hazard_per_leg_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
        )

    def hazard_per_leg_through(
        self,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        return tuple(
            l.hazard_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
            for l in self.legs
        )

    def waypoint_azimuth(self, n: int = None):
        """Azimuth of waypoint n.

        For n=0 only the fwd az of the first leg and for n=-1 only the
        bw az of the last leg is returned. For all other n, the average of the bw az
        of the before leg and the fw az of the after leg are returned.
        """
        _legs = self.legs
        if n == 0:
            return _legs[0].fw_azimuth_degrees
        elif n == len(self) - 1:
            return _legs[-1].bw_azimuth_degrees
        else:
            return (_legs[n - 1].bw_azimuth_degrees + _legs[n].fw_azimuth_degrees) / 2.0

    def split_at_distance(self, distance_meters: float = None):
        """Split at given distance."""
        num_leg = (
            min(
                sum([distance_meters >= d for d in self.distance_meters]), len(self) - 1
            )
            - 1
        )

        old_legs = self.legs
        split_legs = old_legs[num_leg].split_at_distance(
            distance_meters=distance_meters - self.distance_meters[num_leg]
        )
        new_legs_before = old_legs[:num_leg] + split_legs[:1]
        new_legs_after = split_legs[1:] + old_legs[num_leg + 1 :]
        return (
            Route.from_legs(legs=new_legs_before),
            Route.from_legs(legs=new_legs_after),
        )

    def waypoint_at_distance(self, distance_meters: float = None):
        """Generate waypoint at distance."""
        num_leg = (
            min(
                sum([distance_meters >= d for d in self.distance_meters]), len(self) - 1
            )
            - 1
        )

        split_legs = self.legs[num_leg].split_at_distance(
            distance_meters=distance_meters - self.distance_meters[num_leg]
        )
        return split_legs[0].way_point_end

    def segment_at(
        self,
        other,
        resolution: float = SHAPELY_RESOLUTION,
    ):
        """Segment route at other route."""
        self_line_string = self.line_string
        other_line_string = other.line_string
        self_seg, other_seg = segment_lines_with_each_other(
            line_0=self_line_string,
            line_1=other_line_string,
            resolution=resolution,
        )
        self_split_dists = tuple(
            np.cumsum([get_length_meters(s) for s in self_seg[:-1]])
        )
        other_split_dists = tuple(
            np.cumsum([get_length_meters(s) for s in other_seg[:-1]])
        )
        self_segments_rev = []
        self_r = self
        for d in self_split_dists[::-1]:
            s0, s1 = self_r.split_at_distance(distance_meters=d)
            self_segments_rev.append(s1)
            self_r = s0
        # Append remaining prefix (handles zero-intersection case where loop does not run)
        self_segments_rev.append(self_r)
        other_segments_rev = []
        other_r = other
        for d in other_split_dists[::-1]:
            s0, s1 = other_r.split_at_distance(distance_meters=d)
            other_segments_rev.append(s1)
            other_r = s0
        # Append remaining prefix (handles zero-intersection case where loop does not run)
        other_segments_rev.append(other_r)
        return tuple(self_segments_rev[::-1]), tuple(other_segments_rev[::-1])

    def snap_at(self, other, tolerance: float = 1e-3):
        """Snap at other geometry."""
        return Route.from_line_string(
            line_string=snap(self.line_string, other, tolerance=tolerance),
            time=(w.time for w in self.way_points),
        )

    def resample_with_distance(self, distances_meters: float = None):
        """Resample route to given distances."""
        return Route(
            way_points=tuple(
                (self.waypoint_at_distance(distance_meters=d) for d in distances_meters)
            )
        )

    @profile
    def cost_gradient_across_track_left(
        self,
        n: int = None,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        distance_meters: float = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        route_mod_fwd = self.move_waypoint(
            n=n,
            azimuth_degrees=self.waypoint_azimuth(n=n) - 90.0,
            distance_meters=0.5 * distance_meters,
        )
        route_mod_bwd = self.move_waypoint(
            n=n,
            azimuth_degrees=self.waypoint_azimuth(n=n) - 90.0,
            distance_meters=-0.5 * distance_meters,
        )
        return (
            route_mod_fwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
            - route_mod_bwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
        ) / distance_meters

    @profile
    def cost_gradient_along_track(
        self,
        n: int = None,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        distance_meters: float = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        route_mod_fwd = self.move_waypoint(
            n=n,
            azimuth_degrees=self.waypoint_azimuth(n=n),
            distance_meters=0.5 * distance_meters,
        )
        route_mod_bwd = self.move_waypoint(
            n=n,
            azimuth_degrees=self.waypoint_azimuth(n=n),
            distance_meters=-0.5 * distance_meters,
        )
        return (
            route_mod_fwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
            - route_mod_bwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
        ) / distance_meters

    @profile
    def cost_gradient_time_shift(
        self,
        n: int = None,
        current_data_set: xr.Dataset = None,
        wind_data_set: xr.Dataset = None,
        wave_data_set: xr.Dataset = None,
        time_shift_seconds: float = None,
        ship: Ship = SHIP_DEFAULT,
        physics: Physics = PHYSICS_DEFAULT,
    ):
        # This is important: If we just do 0.5 * np.timedelta65(1, "s") * ..., this will
        # map to timediff=0 in the internal integer representation of np.timedelta64 ...
        time_diff_np = time_shift_seconds * 1000 * np.timedelta64(1, "ms")

        route_mod_fwd = self.replace_waypoint(
            n=n,
            new_way_point=self.way_points[n].move_time(time_diff=0.5 * time_diff_np),
        )
        route_mod_bwd = self.replace_waypoint(
            n=n,
            new_way_point=self.way_points[n].move_time(time_diff=-0.5 * time_diff_np),
        )
        return (
            route_mod_fwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
            - route_mod_bwd.cost_through(
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                ship=ship,
                physics=physics,
            )
        ) / time_shift_seconds

    def move_waypoints_left_nonlocal(
        self,
        center_distance_meters: float = None,
        width_meters: float = None,
        max_move_meters: float = None,
    ):
        """Smoothly move multiple waypoints to the left of the route.

        Modifications are done according to a Hann window of width_meters
        which is centered about center_distance_meters.

        Parameters
        ----------
        center_distance_meters: float
            Central distance (along route) of the modification.
        width_meters: float
            Width of the modification window.
        max_move_meters: float
            Max distance of the move.

        Returns
        -------
        Route
            With modified locations of waypoints and unchanged times.

        """
        _x = np.linspace(-1, 1, 101) * width_meters / 2
        _y = hann(101)
        c_hann = interp1d(_x, _y, fill_value=0.0, bounds_error=False)
        rel_distances = np.array(self.distance_meters) - center_distance_meters
        mod_dists = c_hann(rel_distances) * max_move_meters
        az_left = [self.waypoint_azimuth(n=n) - 90.0 for n in range(len(self))]
        _route = self
        for n in range(len(self)):
            _route = _route.move_waypoint(
                n=n,
                azimuth_degrees=az_left[n],
                distance_meters=mod_dists[n],
            )
        return _route
