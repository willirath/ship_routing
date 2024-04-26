from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString, Point
from scipy.interpolate import interp1d

from typing import Iterable, Tuple

from .geodesics import (
    get_length_meters,
    get_distance_meters,
    move_fwd,
    refine_along_great_circle,
    get_leg_azimuth,
)

from .currents import select_currents_for_leg

from .cost import power_maintain_speed


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

    def cost_through(self, current_data_set: xr.Dataset = None):
        us, vs = self.uv_over_ground_ms
        ds_uovo = select_currents_for_leg(
            ds=current_data_set,
            lon_start=self.way_point_start.lon,
            lat_start=self.way_point_start.lat,
            time_start=self.way_point_start.time,
            lon_end=self.way_point_end.lon,
            lat_end=self.way_point_end.lat,
            time_end=self.way_point_end.time,
        )
        return (
            power_maintain_speed(uo=ds_uovo.uo, vo=ds_uovo.vo, us=us, vs=vs)
            .mean()
            .data[()]
            * self.duration_seconds
        )


@dataclass(frozen=True)
class Route:
    """A route containing of multiple waypoints."""

    way_points: Tuple

    def __post_init__(self):
        if not isinstance(self.way_points, tuple):
            raise ValueError("Way_points need to be a tuple.")
        if not len(self.way_points) >= 2:
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
        return Route(way_points=self.way_points + other.way_points)

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

    def remove_consecutive_duplicate_timesteps(self):
        """Route with the first of each 2 consecutive way points having the same time stamp."""

        def generate_non_dupe_wps(wps):
            current_wp = wps[0]
            yield current_wp
            n = 0
            while n < len(wps) - 1:
                n += 1
                candidate_wp = wps[n]
                if candidate_wp.time > current_wp.time:
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

    def move_waypoint(
        self,
        n: int = None,
        azimuth_degrees: float = None,  # abs azimuth !!!!
        distance_meters: float = None,
    ):
        """Move nth waypoint."""
        wps_before = self.way_points[:n]
        wps_after = self.way_points[n + 1 :]
        wp_orig = self.way_points[n]
        wp_moved = wp_orig.move_space(
            azimuth_degrees=azimuth_degrees,
            distance_meters=distance_meters,
        )
        return Route(way_points=wps_before + (wp_moved,) + wps_after)

    def cost_through(self, current_data_set: xr.Dataset = None):
        """Cost along whole route."""
        return sum(self.cost_per_leg_through(current_data_set=current_data_set))

    def cost_per_leg_through(self, current_data_set: xr.Dataset = None):
        """Cost along each leg."""
        return tuple(
            (l.cost_through(current_data_set=current_data_set) for l in self.legs)
        )

    def get_waypoint_azimuth(self, n: int = None):
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
