from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from typing import Iterable, Tuple


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
