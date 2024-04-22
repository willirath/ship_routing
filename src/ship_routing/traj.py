import pandas as pd
import numpy as np
from shapely.geometry import LineString

from .config import (
    DIST_OFFSET_SLICING,
    DEFAULT_START_TIME,
)

from .geodesics import (
    refine_along_great_circle,
    move_first_point_left,
    move_second_point_left,
    move_middle_point_left,
    get_length_meters,
    get_dist_along,
)

from .remix import segment_lines_with_each_other

from .cost import power_for_leg_in_ocean


from scipy.interpolate import interp1d


class Trajectory(object):
    def __init__(
        self,
        lon=None,
        lat=None,
        duration_seconds: float = np.nan,
        start_time: str | np.datetime64 = DEFAULT_START_TIME,
    ):
        """Trajectory.

        Parameters
        ----------
        lon: array
            Longitudes.
        lat: array
            Latitudes.
        duration_seconds: float
            Duration of the journey.
        start_time: timestamp
            Start time stamp.
        """
        if np.isscalar(lon):
            raise ValueError(
                "Trajectory must have at least 2 way points. They can be identical."
            )
        self.lon = list(lon)
        self.lat = list(lat)
        self.duration_seconds = duration_seconds
        if np.isnan(duration_seconds):
            duration_milliseconds = "NaT"
        else:
            duration_milliseconds = round(1000 * duration_seconds)
        self.time = [
            np.datetime64(start_time)
            + np.timedelta64(duration_milliseconds, "ms")
            * d
            / (self.length_meters + 1e-15)
            for d in self.dist
        ]

    def __len__(self):
        return len(self.lon)

    def __getitem__(self, key):
        lon = self.lon[key]
        lat = self.lat[key]
        time = self.time[key]
        if np.isscalar(lon):
            lon = [lon, lon]
            lat = [lat, lat]
            time = [time, time]
        elif len(lon) == 1:
            lon = [lon[0], lon[0]]
            lat = [lat[0], lat[0]]
            time = [time[0], time[0]]
        duration = (time[-1] - time[0]) / np.timedelta64(1, "s")
        return Trajectory(
            lon=lon, lat=lat, duration_seconds=duration, start_time=time[0]
        )

    @property
    def data_frame(self):
        return pd.DataFrame(dict(lon=self.lon, lat=self.lat, dist=self.dist))

    @property
    def time_since_start(self):
        return [t / np.timedelta64(1, "s") for t in (self.time - self.time[0])]

    @property
    def start_time(self):
        return self.time[0]

    @property
    def speed_ms(self):
        return self.length_meters / self.duration_seconds

    @property
    def length_meters(self):
        if len(self) > 1:
            return get_length_meters(self.line_string)
        else:
            return 0

    @property
    def line_string(self):
        return LineString(list(zip(self.lon, self.lat)))

    @classmethod
    def from_line_string(cls, line_string=None, **kwargs):
        return cls(lon=line_string.xy[0], lat=line_string.xy[1], **kwargs)

    @classmethod
    def from_data_frame(cls, data_frame=None, **kwargs):
        lon = data_frame["lon"]
        lat = data_frame["lat"]
        return cls(lon=lon, lat=lat, **kwargs)

    def refine(self, new_dist: float = None):
        lon, lat = refine_along_great_circle(
            lon=self.lon, lat=self.lat, new_dist=new_dist
        )
        return Trajectory(
            lon=lon,
            lat=lat,
            duration_seconds=self.duration_seconds,
            start_time=self.start_time,
        )

    def __repr__(self):
        return repr(self.data_frame)

    def move_node_left(self, node_num: int = None, move_by_meters: float = 0):
        lstr = self.line_string
        if node_num == 0:
            lon1, lat1 = list(lstr.coords)[node_num]
            lon2, lat2 = list(lstr.coords)[node_num + 1]
            lon_moved, lat_moved = move_first_point_left(
                lon1=lon1,
                lat1=lat1,
                lon2=lon2,
                lat2=lat2,
                move_by_meters=move_by_meters,
            )
        elif node_num == len(self) - 1:
            lon1, lat1 = list(lstr.coords)[node_num - 1]
            lon2, lat2 = list(lstr.coords)[node_num]
            lon_moved, lat_moved = move_second_point_left(
                lon1=lon1,
                lat1=lat1,
                lon2=lon2,
                lat2=lat2,
                move_by_meters=move_by_meters,
            )
        else:
            lon1, lat1 = list(lstr.coords)[node_num - 1]
            lon2, lat2 = list(lstr.coords)[node_num]
            lon3, lat3 = list(lstr.coords)[node_num + 1]
            lon_moved, lat_moved = move_middle_point_left(
                lon1=lon1,
                lat1=lat1,
                lon2=lon2,
                lat2=lat2,
                lon3=lon3,
                lat3=lat3,
                move_by_meters=move_by_meters,
            )
        coords = list(lstr.coords)
        coords[node_num] = (lon_moved, lat_moved)
        lstr_new = LineString(coords)
        return Trajectory(
            lon=lstr_new.xy[0],
            lat=lstr_new.xy[1],
            duration_seconds=self.duration_seconds,
            start_time=self.start_time,
        )

    def estimate_cost_through(self, data_set=None):
        return sum(self.estimate_cost_per_leg_through(data_set=data_set))

    def estimate_cost_per_leg_through(self, data_set=None):
        cost_per_leg = [
            _leg_dur
            * power_for_leg_in_ocean(
                leg_pos=_leg_pos,
                leg_speed=_leg_speed,
                ocean_data=data_set,
            )
            for _leg_pos, _leg_speed, _leg_dur in zip(
                self.legs_pos,
                self.legs_speed,
                self.legs_duration,
            )
        ]
        return cost_per_leg

    @property
    def dist(self):
        if len(self) > 1:
            return get_dist_along(self.line_string)
        else:
            return [
                0,
            ]

    def add_waypoint(self, dist: float = None):
        data_frame = self.data_frame
        data_frame = data_frame.set_index("dist")
        data_frame = data_frame.join(
            pd.DataFrame(
                {},
                index=[
                    dist,
                ],
            ),
            how="outer",
        ).interpolate(method="index")
        # data_frame = data_frame.round(decimals=5).drop_duplicates()
        return Trajectory(
            lon=data_frame.lon,
            lat=data_frame.lat,
            duration_seconds=self.duration_seconds,
        )

    def slice_with_dist(self, d0: float = None, d1: float = None):
        if d1 <= d0:
            raise ValueError(f"d1={d1} needs to be larger than d0={d0}.")
        _traj = self.add_waypoint(dist=d0).add_waypoint(dist=d1)
        _traj_df = _traj.data_frame.set_index("dist")
        _d0 = d0 - DIST_OFFSET_SLICING
        _d1 = d1 + DIST_OFFSET_SLICING
        _sub_traj_df = _traj_df.loc[_d0:_d1]
        return Trajectory(
            lon=_sub_traj_df.lon,
            lat=_sub_traj_df.lat,
            duration_seconds=(d1 - d0) / self.length_meters * self.duration_seconds,
        )

    def segment_at_other_traj(self, other):
        self_line = self.line_string
        other_line = other.line_string

        self_line_segments, other_line_segments = segment_lines_with_each_other(
            line_0=self_line, line_1=other_line
        )

        self_dist = [0] + list(
            np.cumsum([get_length_meters(s) for s in self_line_segments])
        )
        other_dist = [0] + list(
            np.cumsum([get_length_meters(s) for s in other_line_segments])
        )

        self_segments = [
            self.slice_with_dist(d0=d0, d1=d1)
            for d0, d1 in zip(self_dist[:-1], self_dist[1:])
        ]
        other_segments = [
            other.slice_with_dist(d0=d0, d1=d1)
            for d0, d1 in zip(other_dist[:-1], other_dist[1:])
        ]

        return self_segments, other_segments

    def __add__(self, other):
        data_frame = pd.concat(
            [self.data_frame[["lon", "lat"]], other.data_frame[["lon", "lat"]]]
        )
        data_frame = data_frame.drop_duplicates()
        duration_seconds = self.duration_seconds + other.duration_seconds
        return Trajectory(
            lon=data_frame.lon,
            lat=data_frame.lat,
            duration_seconds=duration_seconds,
            start_time=self.start_time,
        )

    def copy(self):
        return Trajectory(
            lon=self.lon,
            lat=self.lat,
            duration_seconds=self.duration_seconds,
            start_time=self.start_time,
        )

    def homogenize(self):
        n = len(self)
        dist = self.dist
        _lon = interp1d(x=dist, y=self.lon)
        _lat = interp1d(x=dist, y=self.lat)
        new_dist = np.linspace(0, self.length_meters, n)
        new_lon = list(_lon(new_dist))
        new_lat = list(_lat(new_dist))
        return Trajectory(
            lon=new_lon,
            lat=new_lat,
            duration_seconds=self.duration_seconds,
            start_time=self.start_time,
        )

    @property
    def legs_pos(self):
        return tuple(
            [
                ((lon0, lat0), (lon1, lat1))
                for lon0, lat0, lon1, lat1 in zip(
                    self.lon[:-1],
                    self.lat[:-1],
                    self.lon[1:],
                    self.lat[1:],
                )
            ]
        )

    @property
    def legs_duration(self):
        return tuple(
            t1 - t0
            for t0, t1 in zip(self.time_since_start[:-1], self.time_since_start[1:])
        )

    @property
    def legs_time_since_start(self):
        return tuple(zip(self.time_since_start[:-1], self.time_since_start[1:]))

    @property
    def legs_length_meters(self):
        dist = self.dist
        return tuple(d1 - d0 for d0, d1 in zip(dist[:-1], dist[1:]))

    @property
    def legs_speed(self):
        leg_dur = self.legs_duration
        leg_len = self.legs_length_meters
        return tuple(l / d for l, d in zip(leg_len, leg_dur))
