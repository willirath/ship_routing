import pandas as pd
import numpy as np
from shapely.geometry import LineString

from .config import DIST_OFFSET_SLICING

from .geodesics import (
    refine_along_great_circle,
    move_first_point_left,
    move_second_point_left,
    move_middle_point_left,
    get_length_meters,
    get_dist_along,
)

from .remix import segment_lines_with_each_other

from .cost import power_for_traj_in_ocean


from scipy.interpolate import interp1d


class Trajectory(object):
    def __init__(self, lon=None, lat=None, duration_seconds: float = np.nan):
        """Trajectory.

        Parameters
        ----------
        lon: array
            Longitudes.
        lat: array
            Latitudes.
        duration_seconds: float
            Duration of the journey.
        """
        if np.isscalar(lon):
            lon = [
                lon,
            ]
            lat = [
                lat,
            ]
        self.lon = list(lon)
        self.lat = list(lat)
        self.duration_seconds = duration_seconds

    def __len__(self):
        return len(self.lon)

    def __getitem__(self, key):
        _traj = Trajectory(
            lon=self.lon[key],
            lat=self.lat[key],
        )
        if len(_traj) == 1:
            duration = 0
        else:
            duration = _traj.length_meters / self.speed_ms
        return Trajectory(
            lon=self.lon[key], lat=self.lat[key], duration_seconds=duration
        )

    @property
    def data_frame(self):
        return pd.DataFrame(dict(lon=self.lon, lat=self.lat, dist=self.dist))

    @property
    def time_since_start(self):
        return [d / self.speed_ms for d in self.dist]

    @property
    def speed_ms(self):
        return self.length_meters / self.duration_seconds

    @property
    def length_meters(self):
        return get_length_meters(self.line_string)

    @property
    def line_string(self):
        return LineString(list(zip(self.lon, self.lat)))

    @classmethod
    def from_line_string(cls, line_string=None):
        return cls(lon=line_string.xy[0], lat=line_string.xy[1])

    @classmethod
    def from_data_frame(cls, data_frame=None):
        lon = data_frame["lon"]
        lat = data_frame["lat"]
        return cls(lon=lon, lat=lat)

    def refine(self, new_dist: float = None):
        lon, lat = refine_along_great_circle(
            lon=self.lon, lat=self.lat, new_dist=new_dist
        )
        return Trajectory(lon=lon, lat=lat, duration_seconds=self.duration_seconds)

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
        )

    def estimate_cost_through(self, data_set=None):
        pwr = power_for_traj_in_ocean(
            ship_positions=self.refine(new_dist=20_000).data_frame,
            speed=self.speed_ms,
            ocean_data=data_set,
        )
        if pwr.isnull().sum() > 0:
            return np.nan
        else:
            return pwr.sum().data[()]

    @property
    def dist(self):
        return get_dist_along(self.line_string)

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
        # data_frame = data_frame.loc[data_frame.diff() != 0]
        duration_seconds = self.duration_seconds + other.duration_seconds
        return Trajectory(
            lon=data_frame.lon, lat=data_frame.lat, duration_seconds=duration_seconds
        )

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Trajectory(
            lon=self.lon, lat=self.lat, duration_seconds=self.duration_seconds
        )

    def __deepcopy__(self):
        return Trajectory(
            lon=self.lon, lat=self.lat, duration_seconds=self.duration_seconds
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
            lon=new_lon, lat=new_lat, duration_seconds=self.duration_seconds
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
    def legs_length_meters(self):
        dist = self.dist
        return tuple(d1 - d0 for d0, d1 in zip(dist[:-1], dist[1:]))

    @property
    def legs_speed(self):
        leg_dur = self.legs_duration
        leg_len = self.legs_length_meters
        return tuple(l / d for l, d in zip(leg_len, leg_dur))
