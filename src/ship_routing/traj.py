import pandas as pd
import numpy as np
from shapely.geometry import LineString


from .geodesics import (
    refine_along_great_circle,
    move_first_point_left,
    move_second_point_left,
    move_middle_point_left,
    get_length_meters,
)

from .cost import power_for_traj_in_ocean


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
        return power_for_traj_in_ocean(
            ship_positions=self.data_frame, speed=self.speed_ms, ocean_data=data_set
        )

    @property
    def dist(self):
        return [
            0,
        ] + [get_length_meters(self[:n].line_string) for n in range(2, len(self) + 1)]
