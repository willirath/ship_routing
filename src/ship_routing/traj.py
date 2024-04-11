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


class Trajectory(object):
    def __init__(self, lon=None, lat=None, duration_seconds: float = None):
        """Trajectory.

        Parameters
        ----------
        lon: array
            Longitudes.
        lat: array
            Latitudes.
        duration: float
            Duration of the journey.
        """
        self.data_frame = pd.DataFrame(
            dict(
                lon=lon,
                lat=lat,
            )
        )
        self.duration_seconds = duration_seconds

    def __len__(self):
        return len(self.data_frame)

    @property
    def speed_ms(self):
        return self.length_meters / self.duration_seconds

    @property
    def length_meters(self):
        return get_length_meters(self.line_string)

    @property
    def lon(self):
        return self.data_frame["lon"]

    @property
    def lat(self):
        return self.data_frame["lat"]

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
        return Trajectory(lon=lon, lat=lat)

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
        return Trajectory.from_line_string(lstr_new)
