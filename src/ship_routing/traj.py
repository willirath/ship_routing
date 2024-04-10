import pandas as pd
import numpy as np
from shapely.geometry import LineString


from .geodesics import refine_along_great_circle, move_first_point_left


class Trajectory(object):
    def __init__(self, lon=None, lat=None):
        """Trajectory.

        Parameters
        ----------
        lon: array
            Longitudes.
        lat: array
            Latitudes.
        """
        self.data_frame = pd.DataFrame(
            dict(
                lon=lon,
                lat=lat,
            )
        )

    def __len__(self):
        return len(self.data_frame)

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
        lstr_seg = LineString(list(lstr.coords)[node_num : node_num + 2])
        lstr_seg_moved = move_first_point_left(lstr_seg, move_by_meters=move_by_meters)
        lstr_new = LineString(
            list(lstr.coords)[:node_num]
            + list(lstr_seg_moved.coords)
            + list(lstr.coords)[max(node_num+2, len(lstr.coords)-1):])
        return Trajectory.from_line_string(lstr_new)