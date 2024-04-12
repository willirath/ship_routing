import numpy as np
import pyproj
from shapely.geometry import LineString

from collections import namedtuple


def get_length_meters(line_string: LineString = None) -> float:
    geod = pyproj.Geod(ellps="WGS84")

    return geod.geometry_length(line_string)


def get_dist_along(line_string: LineString = None) -> list:
    return [
        0,
    ] + [
        get_length_meters(LineString(list(line_string.coords)[:n]))
        for n in range(2, len(list(line_string.coords)) + 1)
    ]


def move_middle_point_left(
    lon1: float = None,
    lat1: float = None,
    lon2: float = None,
    lat2: float = None,
    lon3: float = None,
    lat3: float = None,
    move_by_meters: float = 0,
) -> tuple:
    geod = pyproj.Geod(ellps="WGS84")

    fwd_az_23, _, _ = geod.inv(
        lons1=lon2, lats1=lat2, lons2=lon3, lats2=lat3, return_back_azimuth=True
    )
    _, fwd_az_12, _ = geod.inv(
        lons1=lon1, lats1=lat1, lons2=lon2, lats2=lat2, return_back_azimuth=False
    )
    fwd_az = (fwd_az_12 + fwd_az_23) / 2.0
    rot_left = fwd_az - 90.0

    lon2_new, lat2_new, _ = geod.fwd(
        lons=lon2, lats=lat2, az=rot_left, dist=move_by_meters
    )

    return lon2_new, lat2_new


def move_first_point_left(
    lon1: float = None,
    lat1: float = None,
    lon2: float = None,
    lat2: float = None,
    move_by_meters: float = 0,
) -> tuple:
    geod = pyproj.Geod(ellps="WGS84")

    fwd_az, _, _ = geod.inv(
        lons1=lon1, lats1=lat1, lons2=lon2, lats2=lat2, return_back_azimuth=False
    )
    rot_left = fwd_az - 90.0

    lon1_new, lat1_new, _ = geod.fwd(
        lons=lon1, lats=lat1, az=rot_left, dist=move_by_meters
    )

    return lon1_new, lat1_new


def move_second_point_left(
    lon1: float = None,
    lat1: float = None,
    lon2: float = None,
    lat2: float = None,
    move_by_meters: float = 0,
) -> tuple:
    return move_first_point_left(
        lon1=lon2, lat1=lat2, lon2=lon1, lat2=lat1, move_by_meters=-move_by_meters
    )


def get_refinement_factor(
    original_dist: float = None,
    new_dist: float = None,
) -> int:
    return int(np.ceil(original_dist / new_dist))


def refine_along_great_circle(
    lon: float = None,
    lat: float = None,
    new_dist: float = None,
) -> tuple:
    # define geoid
    geod = pyproj.Geod(ellps="WGS84")

    # extract segments
    lon_start = np.array(lon)[:-1]
    lon_end = np.array(lon)[1:]
    lat_start = np.array(lat)[:-1]
    lat_end = np.array(lat)[1:]

    # calculate refinement factors
    # squeeze singleton numpy arrays and explicitly rebuild iterable
    # distances if necessary to get rid of deprecation warning from pyproj
    distances = geod.inv(
        lats1=lat_start.squeeze(),
        lons1=lon_start.squeeze(),
        lats2=lat_end.squeeze(),
        lons2=lon_end.squeeze(),
    )[2]
    if isinstance(distances, float):
        distances = [
            distances,
        ]
    refinements = [
        get_refinement_factor(original_dist=dst, new_dist=new_dist) for dst in distances
    ]

    def _wrap_geod_inv_intermediate(**kwargs):
        if kwargs["npts"] == 0:
            GG = namedtuple("G", ["lons", "lats"])
            r = GG(lons=[], lats=[])
            return r
        else:
            return geod.inv_intermediate(**kwargs)

    # get intermediate positions
    new_intermediates = [
        _wrap_geod_inv_intermediate(
            # geod.inv_intermediate(
            lon1=lon1,
            lat1=lat1,
            lon2=lon2,
            lat2=lat2,
            npts=reffac - 1,
            return_back_azimuth=True,
        )
        for lon1, lat1, lon2, lat2, reffac in zip(
            lon_start,
            lat_start,
            lon_end,
            lat_end,
            refinements,
        )
    ]
    lon_intermediate = [r.lons for r in new_intermediates]
    lat_intermediate = [r.lats for r in new_intermediates]

    # assemble
    # chain of (start, intermediates) for all segments + last end
    lon_refined = sum(
        [
            [
                lon_s,
            ]
            + list(lons_int)
            for lon_s, lons_int in zip(lon_start, lon_intermediate)
        ],
        start=[],
    ) + [
        lon_end[-1],
    ]
    lat_refined = sum(
        [
            [
                lat_s,
            ]
            + list(lats_int)
            for lat_s, lats_int in zip(lat_start, lat_intermediate)
        ],
        start=[],
    ) + [
        lat_end[-1],
    ]
    return lon_refined, lat_refined


def get_directions(lon=None, lat=None):
    geod = pyproj.Geod(ellps="WGS84")

    fwd_az, bwd_az, _ = geod.inv(
        lons1=lon[:-1],
        lons2=lon[1:],
        lats1=lat[:-1],
        lats2=lat[1:],
        return_back_azimuth=False,
    )
    az = np.array(
        [  # use fwd dir for first node
            fwd_az[0],
        ]
        + [  # use averaged dir for middle nodes
            (f + b) / 2.0 for f, b in zip(fwd_az[1:], bwd_az[:-1])
        ]
        + [  # use bwd dir for last node
            bwd_az[-1],
        ]
    )
    uhat = np.sin(np.deg2rad(az))
    vhat = np.cos(np.deg2rad(az))
    return uhat, vhat
