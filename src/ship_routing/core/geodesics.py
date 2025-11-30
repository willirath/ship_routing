import numpy as np
import pint
import pyproj
from shapely.geometry import LineString

from collections import namedtuple

# Create unit registry once at module level
_ureg = pint.UnitRegistry()


def knots_to_ms(speed_knots: float) -> float:
    """Convert speed from knots to meters per second."""
    return float((speed_knots * _ureg.knot) / _ureg.meter_per_second)


def ms_to_knots(speed_ms: float) -> float:
    """Convert speed from meters per second to knots."""
    return float((speed_ms * _ureg.meter_per_second) / _ureg.knot)


def move_fwd(
    lon: float = None,
    lat: float = None,
    azimuth_degrees: float = None,
    distance_meters: float = None,
) -> tuple:
    """Move forward from a point along a geodesic.

    Parameters
    ----------
    lon : float
        Starting longitude in degrees
    lat : float
        Starting latitude in degrees
    azimuth_degrees : float
        Forward azimuth in degrees
    distance_meters : float
        Distance to move in meters

    Returns
    -------
    tuple of float
        New (longitude, latitude) in degrees
    """
    geod = pyproj.Geod(
        ellps="WGS84"
    )  # TODO: move the geod out of this function for re-use
    lon_new, lat_new, _ = geod.fwd(
        lons=lon, lats=lat, az=azimuth_degrees, dist=distance_meters, radians=False
    )
    return lon_new, lat_new


def get_distance_meters(
    lon_start: float = None,
    lon_end: float = None,
    lat_start: float = None,
    lat_end: float = None,
):
    """Calculate geodesic distance between two points.

    Parameters
    ----------
    lon_start : float
        Starting longitude in degrees
    lon_end : float
        Ending longitude in degrees
    lat_start : float
        Starting latitude in degrees
    lat_end : float
        Ending latitude in degrees

    Returns
    -------
    float
        Distance in meters along geodesic
    """
    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance_meters = geod.inv(
        lons1=lon_start,
        lons2=lon_end,
        lats1=lat_start,
        lats2=lat_end,
    )
    return distance_meters


def get_length_meters(line_string: LineString = None) -> float:
    """Calculate geodesic length of a LineString geometry."""
    geod = pyproj.Geod(ellps="WGS84")

    return geod.geometry_length(line_string)


def get_refinement_factor(
    original_dist: float = None,
    new_dist: float = None,
) -> int:
    """Calculate number of segments needed to refine a distance.

    Parameters
    ----------
    original_dist : float
        Original segment distance in meters
    new_dist : float
        Target segment distance in meters

    Returns
    -------
    int
        Number of segments required
    """
    return int(np.ceil(original_dist / new_dist))


def refine_along_great_circle(
    lon: float = None,
    lat: float = None,
    new_dist: float = None,
) -> tuple:
    """Refine a route by adding intermediate points along great circles.

    Takes waypoints and adds intermediate points along geodesic segments
    to ensure segment lengths do not exceed the specified distance.

    Parameters
    ----------
    lon : array-like of float
        Waypoint longitudes in degrees
    lat : array-like of float
        Waypoint latitudes in degrees
    new_dist : float
        Maximum segment distance in meters

    Returns
    -------
    tuple of list
        Refined (longitudes, latitudes) with intermediate points added
    """
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
        """Wrapper for pyproj inv_intermediate handling zero-point edge case."""
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
            [lon_s] + list(lons_int)
            for lon_s, lons_int in zip(lon_start, lon_intermediate)
        ],
        start=[],
    ) + [
        lon_end[-1],
    ]
    lat_refined = sum(
        [
            [lat_s] + list(lats_int)
            for lat_s, lats_int in zip(lat_start, lat_intermediate)
        ],
        start=[],
    ) + [
        lat_end[-1],
    ]
    return lon_refined, lat_refined


def get_leg_azimuth(
    lon_start: float = None,
    lat_start: float = None,
    lon_end: float = None,
    lat_end: float = None,
):
    """Calculate azimuths for a geodesic leg.

    Parameters
    ----------
    lon_start : float
        Starting longitude in degrees
    lat_start : float
        Starting latitude in degrees
    lon_end : float
        Ending longitude in degrees
    lat_end : float
        Ending latitude in degrees

    Returns
    -------
    tuple of float
        (average_azimuth, forward_azimuth, backward_azimuth) in degrees
    """
    geod = pyproj.Geod(ellps="WGS84")
    fwd_az, bwd_az, _ = geod.inv(
        lons1=lon_start,
        lons2=lon_end,
        lats1=lat_start,
        lats2=lat_end,
        return_back_azimuth=False,
    )
    return (fwd_az + bwd_az) / 2.0, fwd_az, bwd_az
