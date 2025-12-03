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


def compute_ellipse_bbox(
    lon_start: float = None,
    lat_start: float = None,
    lon_end: float = None,
    lat_end: float = None,
    multiplier: float = None,  # rename to speed_og_multiplier
    buffer: float = 1.0,  # TODO: buffer_degrees
    grid_resolution: float = None,
) -> tuple:
    """Compute bounding box of geodesic ellipse containing all allowed routes.

    For a journey with two waypoints (start and end), computes a lat/lon
    bounding box that contains all routes where the path length is at most
    N times the direct geodesic distance.

    Uses local Azimuthal Equidistant projection to handle edge cases
    (antimeridian crossing, polar regions) automatically.

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
    multiplier : float
        Maximum route length as multiple of direct distance (e.g., 4.0)
    buffer : float, defaults=1.0
        Safety buffer in degrees to add beyond ellipse bbox
    grid_resolution : float
        Grid resolution in degrees for sampling ellipse points

    Returns
    -------
    tuple of float
        (lon_min, lon_max, lat_min, lat_max) bounding box in degrees
    """
    # Compute route center point
    lon_center = (lon_start + lon_end) / 2.0
    lat_center = (lat_start + lat_end) / 2.0

    # Create local Azimuthal Equidistant CRS centered at route midpoint
    local_proj_dict = (
        f"+proj=aeqd +lon_0={lon_center} +lat_0={lat_center} "
        "+ellps=WGS84 +units=m"
    )

    # Create bidirectional transformer between EPSG:4326 and local projection
    transformer_to_local = pyproj.Transformer.from_crs(
        "EPSG:4326", local_proj_dict, always_xy=True
    )
    transformer_to_latlon = pyproj.Transformer.from_crs(
        local_proj_dict, "EPSG:4326", always_xy=True
    )

    # Transform start and end points to local xy coordinates
    x_start, y_start = transformer_to_local.transform(lon_start, lat_start)
    x_end, y_end = transformer_to_local.transform(lon_end, lat_end)

    # Compute direct distance in local xy coordinates
    direct_dist = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    # Compute ellipse parameters in xy space (Euclidean)
    # focal distance (center to focus)
    c = direct_dist / 2.0
    # semi-major axis
    a = multiplier * c
    # semi-minor axis
    if a > c:
        b = np.sqrt(a ** 2 - c ** 2)
    else:
        b = 0.0

    # Create rough bbox in xy space around the route with buffer
    # Convert grid_resolution from degrees to meters (~111 km per degree)
    grid_resolution_m = grid_resolution * 111_000
    buffer_m = buffer * 111_000

    # Estimate rough bbox in xy space
    xy_extent = max(direct_dist, a) + buffer_m
    x_min_rough = -xy_extent
    x_max_rough = xy_extent
    y_min_rough = -xy_extent
    y_max_rough = xy_extent

    # Generate xy grid
    x_grid = np.arange(x_min_rough, x_max_rough + grid_resolution_m, grid_resolution_m)
    y_grid = np.arange(y_min_rough, y_max_rough + grid_resolution_m, grid_resolution_m)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Compute sum of distances to both foci for all grid points (Euclidean)
    dist_to_start = np.sqrt((x_mesh - x_start) ** 2 + (y_mesh - y_start) ** 2)
    dist_to_end = np.sqrt((x_mesh - x_end) ** 2 + (y_mesh - y_end) ** 2)
    sum_distances = dist_to_start + dist_to_end

    # Filter points within ellipse (with small tolerance)
    max_sum_dist = multiplier * direct_dist
    ellipse_mask = sum_distances <= (max_sum_dist + grid_resolution_m)

    # Find bbox in xy space
    if np.any(ellipse_mask):
        x_ellipse = x_mesh[ellipse_mask]
        y_ellipse = y_mesh[ellipse_mask]

        x_min_xy = np.min(x_ellipse) - buffer_m
        x_max_xy = np.max(x_ellipse) + buffer_m
        y_min_xy = np.min(y_ellipse) - buffer_m
        y_max_xy = np.max(y_ellipse) + buffer_m
    else:
        # Fallback if no points found (shouldn't happen with reasonable parameters)
        x_min_xy = -max_sum_dist - buffer_m
        x_max_xy = max_sum_dist + buffer_m
        y_min_xy = -max_sum_dist - buffer_m
        y_max_xy = max_sum_dist + buffer_m

    # Transform bbox corners + edge midpoints back to lat/lon for accuracy
    bbox_corners_xy = [
        (x_min_xy, y_min_xy),
        (x_max_xy, y_min_xy),
        (x_min_xy, y_max_xy),
        (x_max_xy, y_max_xy),
        ((x_min_xy + x_max_xy) / 2, y_min_xy),
        ((x_min_xy + x_max_xy) / 2, y_max_xy),
        (x_min_xy, (y_min_xy + y_max_xy) / 2),
        (x_max_xy, (y_min_xy + y_max_xy) / 2),
    ]

    # Transform corners back to lat/lon
    bbox_corners_latlon = [
        transformer_to_latlon.transform(x, y) for x, y in bbox_corners_xy
    ]

    # Find min/max lat/lon
    lons = [pt[0] for pt in bbox_corners_latlon]
    lats = [pt[1] for pt in bbox_corners_latlon]

    lon_min = np.min(lons)
    lon_max = np.max(lons)
    lat_min = np.min(lats)
    lat_max = np.max(lats)

    # Clamp latitudes to valid range
    lat_min = np.clip(lat_min, -90, 90)
    lat_max = np.clip(lat_max, -90, 90)

    return (lon_min, lon_max, lat_min, lat_max)
