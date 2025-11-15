import shapely
from shapely.ops import split

SHAPELY_RESOLUTION = 1e-3


def segment_lines_with_each_other(
    line_0: shapely.geometry.LineString = None,
    line_1: shapely.geometry.LineString = None,
    resolution: float = SHAPELY_RESOLUTION,
) -> tuple:
    """Segment line strings with each other.

    Note that due to our handling of round-off errors, results will differ if lines are
    permuted.

    Parameters
    ----------
    line_0: shapely.geometry.LineString
        First line.
    line_1: shapely.geometry.LineString
        Second line.
    resolution: float
        Resolution of all operations.

    Returns
    -------
    tuple:
        Segments of line_0, segments of line_1

    """
    # set precision
    line_0 = shapely.set_precision(line_0, resolution)
    line_1 = shapely.set_precision(line_1, resolution)

    # sanity check: simple geometries (no self-intersection)
    if not (line_0.is_simple and line_1.is_simple):
        raise ValueError("Lines need to be simple (e.g. not self-intersecting).")

    # find intersections
    intersection = list(line_0.intersection(line_1).geoms)
    intersection_points = set()
    for geom in intersection:
        if isinstance(geom, shapely.geometry.Point):
            intersection_points.add(geom)
        elif isinstance(geom, shapely.geometry.LineString):
            for bg in geom.boundary.geoms:
                intersection_points.add(bg)
        else:
            raise ValueError("Intersection can only contain Points and LineStrings.")

    # split

    splitter = shapely.union_all(list(intersection_points))

    # snap lines to splitter
    line_0 = shapely.snap(line_0, reference=splitter, tolerance=2 * resolution)
    line_1 = shapely.snap(line_1, reference=splitter, tolerance=2 * resolution)

    line_0_segments = tuple(split(line_0, splitter=splitter).geoms)
    line_1_segments = tuple(split(line_1, splitter=splitter).geoms)

    return line_0_segments, line_1_segments
