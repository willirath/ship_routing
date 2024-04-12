import shapely
from shapely.ops import split
from shapely import snap

RESOLUTION = 1e-5


def segment_lines_with_each_other(
    line_0: shapely.geometry.LineString = None,
    line_1: shapely.geometry.LineString = None,
    resolution: float = RESOLUTION,
) -> tuple:
    # find intersections (returns points _and_ linestrings if lines are overlapping)
    splitter = list(shapely.intersection(line_0, line_1, grid_size=RESOLUTION).geoms)

    # replace linestrings by their endpoints by first splitting linestrings into start / end and
    # then building one long list of points
    def _transform_to_list_of_points(g):
        if type(g) is shapely.geometry.LineString:
            return list(g.boundary.geoms)
        else:
            return [g]

    splitter = sum([_transform_to_list_of_points(g) for g in splitter], start=[])

    # cast to proper geometry again
    splitter = shapely.geometry.MultiPoint(splitter)

    # segment by first snapping into splitting points and then
    line_0_segments = list(
        split(snap(line_0, splitter, tolerance=RESOLUTION), splitter).geoms
    )
    line_1_segments = list(
        split(snap(line_1, splitter, tolerance=RESOLUTION), splitter).geoms
    )

    return line_0_segments, line_1_segments
