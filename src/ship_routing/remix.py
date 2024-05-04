import shapely
from shapely.ops import split, linemerge
from shapely import snap

SHAPELY_RESOLUTION = 1e-3


def segment_lines_with_each_other(
    line_0: shapely.geometry.LineString = None,
    line_1: shapely.geometry.LineString = None,
    resolution: float = SHAPELY_RESOLUTION,
) -> tuple:
    # find intersections (returns points _and_ linestrings if lines are overlapping)
    _splitter = list(
        shapely.intersection(line_0, line_1, grid_size=SHAPELY_RESOLUTION).geoms
    )

    # replace linestrings by their endpoints by first splitting linestrings into start / end and
    # then building one long list of points
    def _merge_all_ls(s):
        _ls = []
        for e in s:
            if isinstance(e, shapely.geometry.Point):
                if len(_ls) > 0:
                    for p in list(linemerge(_ls).boundary.geoms):
                        yield p
                    _ls = []
                yield e
            else:
                _ls.append(e)
        if len(_ls) > 0:
            for p in list(linemerge(_ls).boundary.geoms):
                yield p

    splitter = list(_merge_all_ls(_splitter))

    # cast to proper geometry again
    splitter = shapely.geometry.MultiPoint(splitter)

    # segment by first snapping into splitting points and then
    line_0_segments = list(
        split(snap(line_0, splitter, tolerance=SHAPELY_RESOLUTION), splitter).geoms
    )
    line_1_segments = list(
        split(snap(line_1, splitter, tolerance=SHAPELY_RESOLUTION), splitter).geoms
    )

    return line_0_segments, line_1_segments
