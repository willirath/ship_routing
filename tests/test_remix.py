import shapely.geometry
from ship_routing.remix import segment_lines_with_each_other

import shapely


def test_segmentation_returns_simple_line_strings():
    # two lines w/ different starting points, w/ same end point, first one intersects itself
    l0 = shapely.geometry.LineString([[0, 0.1], [1, 0], [1, 1], [-0.5, -1.5]])
    l1 = shapely.geometry.LineString([[0, -0.1], [-0.5, -0.5], [0.5, -1], [-0.5, -1.5]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert all([isinstance(s, shapely.geometry.LineString) for s in l0s])
    assert all([isinstance(s, shapely.geometry.LineString) for s in l1s])


def test_segmentation_returns_same_num_of_segments():
    # two lines w/ different starting points, w/ same end point, first one intersects itself
    l0 = shapely.geometry.LineString([[0, 0.1], [1, 0], [1, 1], [-0.5, -1.5]])
    l1 = shapely.geometry.LineString([[0, -0.1], [-0.5, -0.5], [0.5, -1], [-0.5, -1.5]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert len(l0s) == len(l1s)


def test_setmentation_handles_overlapping_lines():
    l0 = shapely.geometry.LineString([[0, 0], [3, 0]])
    l1 = shapely.geometry.LineString([[0, 0], [1, -1], [2, 0], [3, 0]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert all([isinstance(s, shapely.geometry.LineString) for s in l0s])
    assert all([isinstance(s, shapely.geometry.LineString) for s in l1s])
    assert len(l0s) == len(l1s)
