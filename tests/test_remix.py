from ship_routing.remix import segment_lines_with_each_other

import shapely
import pandas as pd

from pathlib import Path


TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


def test_segmentation_returns_simple_line_strings():
    # two lines w/ shared start and end points which intersect each other 1 time
    l0 = shapely.geometry.LineString([[0, 0], [3, 0]])
    l1 = shapely.geometry.LineString([[0, 0], [1, 1], [2, -1], [3, 0]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert all([isinstance(s, shapely.geometry.LineString) for s in l0s])
    assert all([isinstance(s, shapely.geometry.LineString) for s in l1s])


def test_segmentation_returns_same_num_of_segments():
    # two lines w/ shared start and end points which intersect each other 1 time
    l0 = shapely.geometry.LineString([[0, 0], [3, 0]])
    l1 = shapely.geometry.LineString([[0, 0], [1, 1], [2, -1], [3, 0]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert len(l0s) == len(l1s)
    assert len(l0s) == 2


def test_segmentation_handles_overlapping_lines():
    l0 = shapely.geometry.LineString([[0, 0], [2, 0], [2.5, 0], [3, 0]])
    l1 = shapely.geometry.LineString([[0, 0], [1, -1], [2, 0], [2.5, 0], [3, 0]])
    l0s, l1s = segment_lines_with_each_other(line_0=l0, line_1=l1)

    assert all([isinstance(s, shapely.geometry.LineString) for s in l0s])
    assert all([isinstance(s, shapely.geometry.LineString) for s in l1s])
    assert len(l0s) == len(l1s)


def test_segmentation_bad_routes():
    df_a = pd.read_csv(TEST_DATA_DIR / "segmentation/bad_segment_route_a.csv")
    df_b = pd.read_csv(TEST_DATA_DIR / "segmentation/bad_segment_route_b.csv")
    line_a = shapely.geometry.LineString(list(zip(df_a.lon, df_a.lat)))
    line_b = shapely.geometry.LineString(list(zip(df_b.lon, df_b.lat)))

    segments_a, segments_b = segment_lines_with_each_other(line_a, line_b)

    assert len(segments_a) == len(segments_b)

    for s0, s1 in zip(segments_a, segments_b):
        assert s0.coords[0] == s1.coords[0]
        assert s0.coords[-1] == s1.coords[-1]

    for s0, s1 in zip(segments_a[:-1], segments_a[1:]):
        assert s0.coords[-1] == s1.coords[0]
