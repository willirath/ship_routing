from ship_routing.core.geodesics import (
    get_refinement_factor,
)

from shapely.geometry import LineString

import numpy as np

import pint


def test_refinement_factor():
    assert 10 == get_refinement_factor(original_dist=1000, new_dist=100)
    assert 34 == get_refinement_factor(original_dist=100, new_dist=3)
    assert 1 == get_refinement_factor(original_dist=1234, new_dist=1234)
