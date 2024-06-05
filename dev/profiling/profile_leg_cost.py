from ship_routing.core import Route, Leg, WayPoint
from ship_routing.data import load_currents
import numpy as np

import tqdm

currents = load_currents(
    data_file="data/currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_100W-020E_10N-65N.nc"
).compute()

route_ref = Route(
    way_points=(
        WayPoint(lon=-100, lat=20, time=np.datetime64("2001-01-01")),
        WayPoint(lon=20, lat=65, time=np.datetime64("2001-01-11")),
    )
)

print(str(currents))


def _calc_cost(r):
    return r.cost_through(currents)


route = route_ref
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 2)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 4)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 8)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 16)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 32)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 64)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)

route = route_ref.refine(route_ref.length_meters / 128)
print(len(route.legs))
for n in tqdm.trange(100):
    _ = _calc_cost(route)
