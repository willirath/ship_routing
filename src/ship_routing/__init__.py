"""
Ship routing optimization package.

Three-layer architecture:
- core: Routes, waypoints, legs, data structures, cost calculation
- algorithms: Gradient descent, stochastic optimization, crossover
- app: High-level route optimization application

Examples
--------
>>> from ship_routing.core import Route, WayPoint, Leg
>>> from ship_routing.algorithms import gradient_descent, stochastic_search
>>> from ship_routing.app import RoutingApp, RoutingConfig
"""

__version__ = "2025dev"
