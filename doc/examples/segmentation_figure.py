#!/usr/bin/env python
"""Generate segmentation figure showing two crossing routes.

Creates a figure illustrating route crossover segmentation:
- Two routes sharing start and end points
- Multiple intersection points marked with circles
- A shared segment where routes run identically
"""

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

# Shared start and end points
start = (0, 2)
end = (12, 2)

# Route A: more waypoints for smoother appearance
# Includes a segment (around x=7.5-8.8) that will be identical to route B
ctrl_a = [
    start,
    (0.8, 2.5),
    (1.5, 2.8),
    (2.2, 2.6),
    (3.0, 2.0),
    (3.8, 1.6),
    (4.5, 1.8),
    (5.2, 2.4),
    (6.0, 2.8),
    (6.8, 2.6),
    (7.5, 2.3),  # Start of shared segment
    (8.8, 2.0),  # End of shared segment
    (9.5, 1.6),
    (10.2, 1.4),
    (11.0, 1.7),
    end,
]

# Route B: different path but shares segment at x=7.5-8.8
ctrl_b = [
    start,
    (0.7, 1.5),
    (1.4, 1.2),
    (2.1, 1.4),
    (2.8, 1.8),
    (3.6, 2.4),
    (4.4, 2.6),
    (5.1, 2.2),
    (5.8, 1.8),
    (6.5, 2.0),
    (7.5, 2.3),  # Start of shared segment (same as route A)
    (8.8, 2.0),  # End of shared segment (same as route A)
    (9.6, 2.4),
    (10.4, 2.6),
    (11.2, 2.3),
    end,
]


def interpolate_path(control_points, n=300):
    """Interpolate smooth path through control points."""
    ctrl = np.array(control_points)
    t = np.linspace(0, 1, len(ctrl))
    t_new = np.linspace(0, 1, n)
    x = np.interp(t_new, t, ctrl[:, 0])
    y = np.interp(t_new, t, ctrl[:, 1])
    return x, y


def main():
    x_a, y_a = interpolate_path(ctrl_a)
    x_b, y_b = interpolate_path(ctrl_b)

    # Create LineStrings
    route_a = LineString(zip(x_a, y_a))
    route_b = LineString(zip(x_b, y_b))

    # Find intersections
    intersections = route_a.intersection(route_b)

    # Extract intersection points and shared segment endpoints
    intersection_points = []

    if intersections.geom_type == "MultiPoint":
        intersection_points = list(intersections.geoms)
    elif intersections.geom_type == "Point":
        intersection_points = [intersections]
    elif intersections.geom_type == "GeometryCollection":
        from shapely.geometry import Point
        # Get crossing points
        intersection_points = [
            g for g in intersections.geoms if g.geom_type == "Point"
        ]
        # Get shared segment endpoints (from LineStrings)
        lines = [g for g in intersections.geoms if g.geom_type == "LineString"]
        if lines:
            # Start of first line, end of last line
            first_coord = list(lines[0].coords)[0]
            last_coord = list(lines[-1].coords)[-1]
            intersection_points.append(Point(first_coord))
            intersection_points.append(Point(last_coord))

    intersection_points = sorted(intersection_points, key=lambda p: p.x)

    # Filter out start/end points of journey, keep only mid-route intersections
    intersection_points = [
        p for p in intersection_points if p.x > start[0] + 0.1 and p.x < end[0] - 0.1
    ]

    # Waypoints: use control points
    waypoints_a = ctrl_a
    waypoints_b = ctrl_b

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw routes (let matplotlib choose colors)
    (line_a,) = ax.plot(x_a, y_a, "-", linewidth=3)
    (line_b,) = ax.plot(x_b, y_b, "-", linewidth=3)

    # Draw waypoints as dots (same color as route)
    for wx, wy in waypoints_a:
        ax.plot(wx, wy, "o", color=line_a.get_color(), markersize=7)
    for wx, wy in waypoints_b:
        ax.plot(wx, wy, "o", color=line_b.get_color(), markersize=7)

    # Draw intersection points (black circles, no labels)
    for pt in intersection_points:
        ax.plot(
            pt.x, pt.y, "ko", markersize=14, markerfacecolor="none", markeredgewidth=2.5
        )

    # Draw circles at start and end points
    for pt in [start, end]:
        ax.plot(
            pt[0], pt[1], "ko", markersize=14, markerfacecolor="none", markeredgewidth=2.5
        )

    ax.set_xlim(-0.8, 12.8)
    ax.set_ylim(0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    fig = main()
    out_dir = Path(__file__).parent
    fig.savefig(out_dir / "segmentation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "segmentation.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir}/segmentation.{{pdf,png}}")
