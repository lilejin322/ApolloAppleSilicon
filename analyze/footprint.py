"""
Bodies of the ego and the obstacles, as map-frame rectangles.

Apollo's localization pose is the vehicle's reference point, and its vehicle
parameters give each edge's distance from that point. A perception obstacle
gives its centre, heading, length and width; our managers send no polygon
points, so the box is the obstacle's body.
"""

from math import cos, sin
from typing import Tuple

from shapely.geometry import Polygon

from ..utils import ApolloVehicleConfig

__all__ = ["box", "ego_footprint", "front_point"]


def _rectangle(x: float, y: float, heading: float, front: float, back: float, left: float, right: float) -> Polygon:
    c, s = cos(heading), sin(heading)
    corners = [(front, left), (front, -right), (-back, -right), (-back, left)]
    return Polygon([(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in corners])


def ego_footprint(x: float, y: float, heading: float, vehicle: ApolloVehicleConfig) -> Polygon:
    """
    The ego's body around its localization pose.

    :param float x: Reference point, map frame.
    :param float y: Reference point, map frame.
    :param float heading: Heading in radians.
    :param ApolloVehicleConfig vehicle: Edge distances from the reference point.
    :rtype: Polygon
    """
    return _rectangle(x, y, heading, vehicle.front_edge_to_center, vehicle.back_edge_to_center,
                      vehicle.left_edge_to_center, vehicle.right_edge_to_center)


def front_point(x: float, y: float, heading: float, vehicle: ApolloVehicleConfig) -> Tuple[float, float]:
    """
    The middle of the ego's front edge.

    :param float x: Reference point, map frame.
    :param float y: Reference point, map frame.
    :param float heading: Heading in radians.
    :param ApolloVehicleConfig vehicle: ``front_edge_to_center`` is used.
    :rtype: Tuple[float, float]
    """
    return x + cos(heading) * vehicle.front_edge_to_center, y + sin(heading) * vehicle.front_edge_to_center


def box(x: float, y: float, heading: float, length: float, width: float) -> Polygon:
    """
    A perception obstacle's body.

    :param float x: Centre, map frame.
    :param float y: Centre, map frame.
    :param float heading: Heading in radians.
    :param float length: Along the heading.
    :param float width: Across it.
    :rtype: Polygon
    """
    return _rectangle(x, y, heading, length / 2, length / 2, width / 2, width / 2)
