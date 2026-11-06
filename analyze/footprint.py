"""
Bodies of the ego and the obstacles, as map-frame rectangles.

Apollo's localization pose is the vehicle's reference point, and its vehicle
parameters give each edge's distance from that point. A perception obstacle
gives its centre, heading, length and width; our managers also send that
box's corners as its polygon points, so the box is the obstacle's body.
"""

from math import cos, sin
from typing import Tuple
from shapely.geometry import Polygon

from ..utils import ApolloVehicleConfig, rectangle_corners

__all__ = ["box", "ego_footprint", "front_point"]


def _rectangle(x: float, y: float, heading: float, front: float, back: float, left: float, right: float) -> Polygon:
    """
    Build a map-frame rectangle from the distances of its four edges from a reference point.

    ``utils.rectangle_corners`` runs counter-clockwise from the front-left corner; the ring here is clockwise
    (front-left, front-right, rear-right, rear-left), the order our managers publish an obstacle's corners in.

    :param float x: Reference point, map frame.
    :param float y: Reference point, map frame.
    :param float heading: Heading in radians.
    :param float front: Metres from the reference point to the front edge.
    :param float back: Metres to the rear edge.
    :param float left: Metres to the left edge.
    :param float right: Metres to the right edge.
    :returns: The rectangle.
    :rtype: Polygon
    """
    return Polygon(rectangle_corners(x, y, heading, front=front, back=back,
                                     left=left, right=right)[[0, 3, 2, 1]])


def ego_footprint(x: float, y: float, heading: float, vehicle: ApolloVehicleConfig) -> Polygon:
    """
    The ego's body around its localization pose.

    :param float x: Reference point, map frame.
    :param float y: Reference point, map frame.
    :param float heading: Heading in radians.
    :param ApolloVehicleConfig vehicle: Edge distances from the reference point.
    :returns: Its footprint, reaching further ahead of the reference point than behind it.
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
    :returns: Its ``(x, y)``, map frame.
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
    :returns: Its box, centred on ``(x, y)``.
    :rtype: Polygon
    """
    return _rectangle(x, y, heading, length / 2, length / 2, width / 2, width / 2)
