"""
Common geometry conversion and spatial indexing for map queries and relationship analysis.
"""

from bisect import bisect_right
from itertools import accumulate
from math import atan2, hypot, pi
from typing import Callable, Iterable, List, Mapping, NamedTuple, Optional

from google.protobuf.message import Message
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


def curve_to_line(curve: Message) -> LineString:
    """Read every line segment, omitting repeated segment endpoints."""
    points = []
    for segment in curve.segment:
        for point in segment.line_segment.point:
            coordinate = (point.x, point.y)
            if not points or points[-1] != coordinate:
                points.append(coordinate)
    if len(points) < 2:
        raise ValueError("A map curve must contain at least two distinct points")
    return LineString(points)


def polygon_to_shape(polygon: Message) -> Optional[Polygon]:
    """
    Read a junction's or crosswalk's polygon.

    :param Message polygon: An ``apollo.hdmap.Polygon`` message.
    :returns: The polygon, or None if it has no points.
    :rtype: Optional[Polygon]
    :raises ValueError: If the polygon has fewer than three points.
    """
    points = [(point.x, point.y) for point in polygon.point]
    if not points:
        return None
    if len(points) < 3:
        raise ValueError("A map polygon must contain at least three points")
    return Polygon(points)


def curves_to_lines(curves: Iterable[Message]) -> Optional[MultiLineString]:
    """
    Read several curves, such as a signal's or stop sign's stop lines, as one geometry.

    :param Iterable[Message] curves: ``apollo.hdmap.Curve`` messages.
    :returns: The lines, or None if there are no curves.
    :rtype: Optional[MultiLineString]
    :raises ValueError: If a curve has fewer than two distinct points.
    """
    lines = [curve_to_line(curve) for curve in curves]
    return MultiLineString(lines) if lines else None


class SpatialIndex(NamedTuple):
    """
    An STRtree over map object geometries and the ID of each, in tree order.
    """

    ids: List[str]
    tree: STRtree

    @classmethod
    def build(cls, kind: str, objects: Mapping[str, Message],
              geometry: Callable[[Message], Optional[BaseGeometry]]) -> "SpatialIndex":
        """
        Index the geometry of every object, in mapping order.

        :param str kind: Object kind, for error messages.
        :param Mapping[str, Message] objects: Objects by ID.
        :param geometry: Returns an object's geometry, or None to leave it out.
        :returns: The index of the objects that have a geometry.
        :rtype: SpatialIndex
        :raises ValueError: Naming the first object whose geometry is malformed.
        """
        ids, geometries = [], []
        for object_id, obj in objects.items():
            try:
                shape = geometry(obj)
            except ValueError as error:
                raise ValueError(f"invalid geometry for {kind} {object_id!r}: {error}") from error
            if shape is not None:
                ids.append(object_id)
                geometries.append(shape)
        return cls(ids, STRtree(geometries))

    def within(self, point: Point, distance: float) -> List[str]:
        """
        Return the IDs whose geometry lies within ``distance`` of ``point``.

        :param Point point: Query position in map coordinates.
        :param float distance: Search radius in metres; a polygon containing
            ``point`` is at distance 0.
        :returns: IDs nearest first, equally near ones by ID.
        :rtype: List[str]
        """
        geometries = self.tree.geometries
        found = sorted((geometries[index].distance(point), self.ids[index])
                       for index in self.tree.query(point, predicate="dwithin", distance=distance))
        return [object_id for _, object_id in found]

    def nearest(self, point: Point) -> Optional[str]:
        """
        Return the ID whose geometry is nearest to ``point``.

        :param Point point: Query position in map coordinates.
        :returns: The nearest ID, the earliest in index order among equally near
            ones (such as coincident lanes), or None for an empty index.
        :rtype: Optional[str]
        """
        indices = self.tree.query_nearest(point)
        return self.ids[min(indices)] if len(indices) else None


def normalize_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


class CenterlineProfile:
    """Cumulative s and segment headings of a centerline, for heading queries.

    Each segment's direction is anchored at its midpoint, and headings are
    interpolated between neighbouring midpoints. They are continuous and match
    a smooth curve's tangent at both segment midpoints and vertices. Before the
    first and after the last midpoint, the end segment's direction is used.
    Apollo's LaneInfo::Heading interpolates between vertices instead, which
    turns ahead of the curve, so results are not identical to Apollo's.
    """

    def __init__(self, line: LineString) -> None:
        coords = list(line.coords)
        pairs = list(zip(coords, coords[1:]))
        lengths = [hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in pairs]
        self.line = line
        self.cumulative_s = [0.0, *accumulate(lengths)]
        self.headings = [atan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in pairs]
        self.midpoint_s = [start + length / 2 for start, length in zip(self.cumulative_s, lengths)]

    def heading_at(self, s: float) -> float:
        index = bisect_right(self.midpoint_s, s)
        if index == 0:
            return self.headings[0]
        if index == len(self.headings):
            return self.headings[-1]
        start, end = self.midpoint_s[index - 1], self.midpoint_s[index]
        before = self.headings[index - 1]
        turn = normalize_angle(self.headings[index] - before)
        return normalize_angle(before + turn * (s - start) / (end - start))
