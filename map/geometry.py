"""
Common geometry conversion and spatial indexing for map queries and relationship analysis.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp
"""

from bisect import bisect_right
from itertools import accumulate
from math import atan2, hypot, pi
from google.protobuf.message import Message
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring
from typing import Callable, Generic, Iterable, List, Mapping, NamedTuple, Optional, Tuple, TypeVar

T = TypeVar("T")


def curve_to_line(curve: Message) -> LineString:
    """
    Read every line segment, omitting repeated segment endpoints.

    :param Message curve: An ``apollo.hdmap.Curve`` message.
    :returns: A line connecting all points, with consecutive duplicates removed.
    :rtype: LineString
    :raises ValueError: If fewer than two points remain after deduplication.
    """
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


def split_boundary(boundary: Message) -> List[Tuple[float, float, Message, LineString]]:
    """
    Split a lane boundary's curve where its marking changes.

    Each ``LaneBoundaryType`` piece runs from its ``s`` to the next piece's
    ``s``, or to the end of the curve; ``s`` is clamped to the curve's
    geometric length. A stretch before the first piece has no marking.

    :param Message boundary: An ``apollo.hdmap.LaneBoundary`` message.
    :returns: (start s, end s, LaneBoundaryType piece, line) for each piece of
        positive length; none if the boundary has no curve or no pieces.
    :rtype: List[Tuple[float, float, Message, LineString]]
    :raises ValueError: If the curve has fewer than two distinct points.
    """
    if not boundary.curve.segment or not boundary.boundary_type:
        return []
    line = curve_to_line(boundary.curve)
    pieces = sorted(boundary.boundary_type, key=lambda piece: piece.s)
    starts = [min(max(piece.s, 0.0), line.length) for piece in pieces]
    ends = starts[1:] + [line.length]
    # Most boundaries have one piece covering the whole curve; substring is slow.
    return [(start, end, piece, line if (start, end) == (0.0, line.length) else substring(line, start, end))
            for start, end, piece in zip(starts, ends, pieces) if end > start]


class SpatialIndex(NamedTuple, Generic[T]):
    """
    An STRtree over map geometries and the item each belongs to, in tree order.

    Items are object IDs, or records such as ``LaneMarking``; they must be
    orderable, since equally near ones are returned in item order.
    """
    items: List[T]
    """Object IDs or records, in the same order as the tree's geometries."""
    tree: STRtree
    """Spatial index whose geometry indices correspond to positions in items."""

    @classmethod
    def build(cls, kind: str, objects: Mapping[str, Message],
              geometry: Callable[[Message], Optional[BaseGeometry]]) -> "SpatialIndex[str]":
        """
        Index the geometry of every object, in mapping order.

        :param str kind: Object kind, for error messages.
        :param Mapping[str, Message] objects: Objects by ID.
        :param Callable[[Message], Optional[BaseGeometry]] geometry: Returns an object's geometry, or None to leave it out.
        :returns: The index of the IDs of the objects that have a geometry.
        :rtype: SpatialIndex[str]
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

    def within(self, geometry: BaseGeometry, distance: float) -> List[T]:
        """
        Return the items whose geometry lies within ``distance`` of ``geometry``.

        :param BaseGeometry geometry: Query geometry in map coordinates, such as
            a point or a vehicle footprint.
        :param float distance: Search radius in metres; 0 means intersecting,
            and a polygon containing the query geometry is at distance 0.
        :returns: Items nearest first, equally near ones in item order.
        :rtype: List[T]
        """
        geometries = self.tree.geometries
        found = sorted((geometries[index].distance(geometry), self.items[index])
                       for index in self.tree.query(geometry, predicate="dwithin", distance=distance))
        return [item for _, item in found]

    def nearest(self, point: Point) -> Optional[T]:
        """
        Return the item whose geometry is nearest to ``point``.

        :param Point point: Query position in map coordinates.
        :returns: The nearest item, the earliest in index order among equally
            near ones (such as coincident lanes), or None for an empty index.
        :rtype: Optional[T]
        """
        indices = self.tree.query_nearest(point)
        return self.items[min(indices)] if len(indices) else None


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi).

    :param float angle: Angle in radians.
    :returns: The equivalent angle in radians within [-pi, pi).
    :rtype: float
    """
    return (angle + pi) % (2 * pi) - pi


class CenterlineProfile:
    """
    Cumulative s and segment headings of a centerline, for heading queries.

    Each segment's direction is anchored at its midpoint, and headings are
    interpolated between neighbouring midpoints. They are continuous and match
    a smooth curve's tangent at both segment midpoints and vertices. Before the
    first and after the last midpoint, the end segment's direction is used.
    Apollo's LaneInfo::Heading interpolates between vertices instead, which
    turns ahead of the curve, so results are not identical to Apollo's.

    :param LineString line: The lane centerline used to compute cumulative distances
                            and segment headings.
    """
    line: LineString
    cumulative_s: List[float]
    headings: List[float]
    midpoint_s: List[float]

    def __init__(self, line: LineString) -> None:
        """
        Constructor

        :param LineString line: The lane centerline used to compute cumulative distances
                                and segment headings.
        """
        coords = list(line.coords)
        pairs = list(zip(coords, coords[1:]))
        lengths = [hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in pairs]
        self.line = line
        self.cumulative_s = [0.0, *accumulate(lengths)]
        self.headings = [atan2(y2 - y1, x2 - x1) for (x1, y1), (x2, y2) in pairs]
        self.midpoint_s = [start + length / 2 for start, length in zip(self.cumulative_s, lengths)]

    def heading_at(self, s: float) -> float:
        """
        Get the heading at s, interpolated between neighbouring segment midpoints.

        Between two midpoints the heading turns linearly with s, the shorter
        way round; before the first and after the last midpoint it is the end
        segment's direction, so any s, even outside the line, has a heading.
        The line must have no zero-length segments, as ``curve_to_line``
        guarantees.

        :param float s: Metres along the line from its first point.
        :returns: Heading in radians within ``[-pi, pi)``.
        :rtype: float
        """
        index = bisect_right(self.midpoint_s, s)
        if index == 0:
            return self.headings[0]
        if index == len(self.headings):
            return self.headings[-1]
        start, end = self.midpoint_s[index - 1], self.midpoint_s[index]
        before = self.headings[index - 1]
        turn = normalize_angle(self.headings[index] - before)
        return normalize_angle(before + turn * (s - start) / (end - start))
