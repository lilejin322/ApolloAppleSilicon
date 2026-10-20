"""
Where pedestrians walk: sidewalks, crossings and refuges derived from a map.

No bundled map has ``SIDEWALK`` lanes, so the network is derived from lane
geometry (``docs/pedestrian-agents.md``). The road surface is the union of
every lane polygon. Holes in it inside junctions are modelling gaps unless a
crosswalk lands on them (traffic islands); other holes are blocks, medians
and islands. Holes narrower than ``REFUGE_WIDTH`` are refuges, where crossings
may pause. Sidewalks are the surface's boundary rings offset outward by
``SIDEWALK_OFFSET``: measured crosswalk ends lie about 1.5 m outside the curb.
Crosswalk polygons joined end to end form one crossing from sidewalk to
sidewalk.

A walker leaves the sidewalk on an *excursion*: a crossing of one of the
``ExcursionKind``s, or a stretch walked in the road along the curb.
``WalkNetwork.check_excursion`` decides whether an entry and exit point make
an excursion of a kind, and ``WalkNetwork.propose`` draws one.

A ``WalkRoute`` is a walk: a start on a sidewalk, up to ``MAX_EXCURSIONS``
excursions joined along the sidewalk, and an end. It plays the part of
``routing.LaneRoute`` for pedestrians: ``WalkNetwork`` validates walks, splits
them into named elements (``WalkRoute.lanes``) and measures them, as
``RouteSpace`` does for lane routes.
"""

import weakref
from dataclasses import dataclass, replace
from enum import Enum
from math import atan2, cos, hypot, isfinite, pi, radians, sin
from random import Random
from typing import Iterator, NamedTuple, Sequence, Self
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union

from ..map import MapService
from ..map.geometry import curve_to_line, normalize_angle, polygon_to_shape
from ..message.descriptor import DescriptorBundle
from .routing import InvalidRoute, junction_lanes, offset_line
from typing import Dict, FrozenSet, List, Mapping, Tuple
from google.protobuf.message import Message

SIDEWALK_OFFSET = 1.5
"""Metres from the curb (the road surface's edge) to the line walkers follow
on the sidewalk. Crosswalk axis ends lie a median 1.3-1.9 m outside the curb on
the bundled maps."""
SURFACE_CLOSING = 0.3
"""Gaps narrower than twice this between lane polygons are closed."""
ISLAND_TOUCH = 1.0
"""Metres within which a crosswalk makes a hole inside a junction a traffic island."""
REFUGE_WIDTH = 6.0
"""Holes narrower than this (medians, small islands) are refuges, not sidewalks."""
MIN_BLOCK_AREA = 400.0
"""Square metres below which a sidewalk ring no crossing reaches is dropped
(roundabout centre islands, gaps between lane pieces)."""
CHAIN_GAP = 6.0
"""Metres between the facing ends of crosswalk polygons that form one crossing."""
CHAIN_ANGLE = radians(20)
"""Largest angle between crosswalk axes that form one crossing."""
CHAIN_OFFSET = 1.5
"""Largest sideways offset between crosswalk axes that form one crossing."""
CROSSWALK_EXTENSION = 5.0
"""Metres a crossing's end is extended along its axis to reach a sidewalk."""
LANDING_INWARD = 6.0
"""Metres back along the axis from a crossing's end searched for its sidewalk: some
crosswalks are drawn up to 6 m past the curb (sunnyvale_loop)."""
LANDING_SNAP = 3.0
"""Metres from a crossing's end to the sidewalk it lands on where its axis misses the sidewalk."""
JUNCTION_REACH = 10.0
"""Metres within which a crossing belongs to a junction."""
BODY_MARGIN = 0.3
"""Metres a walker's centre keeps inside a crosswalk's edge."""
CROSSWALK_TOLERANCE = 0.5
"""Metres a crosswalk crossing's road part may stray outside the polygon."""
LANDING_REACH = 10.0
"""Metres along the sidewalk an entry or exit may lie from a crossing's landing."""
NEAR_RANGE = (2.0, 20.0)
"""Metres outside a crosswalk's edge that a crossing beside it keeps."""
NEAR_ANGLE = radians(30)
"""Largest angle between a crossing beside a crosswalk and its axis."""
MID_BLOCK_CROSSWALK_CLEARANCE = 20.0
MID_BLOCK_JUNCTION_CLEARANCE = 15.0
MID_BLOCK_ANGLE = radians(45)
"""Largest angle between a mid-block crossing and the road's normal."""
MID_BLOCK_ANGLE_SIGMA = radians(15)
"""Standard deviation of drawn mid-block crossing angles."""
DIAGONAL_SHARE = 0.4
"""Share of a diagonal crossing's road part inside its junction."""
DIAGONAL_REACH = 10.0
"""Metres within which a diagonal crossing's ends lie from its junction."""
DIAGONAL_ANGLE = radians(15)
"""Smallest angle between a diagonal crossing and its junction's crosswalk axes: at a
skewed junction (borregas_ave) the corner-to-corner line is only about 25° off them."""
DIAGONAL_MARKED = 0.5
"""Largest share of a diagonal crossing's road part inside its junction's crosswalks."""
ROADSIDE_RANGE = (5.0, 50.0)
"""Metres along the sidewalk between a roadside walk's entry and exit."""
ROADSIDE_OFFSET = 0.75
"""Metres inside the curb a walker in the road keeps (body centre)."""
ROAD_SHARE = 0.5
"""Share of a crossing that lies on the road or refuges."""
BLOCK_CLEARANCE = 1.8
"""Metres from the road within which every crossing stays: it never cuts through a block."""
MIN_CROSSING = 3.0
"""Shortest crossing in metres."""
EDGE_GATE_LENGTH = 4.0
"""Metres a lane that ends at the edge of the map (no successor, or no
predecessor, at the road surface's boundary) is extended past its end. The
sidewalk ring goes round such a road end across the carriageway, where the road
in fact carries on beyond the map; walks may not cross these gates nor start or
end within ``EDGE_GATE_CLEARANCE`` of one."""
EDGE_GATE_CLEARANCE = 1.0
EDGE_HULL_REACH = 3.0
"""Metres from the road surface's convex hull within which a lane end counts as the map's edge."""
MAX_EXCURSIONS = 2
"""Most excursions in one walk: two crossings at a corner, or a crossing and a
walk in the road. Three in a row are rare."""
MIN_WALK = 10.0
"""Shortest random walk without excursions, from start to end, in metres."""
MIN_TURN_BACK = 1.0
"""Metres from the entry and from the exit within which a walker never turns back."""
TURN_BACK_SHARE = 0.6
"""Largest share of an excursion a walker covers before turning back, when drawn."""
_STRAY = 0.5
"""Metres of a line tolerated outside an area it must keep to."""
_CACHE_SIZE = 4096
_NETWORKS: "weakref.WeakKeyDictionary[MapService, WalkNetwork]" = weakref.WeakKeyDictionary()


class ExcursionKind(Enum):
    """
    How a walker leaves the sidewalk.
    """
    CROSSWALK = "CROSSWALK"
    NEAR_CROSSWALK = "NEAR_CROSSWALK"
    MID_BLOCK = "MID_BLOCK"
    DIAGONAL = "DIAGONAL"
    ROADSIDE = "ROADSIDE"


@dataclass(frozen=True)
class CurbPoint:
    """
    A map position on a sidewalk.

    :param str sidewalk: A sidewalk ring of the walk network.
    :param float s: Metres along the ring from its start, in ``[0, ring length)``.
    """
    sidewalk: str
    """A sidewalk ring of the walk network."""
    s: float
    """Metres along the ring from its start, in ``[0, ring length)``."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If ``sidewalk`` is empty or ``s`` is not finite and nonnegative.
        """
        if not self.sidewalk:
            raise ValueError("a curb point needs a sidewalk")
        if not (isfinite(self.s) and self.s >= 0):
            raise ValueError("curb point s must be finite and nonnegative")

    def to_dict(self) -> Dict:
        """
        Serialize the sidewalk ID and position to a JSON-compatible mapping.

        :returns: JSON-compatible mapping with ``sidewalk`` and ``s``.
        :rtype: Dict
        """
        return {"sidewalk": self.sidewalk, "s": self.s}

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Reconstruct a sidewalk position from its serialized mapping.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: Reconstructed curb point.
        :rtype: CurbPoint
        :raises KeyError: If ``sidewalk`` or ``s`` is absent.
        :raises ValueError: If the reconstructed curb point is invalid.
        """
        return cls(data["sidewalk"], data["s"])


@dataclass(frozen=True)
class Crossing:
    """
    A marked crossing: crosswalk polygons joined end to end, from sidewalk to sidewalk.

    :param str crossing_id: Its crosswalk IDs joined by ``+``.
    :param Tuple[str, ...] crosswalks: Its crosswalk IDs, sorted.
    :param Polygon polygon: The union of its crosswalk polygons.
    :param LineString axis: From the first landing to the second, on the sidewalks.
    :param float width: Metres, the narrowest of its polygons.
    :param Tuple landings: Where it meets the sidewalks, at the axis ends.
    :param str | None junction: The junction it belongs to, if any.
    """
    crossing_id: str
    """Its crosswalk IDs joined by ``+``."""
    crosswalks: Tuple[str, ...]
    """Its crosswalk IDs, sorted."""
    polygon: Polygon
    """The union of its crosswalk polygons."""
    axis: LineString
    """From the first landing to the second, on the sidewalks."""
    width: float
    """Metres, the narrowest of its polygons."""
    landings: Tuple[CurbPoint, CurbPoint]
    """Where it meets the sidewalks, at the axis ends."""
    junction: str | None
    """The junction it belongs to, if any."""

    @property
    def direction(self) -> Tuple[float, float]:
        """
        Unit vector along the axis, from the first landing to the second.

        :returns: Unit direction vector from the first landing to the second.
        :rtype: Tuple[float, float]
        """
        (x0, y0), (x1, y1) = self.axis.coords[0], self.axis.coords[-1]
        length = hypot(x1 - x0, y1 - y0)
        return (x1 - x0) / length, (y1 - y0) / length

    def lateral(self, point: Point) -> float:
        """
        Signed metres of ``point`` left of the axis.

        :param Point point: Map point whose lateral offset is measured.
        :returns: Signed lateral offset left of the crossing axis, in metres.
        :rtype: float
        """
        (x0, y0), (ux, uy) = self.axis.coords[0], self.direction
        return ux * (point.y - y0) - uy * (point.x - x0)

    def along(self, point: Point) -> float:
        """
        Metres of ``point``'s projection along the axis from the first landing.

        :param Point point: Map point whose longitudinal projection is measured.
        :returns: Projection distance along the crossing axis from the first landing, in metres.
        :rtype: float
        """
        (x0, y0), (ux, uy) = self.axis.coords[0], self.direction
        return ux * (point.x - x0) + uy * (point.y - y0)


@dataclass(frozen=True)
class Excursion:
    """
    One trip off the sidewalk.

    :param ExcursionKind kind: How the walker leaves the sidewalk; the network checks it.
    :param CurbPoint entry: Where the walker leaves the sidewalk.
    :param CurbPoint exit: Where it returns; on the same sidewalk for ``ROADSIDE``.
    :param float | None turn_back: Metres from ``entry`` along the excursion where
        the walker stops, turns round and walks back to ``entry``; None to carry on.
    """
    kind: ExcursionKind
    """How the walker leaves the sidewalk; the network checks it."""
    entry: CurbPoint
    """Where the walker leaves the sidewalk."""
    exit: CurbPoint
    """Where it returns; on the same sidewalk for ``ROADSIDE``."""
    turn_back: float | None = None
    """Metres from ``entry`` along the excursion where the walker stops, turns round and walks back to ``entry``; None to carry on."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If ``turn_back`` is not ``None`` and is not finite and positive.
        """
        if self.turn_back is not None and not (isfinite(self.turn_back) and self.turn_back > 0):
            raise ValueError("turn_back must be finite and positive, or None")

    @property
    def arrival(self) -> CurbPoint:
        """
        Where the walker is back on a sidewalk: the entry after a turn back, else the exit.

        :returns: The entry after a turn back, otherwise the excursion exit.
        :rtype: CurbPoint
        """
        return self.entry if self.turn_back is not None else self.exit

    def to_dict(self) -> Dict:
        """
        Serialize an excursion kind, endpoints, and optional turning point.

        :returns: JSON-compatible mapping of the excursion kind, endpoints, and optional turn-back distance.
        :rtype: Dict
        """
        return {"kind": self.kind.name, "entry": self.entry.to_dict(), "exit": self.exit.to_dict(),
                "turn_back": self.turn_back}

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Reconstruct an excursion from its serialized mapping.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: Reconstructed excursion.
        :rtype: Excursion
        :raises KeyError: If a required field is absent or ``kind`` is not a known excursion kind.
        :raises ValueError: If a nested curb point or the reconstructed excursion is invalid.
        """
        return cls(ExcursionKind[data["kind"]], CurbPoint.from_dict(data["entry"]), CurbPoint.from_dict(data["exit"]),
                   data.get("turn_back"))


@dataclass(frozen=True)
class WalkRoute:
    """
    A walk: from ``start`` along the sidewalk to each excursion in turn, and
    from the last one's arrival to ``end``. Between two excursions the walker
    keeps to one sidewalk the shorter way round.

    :param CurbPoint start: Where the walk starts.
    :param Tuple[Excursion, ...] excursions: Up to ``MAX_EXCURSIONS``; only the last may turn back.
    :param CurbPoint end: Where the walk ends, on the last arrival's sidewalk.
    """
    start: CurbPoint
    """Where the walk starts."""
    excursions: Tuple[Excursion, ...]
    """Up to ``MAX_EXCURSIONS``; only the last may turn back."""
    end: CurbPoint
    """Where the walk ends, on the last arrival's sidewalk."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If there are too many excursions or a non-final excursion turns back.
        """
        if len(self.excursions) > MAX_EXCURSIONS:
            raise ValueError(f"a walk has at most {MAX_EXCURSIONS} excursions")
        if any(excursion.turn_back is not None for excursion in self.excursions[:-1]):
            raise ValueError("only the last excursion may turn back")

    @property
    def lanes(self) -> Tuple[str, ...]:
        """
        The walk's elements in walking order: ``w<i>`` the sidewalk before
        excursion ``i`` (``w<n>`` after the last), ``x<i>`` the excursion out,
        ``x<i>b`` its way back after a turn back. Speed points name them.

        :returns: Element IDs in walking order, including outbound and return excursion elements.
        :rtype: Tuple[str, ...]
        """
        names = []
        for index, excursion in enumerate(self.excursions):
            names += [f"w{index}", f"x{index}"] + ([f"x{index}b"] if excursion.turn_back is not None else [])
        return tuple(names + [f"w{len(self.excursions)}"])

    def to_dict(self) -> Dict:
        """
        Serialize the walk endpoints and ordered excursions.

        :returns: JSON-compatible mapping of the walk endpoints and excursions.
        :rtype: Dict
        """
        return {"start": self.start.to_dict(), "excursions": [excursion.to_dict() for excursion in self.excursions],
                "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Reconstruct a walk from its serialized mapping.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: Reconstructed walk.
        :rtype: WalkRoute
        :raises KeyError: If a required serialized field is absent or an excursion kind is unknown.
        :raises ValueError: If a nested value or the reconstructed walk is invalid.
        """
        return cls(CurbPoint.from_dict(data["start"]), tuple(Excursion.from_dict(item) for item in data["excursions"]),
                   CurbPoint.from_dict(data["end"]))


class WalkElement(NamedTuple):
    """
    One element of a walk: its name (``WalkRoute.lanes``), raw vertices and length.

    :param str name: Walk-element name.
    :param Tuple[Tuple[float, float], ...] coords: Ordered map-coordinate vertices.
    :param float length: Length in metres.
    """
    name: str
    """Walk-element name."""
    coords: Tuple[Tuple[float, float], ...]
    """Ordered map-coordinate vertices."""
    length: float
    """Length in metres."""


class WalkFrame:
    """
    Walked distance along a walk's elements, unsmoothed, and conversions
    between (element, s) and that distance, as ``replay.RouteFrame`` does for lanes.

    :param Tuple[WalkElement, ...] elements: Walk elements in walking order.
    """
    elements: Tuple[WalkElement, ...]
    """Walk elements in walking order."""
    index: Dict[str, int]
    """Walk-element name to element-index mapping."""
    offsets: List[float]
    """Walked-distance offset of each element, in metres."""
    total: float
    """Total walked distance represented by the frame, in metres."""

    def __init__(self, elements: Sequence[WalkElement]) -> None:
        """
        Build element lookup and cumulative walked distances.

        :param Sequence[WalkElement] elements: Ordered named walk elements with their geometry and lengths.
        """
        self.elements = tuple(elements)
        self.index = {element.name: index for index, element in enumerate(self.elements)}
        self.offsets = list(np.r_[0.0, np.cumsum([element.length for element in self.elements])][:-1])
        self.total = float(sum(element.length for element in self.elements))

    def window(self, name: str) -> Tuple[float, float]:
        """
        The stretch [0, length] of an element.

        :param str name: Walk-element name.
        :returns: Local element interval ``(0, length)`` in metres.
        :rtype: Tuple[float, float]
        """
        return 0.0, self.elements[self.index[name]].length

    def distance(self, name: str, s: float) -> float:
        """
        Walked distance to ``s`` on an element, clamped to it.

        :param str name: Walk-element name.
        :param float s: Position along the element, in metres.
        :returns: Walked distance to the local element position, clamped to the element, in metres.
        :rtype: float
        """
        index = self.index[name]
        return self.offsets[index] + min(max(s, 0.0), self.elements[index].length)

    def at(self, distance: float) -> Tuple[str, float]:
        """
        The (element, s) at a walked distance; on the later element at a boundary.

        :param float distance: Walked distance from the route start, in metres.
        :returns: Walk-element name and local position at the requested walked distance.
        :rtype: Tuple[str, float]
        """
        distance = min(max(distance, 0.0), self.total)
        last = max((index for index, element in enumerate(self.elements) if element.length > 0), default=0)
        for index, element in enumerate(self.elements):
            if element.length > 0 and (distance < self.offsets[index] + element.length or index == last):
                return element.name, min(max(distance - self.offsets[index], 0.0), element.length)
        return self.elements[-1].name, 0.0

    def check_windows(self, profile: Sequence) -> None:
        """
        Check that every speed point lies on its element.

        :param Sequence profile: Ordered speed points anchored to walk elements.
        :raises InvalidRoute: For the first point outside.
        """
        for point in profile:
            low, high = self.window(point.lane_id)
            if not low - 1e-6 <= point.s <= high + 1e-6:
                raise InvalidRoute(f"speed point s={point.s:.3f} is outside element {point.lane_id!r} "
                                   f"[{low:.3f}, {high:.3f}]")


def _undirected_angle(a: float, b: float) -> float:
    """
    Angle between two directions as lines, in [0, pi / 2].

    :param float a: First direction, in radians.
    :param float b: Second direction, in radians.
    :returns: Smallest undirected angular difference, in radians within [0, pi / 2].
    :rtype: float
    """
    turn = abs(normalize_angle(a - b))
    return min(turn, pi - turn)


def _heading(line: LineString) -> float:
    """
    Compute the heading from the first to the last vertex of a line.

    :param LineString line: Line geometry in map metres.
    :returns: Heading of the line endpoint chord, in radians.
    :rtype: float
    """
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return atan2(y1 - y0, x1 - x0)


def _parts(geometry: BaseGeometry) -> List:
    """
    Return nonempty geometry components as a list.

    :param BaseGeometry geometry: Geometry to query or decompose.
    :returns: The nonempty geometry components, or an empty list for an empty geometry.
    :rtype: List
    """
    return [] if geometry.is_empty else list(getattr(geometry, "geoms", [geometry]))


def _polygons(table: Mapping[str, Message]) -> Dict[str, Polygon]:
    """
    The valid polygons of a junction or crosswalk table, by ID; objects without a polygon are left out.

    :param Mapping[str, Message] table: Map objects keyed by ID, each providing a polygon.
    :returns: Valid polygonal geometries keyed by map object ID.
    :rtype: Dict[str, Polygon]
    """
    shapes = {}
    for key, item in table.items():
        shape = polygon_to_shape(item.polygon)
        if shape is not None:
            shapes[key] = shape if shape.is_valid else shape.buffer(0)
    return shapes


def _lane_polygons(service: MapService) -> List[Polygon]:
    """
    Lane areas between their boundary curves; lanes without both curves are skipped.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :returns: Lane areas with usable boundary geometry.
    :rtype: List[Polygon]
    """
    polygons = []
    for lane in service.lane_table.values():
        try:
            left, right = curve_to_line(lane.left_boundary.curve), curve_to_line(lane.right_boundary.curve)
        except ValueError:
            continue
        polygon = Polygon(list(left.coords) + list(right.coords)[::-1])
        polygon = polygon if polygon.is_valid else polygon.buffer(0)
        polygons.extend(part for part in _parts(polygon) if part.area > 0)
    return polygons


def _min_rect_axis(polygon: Polygon) -> Tuple[LineString, float]:
    """
    The long axis of the minimum rotated rectangle, through its middle, and the short side.

    :param Polygon polygon: Polygon whose geometry is inspected.
    :returns: The long rectangle axis and its short-side width, in metres.
    :rtype: Tuple[LineString, float]
    """
    corners = list(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    first, second = hypot(*np.subtract(corners[1], corners[0])), hypot(*np.subtract(corners[2], corners[1]))

    def middle(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
        """
        Return the coordinate-wise midpoint of two points.

        :param Sequence[float] a: First coordinate.
        :param Sequence[float] b: Second coordinate.
        :returns: Coordinate-wise midpoint of ``a`` and ``b``.
        :rtype: Tuple[float, float]
        """
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    if first >= second:
        return LineString([middle(corners[1], corners[2]), middle(corners[3], corners[0])]), second
    return LineString([middle(corners[0], corners[1]), middle(corners[2], corners[3])]), first


def _ring_line(coords: List) -> LineString:
    """
    A closed ring as a counter-clockwise line starting at its lowest vertex.

    :param List coords: Ordered map coordinates in metres.
    :returns: A closed ring as a counter-clockwise line starting at its lowest vertex.
    :rtype: LineString
    """
    points = [tuple(round(value, 6) for value in point[:2]) for point in coords[:-1]]
    if shapely.LinearRing(points).is_ccw is False:
        points.reverse()
    start = min(range(len(points)), key=lambda index: points[index])
    points = points[start:] + points[:start]
    return LineString(points + [points[0]])


class WalkNetwork:
    """
    The sidewalks, refuges and crossings of one loaded map; read-only after construction.

    Build it with ``walk_network(service)``, which caches one per map.

    :param MapService service: Loaded map service from which the network was derived.
    """
    service: MapService
    """Loaded map service from which the network was derived."""
    sidewalks: Dict[str, LineString]
    """Closed sidewalk rings by ID, counter-clockwise."""
    surface: BaseGeometry
    """Derived road surface used to place and validate crossings."""
    refuges: Tuple[Polygon, ...]
    """Medians and islands narrower than ``REFUGE_WIDTH``."""
    crossings: Dict[str, Crossing]
    """Marked crossings by ID."""
    junctions: Dict[str, Polygon]
    """Junction polygons by ID."""
    unconnected_crossings: int
    """Crosswalk chains without a sidewalk at both ends."""
    edge_gates: Tuple[LineString, ...]
    """Map-edge road-end gates that walks may not cross."""
    _surface_pieces: List[Polygon]
    """Lane-area pieces used by the road-surface spatial index."""
    _surface_tree: STRtree
    """Spatial index over ``_surface_pieces``."""
    _road_pieces: List[Polygon]
    """Road, refuge, and crosswalk pieces that crossings may traverse."""
    _road_tree: STRtree
    """Spatial index over ``_road_pieces``."""
    _junction_ids: List[str]
    """Junction IDs in stable index order."""
    _junction_tree: STRtree
    """Spatial index over junction polygons."""
    _sidewalk_ids: List[str]
    """Sidewalk IDs in stable index order."""
    _sidewalk_tree: STRtree
    """Spatial index over sidewalk rings."""
    _crossing_ids: List[str]
    """Crossing IDs in stable index order."""
    _crossing_tree: STRtree
    """Spatial index over crossing polygons."""
    _road_lanes: List[str]
    """Non-junction city-driving lane IDs used for random mid-block proposals."""
    _road_lane_lengths: np.ndarray
    """Cumulative lengths of ``_road_lanes``."""
    _ring_lengths: np.ndarray
    """Cumulative lengths of ``_sidewalk_ids``."""
    _diagonals: Dict[str, List]
    """Cached feasible diagonal landing pairs by junction ID."""
    _checks: Dict
    """Bounded cache of excursion and walk-element validation results."""
    _gate_tree: STRtree
    """Spatial index over ``edge_gates``."""

    def __init__(self, service: MapService) -> None:
        """
        Derive sidewalks, crossings, refuges, and spatial indexes from a loaded map.

        :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
        :raises RuntimeError: If the map was loaded without its routing graph or lane index.
        """
        if service.routing_graph is None or service.lane_index is None:
            raise RuntimeError("WalkNetwork needs a map loaded with load_only=False")
        self.service = service
        lanes = _lane_polygons(service)
        surface = unary_union(lanes).buffer(SURFACE_CLOSING).buffer(-SURFACE_CLOSING)
        self.junctions = _polygons(service.junction_table)
        crosswalks = _polygons(service.crosswalk_table)
        junction_area = unary_union(list(self.junctions.values())) if self.junctions else Polygon()
        crosswalk_area = unary_union(list(crosswalks.values())) if crosswalks else Polygon()
        filled, refuges, parts = [], [], []
        for part in _parts(surface):
            holes = []
            for ring in part.interiors:
                hole = Polygon(ring)
                inside = not junction_area.is_empty and hole.intersection(junction_area).area > 0.5 * hole.area
                if inside and (crosswalk_area.is_empty or hole.distance(crosswalk_area) >= ISLAND_TOUCH):
                    filled.append(hole)
                    continue
                holes.append(ring)
                if hole.buffer(-REFUGE_WIDTH / 2).is_empty:
                    refuges.append(hole)
            parts.append(Polygon(part.exterior, holes))
        self.surface = unary_union(parts)
        self.refuges = tuple(refuges)
        self._surface_pieces = lanes + filled
        self._surface_tree = STRtree(self._surface_pieces)
        # What a crossing may pass over: the road, refuges, and wherever a
        # crosswalk is painted, which may cut through a median wider than a refuge.
        self._road_pieces = lanes + filled + list(refuges) + [crosswalks[key] for key in sorted(crosswalks)]
        self._road_tree = STRtree(self._road_pieces)
        self._junction_ids = sorted(self.junctions)
        self._junction_tree = STRtree([self.junctions[key] for key in self._junction_ids])
        rings = self._rings()
        self.sidewalks = {f"sw{index}": line for index, line in enumerate(sorted(rings, key=lambda l: l.coords[0]))}
        self._sidewalk_ids = sorted(self.sidewalks)
        self._sidewalk_tree = STRtree([self.sidewalks[key] for key in self._sidewalk_ids])
        self.crossings, self.unconnected_crossings = self._crossings(crosswalks)
        self._prune_sidewalks()
        self._crossing_ids = sorted(self.crossings)
        self._crossing_tree = STRtree([self.crossings[key].polygon for key in self._crossing_ids])
        city = DescriptorBundle(service.descriptor_path).message(
            "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name["type"].enum_type.values_by_name["CITY_DRIVING"].number
        junction = junction_lanes(service)
        self._road_lanes = sorted(lane_id for lane_id, lane in service.lane_table.items()
                                  if lane.type == city and lane_id not in junction)
        self._road_lane_lengths = np.cumsum([service.get_lane_central_curve_by_id(lane_id).length
                                             for lane_id in self._road_lanes]) if self._road_lanes else np.zeros(0)
        self._ring_lengths = np.cumsum([self.sidewalks[key].length for key in self._sidewalk_ids]) \
            if self._sidewalk_ids else np.zeros(0)
        self._diagonals: Dict[str, List] = {}
        self._checks: Dict = {}
        self.edge_gates = self._edge_gates()
        self._gate_tree = STRtree(list(self.edge_gates))

    def _edge_gates(self) -> Tuple[LineString, ...]:
        """
        Lines continuing every lane that ends at the edge of the map (``EDGE_GATE_LENGTH``): without a
        successor (or predecessor), at the road surface's boundary and within ``EDGE_HULL_REACH`` of the
        surface's convex hull. Lanes that end inside the map (at a curb, a driveway, a junction's edge)
        are not gates.

        :returns: Map-edge gate segments in stable lane order.
        :rtype: Tuple[LineString, ...]
        """
        boundary = self.surface.boundary
        hull = self.surface.convex_hull.boundary
        gates = []
        for lane_id in sorted(self.service.lane_table):
            lane = self.service.lane_table[lane_id]
            coords = list(self.service.get_lane_central_curve_by_id(lane_id).coords)
            if len(coords) < 2:
                continue
            for ids, (a, b) in ((lane.successor_id, (coords[-2], coords[-1])),
                                (lane.predecessor_id, (coords[1], coords[0]))):
                if len(ids) or boundary.distance(Point(b)) > 1.0 or hull.distance(Point(b)) > EDGE_HULL_REACH:
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                norm = hypot(dx, dy)
                if norm < 1e-9:
                    continue
                gates.append(LineString([b, (b[0] + dx / norm * EDGE_GATE_LENGTH,
                                             b[1] + dy / norm * EDGE_GATE_LENGTH)]))
        return tuple(gates)

    def _check_gates(self, name: str, coords: Sequence[Tuple[float, float]], ends: bool = True) -> None:
        """
        Reject a sidewalk stretch or roadside walk that goes round a road end at the map edge.

        :param str name: Walk-element name used in diagnostics.
        :param Sequence[Tuple[float, float]] coords: Ordered map coordinates in metres.
        :param bool ends: Whether to additionally require both geometry endpoints to keep the gate clearance.
        :raises InvalidRoute: If the geometry crosses a map-edge gate or a checked endpoint lies too close to one.
        """
        if not self.edge_gates:
            return
        geometry = LineString(coords) if len(coords) > 1 else Point(coords[0])
        for index in self._gate_tree.query(geometry.buffer(EDGE_GATE_CLEARANCE)):
            gate = self.edge_gates[int(index)]
            if len(coords) > 1 and geometry.intersects(gate):
                raise InvalidRoute(f"{name} goes round the end of a road at the map edge")
        for end in ((coords[0], coords[-1]) if ends else ()):
            point = Point(end)
            if any(self.edge_gates[int(index)].distance(point) < EDGE_GATE_CLEARANCE
                   for index in self._gate_tree.query(point.buffer(EDGE_GATE_CLEARANCE))):
                raise InvalidRoute(f"{name} starts or ends on a road end at the map edge")

    def _rings(self) -> List[LineString]:
        """
        Candidate sidewalk rings: the surface's boundary offset outward, outside refuges.

        :returns: Candidate sidewalk-ring centerlines.
        :rtype: List[LineString]
        """
        grown = self.surface.buffer(SIDEWALK_OFFSET)
        lines = []
        for part in _parts(grown):
            for ring in [part.exterior, *part.interiors]:
                line = _ring_line(list(ring.coords))
                if line.length <= 1.0:
                    continue
                if ring is not part.exterior:
                    region = Polygon(ring)
                    if region.buffer(-(REFUGE_WIDTH / 2 - SIDEWALK_OFFSET)).is_empty:
                        continue   # the hole it rings is a refuge
                lines.append(line)
        return lines

    def _crossings(self, crosswalks: Dict[str, Polygon]) -> Tuple[Dict[str, Crossing], int]:
        """
        Join crosswalks end to end and land each chain on a sidewalk at both ends.

        :param Dict[str, Polygon] crosswalks: Crosswalk polygons keyed by map ID.
        :returns: Connected marked crossings keyed by ID and the count of unconnected crosswalk chains.
        :rtype: Tuple[Dict[str, Crossing], int]
        """
        ids = sorted(crosswalks)
        axes = {key: _min_rect_axis(crosswalks[key]) for key in ids}
        parent = {key: key for key in ids}

        def root(key: str) -> str:
            """
            Find and compress the representative of a crosswalk chain.

            :param str key: Crosswalk ID whose chain representative is requested.
            :returns: Representative crosswalk ID of the chain.
            :rtype: str
            """
            while parent[key] != key:
                key = parent[key]
            return key

        for index, first in enumerate(ids):
            a, _ = axes[first]
            for second in ids[index + 1:]:
                b, _ = axes[second]
                if _undirected_angle(_heading(a), _heading(b)) > CHAIN_ANGLE:
                    continue
                gap, p, q = min((Point(p).distance(Point(q)), p, q) for p in a.coords for q in b.coords)
                # Misalignment where they meet: each facing end's offset from the other's axis line.
                if gap <= CHAIN_GAP and max(_sideways(a, q), _sideways(b, p)) <= CHAIN_OFFSET:
                    parent[root(first)] = root(second)
        chains: Dict[str, List[str]] = {}
        for key in ids:
            chains.setdefault(root(key), []).append(key)
        result, unconnected = {}, 0
        for members in chains.values():
            members = sorted(members)
            ends = [Point(p) for key in members for p in axes[key][0].coords]
            a, b = max(((p, q) for p in ends for q in ends), key=lambda pair: pair[0].distance(pair[1]))
            if a.distance(b) < MIN_CROSSING:
                unconnected += 1
                continue
            ux, uy = (b.x - a.x) / a.distance(b), (b.y - a.y) / a.distance(b)
            landings = [self._landing(a, (-ux, -uy)), self._landing(b, (ux, uy))]
            if None in landings:
                unconnected += 1
                continue
            first, second = landings
            p, q = self.point(first), self.point(second)
            if p.distance(q) < MIN_CROSSING:
                unconnected += 1
                continue
            polygon = unary_union([crosswalks[key] for key in members])
            crossing_id = "+".join(members)
            middle = LineString([p, q]).interpolate(0.5, normalized=True)
            near = [key for key in self._junction_ids if self.junctions[key].distance(middle) <= JUNCTION_REACH]
            junction = min(near, key=lambda key: self.junctions[key].distance(middle)) if near else None
            result[crossing_id] = Crossing(crossing_id, tuple(members), polygon, LineString([p, q]),
                                           min(axes[key][1] for key in members), (first, second), junction)
        return result, unconnected

    def _landing(self, end: Point, outward: Tuple[float, float]) -> CurbPoint | None:
        """
        Where the axis, extended outward from ``end``, meets the nearest
        sidewalk; at a corner, where the axis runs past the sidewalk's bend,
        the sidewalk point nearest the end within ``LANDING_SNAP``.

        :param Point end: Crosswalk-axis endpoint to connect to a sidewalk.
        :param Tuple[float, float] outward: Unit direction extending a crossing axis toward a sidewalk.
        :returns: Nearest sidewalk landing reached by the outward probe, or ``None`` if none is found.
        :rtype: CurbPoint | None
        """
        ux, uy = outward
        probe = LineString([(end.x - LANDING_INWARD * ux, end.y - LANDING_INWARD * uy),
                            (end.x + CROSSWALK_EXTENSION * ux, end.y + CROSSWALK_EXTENSION * uy)])
        best = None
        for index in self._sidewalk_tree.query(probe, predicate="intersects"):
            sidewalk_id = self._sidewalk_ids[int(index)]
            for hit in _points(probe.intersection(self.sidewalks[sidewalk_id])):
                gap = hit.distance(end)
                if best is None or gap < best[0]:
                    best = (gap, sidewalk_id, hit)
        if best is not None:
            return self.nearest(best[2], best[1])
        for index in sorted(self._sidewalk_tree.query(end, predicate="dwithin", distance=LANDING_SNAP)):
            sidewalk_id = self._sidewalk_ids[int(index)]
            gap = self.sidewalks[sidewalk_id].distance(end)
            if best is None or gap < best[0]:
                best = (gap, sidewalk_id, end)
        return None if best is None else self.nearest(best[2], best[1])

    def _prune_sidewalks(self) -> None:
        """
        Drop small rings that no crossing reaches.
        """
        reached = {landing.sidewalk for crossing in self.crossings.values() for landing in crossing.landings}
        keep = {key for key, line in self.sidewalks.items()
                if key in reached or Polygon(line.coords).buffer(SIDEWALK_OFFSET).area >= MIN_BLOCK_AREA}
        self.sidewalks = {key: line for key, line in self.sidewalks.items() if key in keep}
        self._sidewalk_ids = sorted(self.sidewalks)
        self._sidewalk_tree = STRtree([self.sidewalks[key] for key in self._sidewalk_ids])

    def ring_length(self, sidewalk: str) -> float:
        """
        :param str sidewalk: Sidewalk ring ID.
        :returns: Sidewalk-ring length, in metres.
        :rtype: float
        :raises KeyError: If the network has no such sidewalk.
        """
        return self.sidewalks[sidewalk].length

    def wrapped(self, sidewalk: str, s: float) -> CurbPoint:
        """
        The curb point at ``s`` metres along a ring, wrapped into ``[0, length)``.

        :param str sidewalk: Sidewalk ring ID.
        :param float s: Position along the ring before wrapping, in metres.
        :returns: The curb point at ``s`` metres along a ring, wrapped into ``[0, length)``.
        :rtype: CurbPoint
        """
        return CurbPoint(sidewalk, s % self.ring_length(sidewalk))

    def point(self, curb: CurbPoint) -> Point:
        """
        The map point of a curb point.

        :param CurbPoint curb: Position on a sidewalk ring.
        :returns: The map position as a Shapely point.
        :rtype: Point
        :raises InvalidRoute: If the sidewalk is unknown or ``s`` is outside the ring.
        """
        line = self.sidewalks.get(curb.sidewalk)
        if line is None:
            raise InvalidRoute(f"unknown sidewalk {curb.sidewalk!r}")
        if not curb.s < line.length:
            raise InvalidRoute(f"s={curb.s:.3f} is outside sidewalk {curb.sidewalk!r} [0, {line.length:.3f})")
        return line.interpolate(curb.s)

    def nearest(self, point: Point, sidewalk: str | None = None) -> CurbPoint:
        """
        The curb point nearest ``point``, on ``sidewalk`` if given.

        :param Point point: Map point to project onto a sidewalk.
        :param str | None sidewalk: Sidewalk ring ID; ``None`` allows selection among all rings.
        :returns: The curb point nearest ``point``, on ``sidewalk`` if given.
        :rtype: CurbPoint
        """
        if sidewalk is None:
            index = self._sidewalk_tree.query_nearest(point)
            sidewalk = self._sidewalk_ids[int(min(index))]
        line = self.sidewalks[sidewalk]
        return CurbPoint(sidewalk, line.project(point) % line.length)

    def gap(self, a: CurbPoint, b: CurbPoint) -> float:
        """
        Metres along the sidewalk between two points the shorter way; infinite on different rings.

        :param CurbPoint a: First curb point.
        :param CurbPoint b: Second curb point.
        :returns: Shorter sidewalk distance in metres, or infinity if the points are on different rings.
        :rtype: float
        """
        if a.sidewalk != b.sidewalk:
            return float("inf")
        length = self.ring_length(a.sidewalk)
        forward = (b.s - a.s) % length
        return min(forward, length - forward)

    def along(self, a: CurbPoint, b: CurbPoint) -> List[Tuple[float, float]]:
        """
        The sidewalk's vertices from ``a`` to ``b`` the shorter way round,
        ``a`` and ``b`` included; one point if they coincide.

        :param CurbPoint a: Start curb point.
        :param CurbPoint b: End curb point.
        :returns: The sidewalk's vertices from ``a`` to ``b`` the shorter way round, ``a`` and ``b`` included; one point if they coincide.
        :rtype: List[Tuple[float, float]]
        :raises InvalidRoute: If they lie on different sidewalks.
        """
        if a.sidewalk != b.sidewalk:
            raise InvalidRoute(f"sidewalks {a.sidewalk!r} and {b.sidewalk!r} are not joined")
        line = self.sidewalks[a.sidewalk]
        length = line.length
        forward = (b.s - a.s) % length
        if min(forward, length - forward) < 1e-9:
            return [tuple(self.point(a).coords[0])]
        if forward <= length - forward:
            return _ring_stretch(line, a.s, a.s + forward)
        return _ring_stretch(line, b.s, b.s + length - forward)[::-1]

    def road_near(self, geometry: BaseGeometry, reach: float = 0.0) -> BaseGeometry:
        """
        What a crossing may pass over around ``geometry`` (the road, refuges and
        crosswalks), from the pieces within ``reach``.

        :param BaseGeometry geometry: Geometry to query or decompose.
        :param float reach: Search distance around ``geometry``, in metres.
        :returns: Union of nearby road, refuge, and crosswalk pieces.
        :rtype: BaseGeometry
        """
        return _near(self._road_tree, self._road_pieces, geometry, reach)

    def surface_near(self, geometry: BaseGeometry, reach: float = 0.0) -> BaseGeometry:
        """
        The road surface around ``geometry``, from the lane pieces within ``reach``.

        :param BaseGeometry geometry: Geometry to query or decompose.
        :param float reach: Search distance around ``geometry``, in metres.
        :returns: The road surface around ``geometry``, from the lane pieces within ``reach``.
        :rtype: BaseGeometry
        """
        return _near(self._surface_tree, self._surface_pieces, geometry, reach)

    def curb_distance(self, line: LineString) -> float:
        """
        Metres along ``line`` from its start to where it first enters the road; its length if never.

        :param LineString line: Line geometry in map metres.
        :returns: Distance from the line start to its first road entry, or the full line length if it never enters.
        :rtype: float
        """
        road = self.surface_near(line)
        entered = [min(line.project(Point(xy)) for xy in part.coords) for part in _parts(line.intersection(road))
                   if part.length > 0]
        return min(entered) if entered else line.length

    def island_spans(self, line: LineString) -> List[Tuple[float, float]]:
        """
        (start, end) metres along a crossing between two stretches of road:
        refuges, and medians and islands a crosswalk cuts through. A walker
        crossing in two stages waits there.

        :param LineString line: Line geometry in map metres.
        :returns: Sorted interior spans outside the road surface, in line metres.
        :rtype: List[Tuple[float, float]]
        """
        spans = []
        for part in _parts(line.difference(self.surface_near(line))):
            if part.length < 0.5:
                continue
            positions = [line.project(Point(xy)) for xy in part.coords]
            low, high = min(positions), max(positions)
            if low > 1e-6 and high < line.length - 1e-6:
                spans.append((low, high))
        return sorted(spans)

    def excursion_line(self, kind: ExcursionKind, entry: CurbPoint, exit: CurbPoint) -> LineString:
        """
        The path of an excursion: a straight crossing, or for ``ROADSIDE`` a step
        into the road, a walk ``ROADSIDE_OFFSET`` inside the curb, and a step back.

        :param ExcursionKind kind: Excursion category whose geometry or constraints apply.
        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :returns: The path of an excursion: a straight crossing, or for ``ROADSIDE`` a step into the road, a walk ``ROADSIDE_OFFSET`` inside the curb, and a step back.
        :rtype: LineString
        :raises InvalidRoute: If a point is off the network or a roadside walk has no road beside it.
        """
        a, b = self.point(entry), self.point(exit)
        if kind is not ExcursionKind.ROADSIDE:
            if a.distance(b) < 1e-6:
                raise InvalidRoute("an excursion needs distinct entry and exit points")
            return LineString([a, b])
        stretch = self.along(entry, exit)
        if len(stretch) < 2:
            raise InvalidRoute("a roadside walk needs distinct entry and exit points")
        line = LineString(stretch)
        road = self.road_near(line, SIDEWALK_OFFSET + 1)
        inside = SIDEWALK_OFFSET + ROADSIDE_OFFSET
        for offset in (inside, -inside):
            shifted = offset_line(line, offset)
            if road.contains(shifted.interpolate(0.5, normalized=True)):
                return LineString([tuple(a.coords[0]), *shifted.coords, tuple(b.coords[0])])
        raise InvalidRoute("no road beside the roadside walk")

    def check_excursion(self, kind: ExcursionKind, entry: CurbPoint, exit: CurbPoint) -> None:
        """
        Check that leaving the sidewalk at ``entry`` and returning at ``exit``
        is an excursion of ``kind`` (``docs/pedestrian-agents.md``).

        :param ExcursionKind kind: Excursion category whose geometry or constraints apply.
        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :raises InvalidRoute: If the entry and exit do not satisfy the requested excursion kind.
        """
        key = (kind, entry, exit)
        if key not in self._checks:
            error = None
            try:
                self._check_excursion(kind, entry, exit)
            except InvalidRoute as problem:
                error = str(problem)
            if len(self._checks) >= _CACHE_SIZE:
                del self._checks[next(iter(self._checks))]
            self._checks[key] = error
        if self._checks[key] is not None:
            raise InvalidRoute(self._checks[key])

    def _check_excursion(self, kind: ExcursionKind, entry: CurbPoint, exit: CurbPoint) -> None:
        """
        Validate the geometry and kind-specific constraints of one excursion.

        :param ExcursionKind kind: Excursion category whose geometry or constraints apply.
        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :raises InvalidRoute: If the excursion is too short, misses the road, cuts through a block, or fails its kind-specific check.
        """
        line = self.excursion_line(kind, entry, exit)
        if kind is ExcursionKind.ROADSIDE:
            self._check_roadside(entry, exit, line)
            return
        if line.length < MIN_CROSSING:
            raise InvalidRoute(f"crossing of {line.length:.2f} m is shorter than {MIN_CROSSING} m")
        # Every lane piece the clearance test could measure from, not only those the line touches.
        road = self.road_near(line, BLOCK_CLEARANCE)
        on_road = line.intersection(road)
        if on_road.length < ROAD_SHARE * line.length:
            raise InvalidRoute("the crossing does not cross the road")
        if line.difference(road.buffer(BLOCK_CLEARANCE)).length > _STRAY:
            raise InvalidRoute("the crossing cuts through a block")
        check = {
            ExcursionKind.CROSSWALK: self._check_crosswalk,
            ExcursionKind.NEAR_CROSSWALK: self._check_near_crosswalk,
            ExcursionKind.MID_BLOCK: self._check_mid_block,
            ExcursionKind.DIAGONAL: self._check_diagonal,
        }[kind]
        check(entry, exit, line, on_road)

    def _check_crosswalk(self, entry: CurbPoint, exit: CurbPoint, line: LineString, on_road: BaseGeometry) -> None:
        """
        Require the crossing to connect matching landings and stay inside a marked crosswalk.

        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :param LineString line: Line geometry in map metres.
        :param BaseGeometry on_road: Part of the crossing geometry lying on road or refuge surfaces.
        :raises InvalidRoute: If the crossing cannot be matched to compatible marked-crosswalk geometry.
        """
        a, b = self.point(entry), self.point(exit)
        for crossing in self._crossings_near(line, 0.0):
            for first, second in ((0, 1), (1, 0)):
                if (self.gap(entry, crossing.landings[first]) > LANDING_REACH
                        or self.gap(exit, crossing.landings[second]) > LANDING_REACH):
                    continue
                half = max(crossing.width / 2 - BODY_MARGIN, 0.0)
                if abs(crossing.lateral(a)) > half + 1e-6 or abs(crossing.lateral(b)) > half + 1e-6:
                    continue
                if on_road.difference(crossing.polygon.buffer(CROSSWALK_TOLERANCE)).length <= _STRAY:
                    return
        raise InvalidRoute("the crossing does not run inside a crosswalk")

    def _check_near_crosswalk(self, entry: CurbPoint, exit: CurbPoint, line: LineString, on_road: BaseGeometry) -> None:
        """
        Require the crossing to run beside and roughly parallel to a marked crosswalk.

        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :param LineString line: Line geometry in map metres.
        :param BaseGeometry on_road: Part of the crossing geometry lying on road or refuge surfaces.
        :raises InvalidRoute: If the crossing is not suitably positioned beside a marked crosswalk.
        """
        middle = on_road.interpolate(0.5, normalized=True) if on_road.length else line.interpolate(0.5, normalized=True)
        for crossing in self._crossings_near(line, NEAR_RANGE[1] + 10):
            if _undirected_angle(_heading(line), _heading(crossing.axis)) > NEAR_ANGLE:
                continue
            if line.intersects(crossing.polygon):
                continue
            outside = abs(crossing.lateral(middle)) - crossing.width / 2
            if NEAR_RANGE[0] <= outside <= NEAR_RANGE[1] and 0 <= crossing.along(middle) <= crossing.axis.length:
                return
        raise InvalidRoute("the crossing is not beside a crosswalk")

    def _check_mid_block(self, entry: CurbPoint, exit: CurbPoint, line: LineString, on_road: BaseGeometry) -> None:
        """
        Require clearance from crosswalks and junctions and an approximately road-normal crossing.

        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :param LineString line: Line geometry in map metres.
        :param BaseGeometry on_road: Part of the crossing geometry lying on road or refuge surfaces.
        :raises InvalidRoute: If the crossing is too close to a crosswalk or junction, or is too oblique to the road normal.
        """
        if len(self._crossing_tree.query(on_road, predicate="dwithin", distance=MID_BLOCK_CROSSWALK_CLEARANCE)):
            raise InvalidRoute(f"a mid-block crossing keeps more than {MID_BLOCK_CROSSWALK_CLEARANCE} m from crosswalks")
        if len(self._junction_tree.query(on_road, predicate="dwithin", distance=MID_BLOCK_JUNCTION_CLEARANCE)):
            raise InvalidRoute(f"a mid-block crossing keeps {MID_BLOCK_JUNCTION_CLEARANCE} m from junctions")
        heading = self._road_heading(on_road.interpolate(0.5, normalized=True))
        if _undirected_angle(_heading(line), heading) < pi / 2 - MID_BLOCK_ANGLE - 1e-9:
            raise InvalidRoute("a mid-block crossing keeps within 45 degrees of the road's normal")

    def _check_diagonal(self, entry: CurbPoint, exit: CurbPoint, line: LineString, on_road: BaseGeometry) -> None:
        """
        Require a diagonal crossing through a junction with limited marked-crosswalk overlap.

        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :param LineString line: Line geometry in map metres.
        :param BaseGeometry on_road: Part of the crossing geometry lying on road or refuge surfaces.
        :raises InvalidRoute: If no junction satisfies the diagonal-crossing constraints.
        """
        a, b = self.point(entry), self.point(exit)
        for index in self._junction_tree.query(on_road, predicate="intersects"):
            junction_id = self._junction_ids[int(index)]
            area = self.junctions[junction_id]
            if on_road.intersection(area).length < DIAGONAL_SHARE * on_road.length:
                continue
            if area.distance(a) > DIAGONAL_REACH or area.distance(b) > DIAGONAL_REACH:
                continue
            own = [crossing for crossing in self.crossings.values() if crossing.junction == junction_id]
            if not own or any(_undirected_angle(_heading(line), _heading(crossing.axis)) <= DIAGONAL_ANGLE
                              for crossing in own):
                continue
            marked = on_road.intersection(unary_union([crossing.polygon for crossing in own])).length
            if marked <= DIAGONAL_MARKED * on_road.length:
                return
        raise InvalidRoute("the crossing does not cut diagonally through a junction")

    def _check_roadside(self, entry: CurbPoint, exit: CurbPoint, line: LineString) -> None:
        """
        Require a roadside walk to stay on the road and outside junctions within length bounds.

        :param CurbPoint entry: Excursion entry position on a sidewalk.
        :param CurbPoint exit: Excursion exit position on a sidewalk.
        :param LineString line: Line geometry in map metres.
        :raises InvalidRoute: If the roadside walk has an invalid length, leaves the road, or enters a junction.
        """
        gap = self.gap(entry, exit)
        if not ROADSIDE_RANGE[0] <= gap <= ROADSIDE_RANGE[1]:
            raise InvalidRoute(f"a roadside walk runs {ROADSIDE_RANGE[0]}-{ROADSIDE_RANGE[1]} m along one sidewalk")
        inner = LineString(line.coords[1:-1])
        road = self.road_near(inner)
        if inner.difference(road).length > _STRAY:
            raise InvalidRoute("a roadside walk leaves the road")
        if len(self._junction_tree.query(inner, predicate="intersects")):
            raise InvalidRoute("a roadside walk stays out of junctions")

    def _crossings_near(self, geometry: BaseGeometry, reach: float) -> List[Crossing]:
        """
        Find marked crossings near a geometry in stable spatial-index order.

        :param BaseGeometry geometry: Geometry to query or decompose.
        :param float reach: Search distance from ``geometry``, in metres; zero uses the landing-reach buffer.
        :returns: Marked crossings returned in stable spatial-index order.
        :rtype: List[Crossing]
        """
        found = self._crossing_tree.query(geometry, predicate="dwithin", distance=reach) if reach \
            else self._crossing_tree.query(geometry.buffer(LANDING_REACH), predicate="intersects")
        return [self.crossings[self._crossing_ids[int(index)]] for index in sorted(found)]

    # ----- walks -----

    def elements(self, route: WalkRoute) -> Tuple[WalkElement, ...]:
        """
        The walk's elements with their raw geometry, in walking order.

        :param WalkRoute route: Walk whose elements are requested.
        :returns: The walk's elements with their raw geometry, in walking order.
        :rtype: Tuple[WalkElement, ...]
        :raises InvalidRoute: If a point is off the network, two consecutive
            points lie on different sidewalks, or a turn back is out of range.
        """
        key = ("elements", route)
        if key not in self._checks:
            try:
                value = self._elements(route)
            except InvalidRoute as problem:
                value = problem
            if len(self._checks) >= _CACHE_SIZE:
                del self._checks[next(iter(self._checks))]
            self._checks[key] = value
        value = self._checks[key]
        if isinstance(value, InvalidRoute):
            raise InvalidRoute(str(value))
        return value

    def _elements(self, route: WalkRoute) -> Tuple[WalkElement, ...]:
        """
        Construct ordered sidewalk and excursion elements, checking turn backs and map-edge gates.

        :param WalkRoute route: Walk whose elements are constructed.
        :returns: Raw sidewalk and excursion elements in walking order.
        :rtype: Tuple[WalkElement, ...]
        :raises InvalidRoute: If a curb point or sidewalk connection is invalid, a turn-back position is out of range, or an element violates a map-edge gate.
        """
        elements, here = [], route.start
        self.point(here)
        for index, excursion in enumerate(route.excursions):
            elements.append(self._sidewalk_element(f"w{index}", here, excursion.entry))
            line = self.excursion_line(excursion.kind, excursion.entry, excursion.exit)
            if excursion.kind is not ExcursionKind.CROSSWALK:
                # Crossings off the paint (mid-block, beside a crosswalk, diagonal) and
                # roadside walks may not pass the end of a road at the map edge either.
                self._check_gates(f"x{index}", list(line.coords), ends=excursion.kind is ExcursionKind.ROADSIDE)
            if excursion.turn_back is None:
                elements.append(_element(f"x{index}", line))
            else:
                if not MIN_TURN_BACK <= excursion.turn_back <= line.length - MIN_TURN_BACK:
                    raise InvalidRoute(f"turn_back {excursion.turn_back:.3f} must lie within "
                                       f"[{MIN_TURN_BACK}, {line.length - MIN_TURN_BACK:.3f}] m of excursion {index}")
                out = substring(line, 0, excursion.turn_back)
                elements.append(_element(f"x{index}", out))
                elements.append(_element(f"x{index}b", LineString(list(out.coords)[::-1])))
            here = excursion.arrival
        elements.append(self._sidewalk_element(f"w{len(route.excursions)}", here, route.end))
        return tuple(elements)

    def _sidewalk_element(self, name: str, a: CurbPoint, b: CurbPoint) -> WalkElement:
        """
        Construct a sidewalk element along the shorter ring arc, rejecting map-edge gates.

        :param str name: Walk-element name.
        :param CurbPoint a: Start curb point.
        :param CurbPoint b: End curb point.
        :returns: Named sidewalk element following the shorter ring arc.
        :rtype: WalkElement
        """
        self.point(b)
        coords = self.along(a, b)
        self._check_gates(name, coords)
        return WalkElement(name, tuple(coords), LineString(coords).length if len(coords) > 1 else 0.0)

    def validate(self, route: WalkRoute) -> None:
        """
        Check a walk: its points on the network, consecutive points on one
        sidewalk, every excursion of its kind, and turn backs within range.

        :param WalkRoute route: Walk to validate.
        :raises InvalidRoute: If an excursion is invalid or the walk has zero total length.
        """
        for excursion in route.excursions:
            self.check_excursion(excursion.kind, excursion.entry, excursion.exit)
        if not self.frame(route).total > 0:
            raise InvalidRoute("the walk has no length")

    def frame(self, route: WalkRoute) -> WalkFrame:
        """
        Distances along the walk's elements (``WalkFrame``).

        :param WalkRoute route: Walk whose distance frame is requested.
        :returns: Distance frame over the walk elements.
        :rtype: WalkFrame
        :raises InvalidRoute: If the walk cannot be converted to valid elements.
        """
        return WalkFrame(self.elements(route))

    def length(self, route: WalkRoute, *, lane_change_s: Tuple[float, ...] = ()) -> float:
        """
        Walked metres along the walk's raw elements; ``lane_change_s`` is ignored,
        so code shared with lane routes can measure walks.

        :param WalkRoute route: Walk whose raw length is measured.
        :param Tuple[float, ...] lane_change_s: Compatibility argument for lane-route callers; ignored for walks.
        :returns: Total walked length along the raw walk elements, in metres.
        :rtype: float
        :raises InvalidRoute: If the walk cannot be converted to valid elements.
        """
        return self.frame(route).total

    def random_route(self, rng: Random, kinds: Sequence[ExcursionKind], *, max_link: float = 40.0,
                     max_approach: float = 30.0, turn_back_probability: float = 0.05,
                     min_walk: float = MIN_WALK) -> WalkRoute | None:
        """
        One attempt at a random walk with excursions of ``kinds``, in order.

        Each further excursion enters within ``max_link`` metres along the
        sidewalk from the previous one's exit. The last turns back with
        ``turn_back_probability``, at ``MIN_TURN_BACK`` metres to
        ``TURN_BACK_SHARE`` of its length. The start lies up to
        ``max_approach`` metres along the sidewalk from the first entry, and
        the end as far from the last arrival, either way; a walk without
        excursions runs ``min_walk`` to twice ``max_approach`` metres.

        :param Random rng: Random number generator used for sampling and selection.
        :param Sequence[ExcursionKind] kinds: Excursion kinds to include, in walk order.
        :param float max_link: Maximum sidewalk distance between connected excursions, in metres.
        :param float max_approach: Maximum sidewalk distance from an endpoint to the adjacent excursion, in metres.
        :param float turn_back_probability: Probability that the final excursion contains a turning point.
        :param float min_walk: Minimum length of a sidewalk-only walk, in metres.
        :returns: A valid sampled walk, or ``None`` if this attempt cannot satisfy the requested connections.
        :rtype: WalkRoute | None
        :raises RuntimeError: If a sidewalk point is required but the network has no sidewalks.
        """
        excursions, near = [], None
        for kind in kinds:
            pair = self.propose(kind, rng, near, max_link)
            if pair is None:
                return None
            try:
                self.check_excursion(kind, *pair)
            except InvalidRoute:
                return None
            excursions.append(Excursion(kind, *pair))
            near = pair[1]
        if excursions and rng.random() < turn_back_probability:
            last = excursions[-1]
            length = self.excursion_line(last.kind, last.entry, last.exit).length
            if length >= 3 * MIN_TURN_BACK:
                high = max(TURN_BACK_SHARE * length, MIN_TURN_BACK)
                excursions[-1] = replace(last, turn_back=rng.uniform(MIN_TURN_BACK, min(high, length - MIN_TURN_BACK)))
        if excursions:
            first, arrival = excursions[0].entry, excursions[-1].arrival
            start = self.wrapped(first.sidewalk, first.s + rng.uniform(-max_approach, max_approach))
            end = self.wrapped(arrival.sidewalk, arrival.s + rng.uniform(-max_approach, max_approach))
        else:
            start = self.random_point(rng)
            step = rng.uniform(min_walk, max(min_walk, 2 * max_approach)) * rng.choice((-1, 1))
            end = self.wrapped(start.sidewalk, start.s + step)
        route = WalkRoute(start, tuple(excursions), end)
        try:
            self.validate(route)
        except InvalidRoute:
            return None
        return route

    def kinds(self) -> FrozenSet[ExcursionKind]:
        """
        The excursion kinds this map has places for.

        :returns: The excursion kinds this map has places for.
        :rtype: FrozenSet[ExcursionKind]
        """
        kinds = set()
        if self.sidewalks:
            kinds.add(ExcursionKind.ROADSIDE)
            if self._road_lanes:
                kinds.add(ExcursionKind.MID_BLOCK)
        if self.crossings:
            kinds |= {ExcursionKind.CROSSWALK, ExcursionKind.NEAR_CROSSWALK}
        if any(self._diagonal_pairs(junction_id) for junction_id in self._junction_ids):
            kinds.add(ExcursionKind.DIAGONAL)
        return frozenset(kinds)

    def random_point(self, rng: Random) -> CurbPoint:
        """
        A curb point uniformly along all sidewalks.

        :param Random rng: Random number generator used for sampling and selection.
        :returns: A curb point uniformly along all sidewalks.
        :rtype: CurbPoint
        :raises RuntimeError: If the network has no sidewalks.
        """
        if not self._sidewalk_ids:
            raise RuntimeError("the map has no sidewalks")
        distance = rng.uniform(0, float(self._ring_lengths[-1]))
        index = min(int(np.searchsorted(self._ring_lengths, distance, side="right")), len(self._sidewalk_ids) - 1)
        sidewalk = self._sidewalk_ids[index]
        before = float(self._ring_lengths[index - 1]) if index else 0.0
        return self.wrapped(sidewalk, distance - before)

    def propose(self, kind: ExcursionKind, rng: Random, near: CurbPoint | None = None,
                reach: float = 40.0) -> Tuple[CurbPoint, CurbPoint] | None:
        """
        Draw an entry and exit for an excursion of ``kind``, entering within
        ``reach`` metres along the sidewalk of ``near`` if given. One draw:
        the caller checks it (``check_excursion``) and retries.

        :param ExcursionKind kind: Excursion category to propose.
        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to the proposed entry, in metres.
        :returns: (entry, exit), or None if nothing of the kind lies within reach.
        :rtype: Tuple[CurbPoint, CurbPoint] | None
        """
        draw = {
            ExcursionKind.CROSSWALK: self._propose_crosswalk,
            ExcursionKind.NEAR_CROSSWALK: self._propose_near_crosswalk,
            ExcursionKind.MID_BLOCK: self._propose_mid_block,
            ExcursionKind.DIAGONAL: self._propose_diagonal,
            ExcursionKind.ROADSIDE: self._propose_roadside,
        }[kind]
        try:
            return draw(rng, near, reach)
        except InvalidRoute:
            return None

    def _landings_within(self, near: CurbPoint | None, reach: float) -> List[Tuple[Crossing, int]]:
        """
        List crossing landings within the requested sidewalk distance of an optional anchor.

        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near``, in metres.
        :returns: Crossing and landing-index pairs satisfying the optional sidewalk-distance bound.
        :rtype: List[Tuple[Crossing, int]]
        """
        return [(self.crossings[key], index) for key in self._crossing_ids for index in (0, 1)
                if near is None or self.gap(self.crossings[key].landings[index], near) <= reach]

    def _beside(self, crossing: Crossing, index: int, lateral: float) -> CurbPoint:
        """
        The point on landing ``index``'s sidewalk ``lateral`` metres left of the axis.

        :param Crossing crossing: Marked crossing supplying its axis and sidewalk landings.
        :param int index: Landing index, 0 or 1.
        :param float lateral: Signed lateral displacement from the crossing axis, in metres.
        :returns: The point on landing ``index``'s sidewalk ``lateral`` metres left of the axis.
        :rtype: CurbPoint
        """
        (ux, uy), base = crossing.direction, self.point(crossing.landings[index])
        target = Point(base.x - uy * lateral, base.y + ux * lateral)
        sidewalk = crossing.landings[index].sidewalk
        return self.nearest(target, sidewalk)

    def _propose_crosswalk(self, rng: Random, near: CurbPoint | None, reach: float) -> Tuple[CurbPoint, CurbPoint] | None:
        """
        Sample an entry and exit within a marked crossing usable from the optional anchor.

        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to a proposed entry, in metres.
        :returns: The sampled entry and exit, or ``None`` if no proposal is available.
        :rtype: Tuple[CurbPoint, CurbPoint] | None
        """
        options = self._landings_within(near, reach)
        if not options:
            return None
        crossing, index = rng.choice(options)
        half = max(crossing.width / 2 - BODY_MARGIN, 0.0)
        start = rng.uniform(-half, half)
        end = min(max(start + rng.gauss(0, crossing.width / 6), -half), half)
        return self._beside(crossing, index, start), self._beside(crossing, 1 - index, end)

    def _propose_near_crosswalk(self, rng: Random, near: CurbPoint | None, reach: float) -> Tuple[CurbPoint, CurbPoint] | None:
        """
        Sample a crossing beside a marked crosswalk near the optional anchor.

        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to a proposed entry, in metres.
        :returns: The sampled entry and exit, or ``None`` if no proposal is available.
        :rtype: Tuple[CurbPoint, CurbPoint] | None
        """
        options = self._landings_within(near, reach + NEAR_RANGE[1])
        if not options:
            return None
        crossing, index = rng.choice(options)
        side = rng.choice((-1, 1))
        start = side * (crossing.width / 2 + rng.uniform(*NEAR_RANGE))
        end = start + rng.gauss(0, 1.0)
        entry, exit = self._beside(crossing, index, start), self._beside(crossing, 1 - index, end)
        if near is not None and self.gap(entry, near) > reach:
            return None
        return entry, exit

    def _propose_mid_block(self, rng: Random, near: CurbPoint | None, reach: float) -> Tuple[CurbPoint, CurbPoint] | None:
        """
        Sample a road-normal crossing from an anchored sidewalk or a random road-lane position.

        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to a proposed entry, in metres.
        :returns: The sampled entry and exit, or ``None`` if no proposal is available.
        :rtype: Tuple[CurbPoint, CurbPoint] | None
        """
        angle = rng.gauss(0, MID_BLOCK_ANGLE_SIGMA)
        while abs(angle) > MID_BLOCK_ANGLE:
            angle = rng.gauss(0, MID_BLOCK_ANGLE_SIGMA)
        if near is not None:
            entry = self.wrapped(near.sidewalk, near.s + rng.uniform(-reach, reach))
            start = self.point(entry)
            inward = self._toward_road(entry)
            if inward is None:
                return None
            heading = self._road_heading(Point(start.x + 3.5 * cos(inward), start.y + 3.5 * sin(inward)))
            normal = heading + pi / 2 if cos(heading + pi / 2 - inward) > 0 else heading - pi / 2
            exit = self._cast(start, normal + angle)
            return None if exit is None else (entry, exit)
        if not self._road_lanes:
            return None
        distance = rng.uniform(0, float(self._road_lane_lengths[-1]))
        index = min(int(np.searchsorted(self._road_lane_lengths, distance, side="right")), len(self._road_lanes) - 1)
        lane_id = self._road_lanes[index]
        line = self.service.get_lane_central_curve_by_id(lane_id)
        s = distance - (float(self._road_lane_lengths[index - 1]) if index else 0.0)
        point, heading = self.service.get_lane_coord_and_heading(lane_id, min(max(s, 0.0), line.length))
        direction = heading + pi / 2 + angle
        first, second = self._cast(point, direction + pi), self._cast(point, direction)
        if first is None or second is None:
            return None
        return (first, second) if rng.random() < 0.5 else (second, first)

    def _road_heading(self, point: Point) -> float:
        """
        The heading of the lane nearest ``point``, there.

        :param Point point: Map point at which the nearest road-lane heading is requested.
        :returns: The heading of the lane nearest ``point``, there.
        :rtype: float
        """
        lane_id = self.service.get_nearest_lane(point)
        line = self.service.get_lane_central_curve_by_id(lane_id)
        return self.service.get_lane_coord_and_heading(lane_id, line.project(point))[1]

    def _toward_road(self, curb: CurbPoint) -> float | None:
        """
        The heading from a sidewalk point toward the road beside it, or None.

        :param CurbPoint curb: Position on a sidewalk ring.
        :returns: The heading from a sidewalk point toward the road beside it, or None.
        :rtype: float | None
        """
        line = self.sidewalks[curb.sidewalk]
        length = line.length
        a, b = line.interpolate((curb.s - 1) % length), line.interpolate((curb.s + 1) % length)
        tangent = atan2(b.y - a.y, b.x - a.x)
        here = self.point(curb)
        road = self.road_near(here, SIDEWALK_OFFSET + 2)
        for side in (pi / 2, -pi / 2):
            probe = Point(here.x + (SIDEWALK_OFFSET + 1) * cos(tangent + side),
                          here.y + (SIDEWALK_OFFSET + 1) * sin(tangent + side))
            if road.contains(probe):
                return tangent + side
        return None

    def _cast(self, origin: Point, heading: float, length: float = 80.0) -> CurbPoint | None:
        """
        The first sidewalk point at least ``MIN_CROSSING`` from ``origin`` along ``heading``.

        :param Point origin: Origin of the ray, in map coordinates.
        :param float heading: Heading in radians.
        :param float length: Maximum ray length, in metres.
        :returns: The first sidewalk point at least ``MIN_CROSSING`` from ``origin`` along ``heading``.
        :rtype: CurbPoint | None
        """
        ray = LineString([origin, (origin.x + length * cos(heading), origin.y + length * sin(heading))])
        best = None
        for index in self._sidewalk_tree.query(ray, predicate="intersects"):
            sidewalk_id = self._sidewalk_ids[int(index)]
            for hit in _points(ray.intersection(self.sidewalks[sidewalk_id])):
                gap = hit.distance(origin)
                if gap >= MIN_CROSSING and (best is None or gap < best[0]):
                    best = (gap, sidewalk_id, hit)
        return None if best is None else self.nearest(best[2], best[1])

    def _diagonal_pairs(self, junction_id: str) -> List[Tuple[CurbPoint, CurbPoint]]:
        """
        Landing pairs of one junction's crossings that make diagonal crossings, both ways.

        :param str junction_id: Junction ID whose crossings or geometry are queried.
        :returns: Feasible diagonal landing pairs through the junction, in both directions.
        :rtype: List[Tuple[CurbPoint, CurbPoint]]
        """
        if junction_id not in self._diagonals:
            corners = [crossing.landings[index] for crossing in self.crossings.values()
                       if crossing.junction == junction_id for index in (0, 1)]
            pairs = []
            for i, first in enumerate(corners):
                for second in corners[i + 1:]:
                    for entry, exit in ((first, second), (second, first)):
                        try:
                            self.check_excursion(ExcursionKind.DIAGONAL, entry, exit)
                        except InvalidRoute:
                            continue
                        pairs.append((entry, exit))
            self._diagonals[junction_id] = pairs
        return self._diagonals[junction_id]

    def _propose_diagonal(self, rng: Random, near: CurbPoint | None, reach: float) -> Tuple[CurbPoint, CurbPoint] | None:
        """
        Sample a feasible diagonal landing pair and jitter both positions along their sidewalks.

        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to a proposed entry, in metres.
        :returns: The sampled entry and exit, or ``None`` if no proposal is available.
        :rtype: Tuple[CurbPoint, CurbPoint] | None
        """
        options = [pair for junction_id in self._junction_ids for pair in self._diagonal_pairs(junction_id)
                   if near is None or self.gap(pair[0], near) <= reach]
        if not options:
            return None
        entry, exit = rng.choice(options)
        # Walkers spread over the corner rather than all leaving from one spot.
        return (self.wrapped(entry.sidewalk, entry.s + rng.uniform(-1.5, 1.5)),
                self.wrapped(exit.sidewalk, exit.s + rng.uniform(-1.5, 1.5)))

    def _propose_roadside(self, rng: Random, near: CurbPoint | None, reach: float) -> Tuple[CurbPoint, CurbPoint]:
        """
        Sample a roadside excursion from a random or anchored sidewalk position.

        :param Random rng: Random number generator used for sampling and selection.
        :param CurbPoint | None near: Optional sidewalk anchor for the excursion entry.
        :param float reach: Maximum sidewalk distance from ``near`` to a proposed entry, in metres.
        :returns: Sampled roadside entry and exit on the same sidewalk.
        :rtype: Tuple[CurbPoint, CurbPoint]
        """
        entry = self.random_point(rng) if near is None else self.wrapped(
            near.sidewalk, near.s + rng.uniform(-reach, reach))
        step = rng.uniform(*ROADSIDE_RANGE) * rng.choice((-1, 1))
        return entry, self.wrapped(entry.sidewalk, entry.s + step)


def _element(name: str, line: LineString) -> WalkElement:
    """
    Build a named walk element from the two-dimensional vertices of a line.

    :param str name: Walk-element name.
    :param LineString line: Line geometry in map metres.
    :returns: Named walk element built from the line vertices.
    :rtype: WalkElement
    """
    return WalkElement(name, tuple(tuple(xy[:2]) for xy in line.coords), line.length)


def _near(tree: STRtree, pieces: List, geometry: BaseGeometry, reach: float) -> BaseGeometry:
    """
    The union of the pieces within ``reach`` of ``geometry``, small gaps closed.

    :param STRtree tree: Spatial index over the supplied geometry components.
    :param List pieces: Geometry components used by the spatial index.
    :param BaseGeometry geometry: Geometry to query or decompose.
    :param float reach: Search distance around ``geometry``, in metres.
    :returns: The union of the pieces within ``reach`` of ``geometry``, small gaps closed.
    :rtype: BaseGeometry
    """
    found = tree.query(geometry, predicate="dwithin", distance=reach + 2 * SURFACE_CLOSING)
    if not len(found):
        return Polygon()
    return unary_union([pieces[int(index)] for index in sorted(found)]).buffer(SURFACE_CLOSING).buffer(
        -SURFACE_CLOSING)


def _sideways(axis: LineString, point: Tuple[float, float]) -> float:
    """
    Distance of ``point`` from the infinite line through ``axis``.

    :param LineString axis: Reference crossing axis.
    :param Tuple[float, float] point: Map coordinate whose perpendicular distance is measured.
    :returns: Perpendicular distance from ``point`` to the infinite axis line, in metres.
    :rtype: float
    """
    (x0, y0), (x1, y1) = axis.coords[0], axis.coords[-1]
    length = hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    return abs(ux * (point[1] - y0) - uy * (point[0] - x0))


def _points(geometry: BaseGeometry) -> Iterator[Point]:
    """
    The points of an intersection result; lines contribute their vertices.

    :param BaseGeometry geometry: Geometry to query or decompose.
    :returns: Intersection points; line components contribute their vertices.
    :rtype: Iterator[Point]
    """
    for part in _parts(geometry):
        if isinstance(part, Point):
            yield part
        else:
            yield from (Point(xy) for xy in part.coords)


def _ring_stretch(line: LineString, start: float, end: float) -> List[Tuple[float, float]]:
    """
    Vertices of a closed ring from ``start`` to ``end`` metres, which may wrap past its start.

    :param LineString line: Line geometry in map metres.
    :param float start: Start distance along the ring, in metres.
    :param float end: End distance along the ring, in metres; may extend past one ring length to wrap.
    :returns: Ordered vertices from ``start`` to ``end``, wrapping past the ring origin if needed.
    :rtype: List[Tuple[float, float]]
    """
    length = line.length
    span = end - start
    start %= length
    end = start + span
    if end <= length + 1e-12:
        piece = substring(line, start, min(end, length))
        return [tuple(xy[:2]) for xy in piece.coords]
    first = [tuple(xy[:2]) for xy in substring(line, start, length).coords]
    second = [tuple(xy[:2]) for xy in substring(line, 0, end - length).coords]
    return first + second[1:]


def walk_network(service: MapService) -> WalkNetwork:
    """
    The walk network of a loaded map, built once per map.

    :param MapService service: A map loaded with ``load_only=False``.
    :returns: The walk network of a loaded map, built once per map.
    :rtype: WalkNetwork
    :raises RuntimeError: If the map was loaded without its routing graph or lane index.
    """
    network = _NETWORKS.get(service)
    if network is None:
        network = WalkNetwork(service)
        _NETWORKS[service] = network
    return network
