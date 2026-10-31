"""
Virtual connectors that carry bicycles across the gaps between bike lanes.

On the bundled maps each group of bike lanes is a single chain that ends at
a junction or a mid-road break and resumes beyond it. A connector joins a
dead end, a bike lane without a bike successor, to a start, a bike lane
without a bike predecessor. Pairs come from the car lanes beside the bike
lanes, whose topology runs through junctions, and from straight
continuations ahead. Each connector is a ``VirtualLane`` for the bicycles'
``RouteSpace``; the map itself is never changed.

Given ``road_lanes`` (the curb lanes bicycles may ride, with their shifted
lines), a dead end is also joined to the road lanes its car neighbours lead
to: where a bike lane stops, a rider carries on along the road.
"""

from enum import Enum
from math import ceil, cos, dist, hypot, radians, sin
from typing import Iterable, Iterator, Mapping
import networkx as nx
from shapely.geometry import LineString

from ..map import MapService
from ..map.geometry import chord_heading, normalize_angle
from ..message.descriptor import DescriptorBundle
from .replay import curvature_spans
from .routing import VirtualLane
from typing import FrozenSet, List, Tuple


class ConnectorKind(Enum):
    """
    How a connector crosses: by the change of heading from dead end to start.
    """

    STRAIGHT = "STRAIGHT"
    """Straight connector"""
    RIGHT = "RIGHT"
    """Right connector"""
    LEFT_TWO_STAGE = "LEFT_TWO_STAGE"
    """Straight to the far corner, a stop to turn in place, then straight across."""
    LEFT_DIAGONAL = "LEFT_DIAGONAL"
    """One smooth curve across the junction: a left turn ridden diagonally across the junction
    instead of in two stages, which riders do in practice."""


DEFAULT_KINDS = frozenset({ConnectorKind.STRAIGHT, ConnectorKind.RIGHT, ConnectorKind.LEFT_TWO_STAGE})
"""Default"""

_TOPOLOGY_STEPS = 3
"""Car-lane forward steps followed from a dead end's neighbour, enough to cross a junction."""
_AHEAD_DISTANCE = 80.0
"""Metres within which a straight continuation ahead is looked for."""
_AHEAD_HEADING = radians(30)
_AHEAD_OFFSET = 6.0
"""Largest lateral offset in metres of a straight continuation."""
_HEADING_CHORD = 3.0
"""Metres over which a lane's end or start heading is measured; single segments are noisy."""
_STRAIGHT_TURN = radians(45)
_TURN_LIMIT = radians(135)
"""Heading changes up to ``_STRAIGHT_TURN`` are straight; beyond ``_TURN_LIMIT``, U-turns get no connector."""
_MOVEMENT_ANGLE = radians(30)
"""Largest difference between the chord headings of a connector leg and of a car lane for the leg to ride
beside that lane's movement (``parallel_car_lanes``). On the bundled maps matched movements differ by at
most 4 degrees and the next-closest ones by 40 or more."""
_SAMPLE_STEP = 0.5
"""Approximate spacing in metres of a curved connector's vertices."""
_CORNER_MARGIN = 1.0
"""Metres a two-stage corner must lie beyond the dead end and before the start."""


def bike_connectors(service: MapService, *, min_turn_radius: float,
                    kinds: Iterable[ConnectorKind] = DEFAULT_KINDS,
                    max_length: float = 100.0,
                    road_lanes: Mapping[str, LineString] | None = None) -> Tuple[VirtualLane, ...]:
    """
    Return virtual bicycle connectors from bike-lane dead ends, sorted by ID.

    :param MapService service: A map with its routing graph built; it is only read.
    :param float min_turn_radius: Connectors tighter than this, in metres, are
        dropped; corners at pivots are turned at rest and not measured.
    :param Iterable[ConnectorKind] kinds: The connector kinds to generate.
    :param float max_length: Longest connector in metres.
    :param Mapping[str, LineString] | None road_lanes: Optional road lanes a
        bike-lane dead end may continue onto, keyed by lane ID, with the line
        a rider follows on each; the line's start is the connector's end.
    :returns: Generated virtual bicycle connectors sorted by lane ID.
    :rtype: Tuple[VirtualLane, ...]
    :raises RuntimeError: If the map was loaded without its routing graph.
    """
    if service.routing_graph is None:
        raise RuntimeError("bike_connectors needs a map loaded with load_only=False")
    kinds = frozenset(kinds)
    biking, city = _lane_type(service, "BIKING"), _lane_type(service, "CITY_DRIVING")
    lanes = service.lane_table
    bikes = {lane_id for lane_id, lane in lanes.items() if lane.type == biking}

    chains = nx.DiGraph()
    chains.add_nodes_from(bikes)
    chains.add_edges_from((lane_id, target) for lane_id in bikes for target in _forward(service, lane_id) if target in bikes)
    chain = {lane_id: index for index, group in enumerate(nx.weakly_connected_components(chains))
             for lane_id in group}
    ends = sorted(lane_id for lane_id in bikes if chains.out_degree(lane_id) == 0)
    starts = sorted(lane_id for lane_id in bikes if chains.in_degree(lane_id) == 0)

    pairs = set()
    beside = {}
    for start in starts:
        for car in _car_neighbours(service, start):
            beside.setdefault(car, []).append(start)
    road_lanes = road_lanes or {}
    road_pairs = set()
    for end in ends:
        frontier = set(_car_neighbours(service, end))
        for _ in range(_TOPOLOGY_STEPS):
            frontier = {target for car in frontier for target in _forward(service, car) if lanes[target].type == city}
            pairs.update((end, start) for car in frontier for start in beside.get(car, ()))
            road_pairs.update((end, car) for car in frontier if car in road_lanes)
    poses = {lane_id: _end_pose(service.get_lane_central_curve_by_id(lane_id)) for lane_id in ends}
    firsts = {lane_id: _start_pose(service.get_lane_central_curve_by_id(lane_id)) for lane_id in starts}
    for end in ends:
        (x, y), heading = poses[end]
        for start in starts:
            (sx, sy), start_heading = firsts[start]
            dx, dy = sx - x, sy - y
            along = dx * cos(heading) + dy * sin(heading)
            offset = -dx * sin(heading) + dy * cos(heading)
            if (hypot(dx, dy) <= _AHEAD_DISTANCE and along > 0 and abs(offset) < _AHEAD_OFFSET
                    and abs(normalize_angle(start_heading - heading)) < _AHEAD_HEADING):
                pairs.add((end, start))

    def usable(connector: VirtualLane) -> bool:
        """
        Check connector length, turning radius, and physical-curb conflicts.

        :param VirtualLane connector: Virtual bicycle lane whose geometry is checked.
        :returns: ``True`` if the connector meets length, curvature, and curb checks.
        :rtype: bool
        """
        return (connector.line.length <= max_length and not _too_tight(connector, min_turn_radius)
                and not any(not mark.virtual and "CURB" in mark.types
                            for mark in service.get_lane_markings(connector.line)))

    connectors = []
    for end, start in sorted(pairs):
        if chain[end] != chain[start]:
            connectors.extend(filter(usable, _connect(end, poses[end], start, firsts[start], kinds)))
    for end, road in sorted(road_pairs):
        connectors.extend(filter(usable, _connect(end, poses[end], road, _start_pose(road_lanes[road]), kinds)))
    return tuple(sorted(connectors, key=lambda connector: connector.lane_id))


Pose = Tuple[Tuple[float, float], float]
"""A point and a heading in radians."""


def parallel_car_lanes(service: MapService, connector: VirtualLane) -> Tuple[Tuple[str, ...], ...]:
    """
    The car lanes each leg of a bicycle connector rides beside: one tuple for a straight or right
    connector, two for a two-stage left (before and after its corner).

    A rider crosses with the car movement beside it and so obeys that movement's light
    (``signal_compliance.connector_signal_stops``); bike lanes carry no signal of their own. The
    first leg looks at the car lanes next to the bike lane it leaves and the lanes they lead to,
    the second at the lanes next to the lane it joins (that lane itself when it is a road lane)
    and the lanes leading into them. A car lane counts when its chord heading is within
    ``_MOVEMENT_ANGLE`` of the leg's.

    :param MapService service: The map the connector was built on (``bike_connectors``).
    :param VirtualLane connector: A connector from ``bike_connectors``.
    :returns: Sorted car-lane IDs per leg; a leg beside no car movement gets an empty tuple.
    :rtype: Tuple[Tuple[str, ...], ...]
    """
    city = _lane_type(service, "CITY_DRIVING")
    (end,), (start,) = connector.predecessors, connector.successors
    out = set(_car_neighbours(service, end))
    out |= {target for lane in list(out) for target in _forward(service, lane)}
    joined = [start] if service.lane_table[start].type == city else _car_neighbours(service, start)
    into = set(joined) | {source for lane in joined for source in _backward(service, lane)}
    cuts = (0.0, *connector.pivots, connector.line.length)
    legs = []
    for (a, b), candidates in zip(zip(cuts, cuts[1:]), (out, into) if connector.pivots else (out,)):
        heading = chord_heading(connector.line, a, b)
        legs.append(tuple(sorted(
            lane for lane in candidates if service.lane_table[lane].type == city
            and abs(normalize_angle(_lane_heading(service, lane) - heading)) <= _MOVEMENT_ANGLE)))
    return tuple(legs)


def _lane_type(service: MapService, name: str) -> int:
    """
    The number of a ``apollo.hdmap.Lane.Type`` value.

    :param MapService service: Map whose descriptor defines the enum.
    :param str name: Type name, e.g. ``"BIKING"``.
    :returns: Its enum number.
    :rtype: int
    """
    return DescriptorBundle(service.descriptor_path).message(
        "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name["type"].enum_type.values_by_name[name].number


def _forward(service: MapService, lane_id: str) -> List[str]:
    """
    Forward successor lanes in stable order.

    :param MapService service: Map with its routing graph.
    :param str lane_id: Map lane ID.
    :returns: Forward-connected map lane IDs, sorted.
    :rtype: List[str]
    """
    graph = service.routing_graph
    return sorted(target for target in graph.successors(lane_id) if graph[lane_id][target]["direction_type"] == "F")


def _backward(service: MapService, lane_id: str) -> List[str]:
    """
    Lanes that lead forward into ``lane_id``, in stable order.

    :param MapService service: Map with its routing graph.
    :param str lane_id: Map lane ID.
    :returns: Forward-connected predecessor lane IDs, sorted.
    :rtype: List[str]
    """
    graph = service.routing_graph
    return sorted(source for source in graph.predecessors(lane_id) if graph[source][lane_id]["direction_type"] == "F")


def _car_neighbours(service: MapService, lane_id: str) -> List[str]:
    """
    City-driving lanes beside a lane (its left and right forward neighbours).

    :param MapService service: Map supplying the lane table.
    :param str lane_id: Map lane ID, usually a bike lane.
    :returns: IDs of adjacent city-driving lanes, left before right.
    :rtype: List[str]
    """
    lanes, city = service.lane_table, _lane_type(service, "CITY_DRIVING")
    lane = lanes[lane_id]
    return [other.id for other in (*lane.left_neighbor_forward_lane_id, *lane.right_neighbor_forward_lane_id)
            if other.id in lanes and lanes[other.id].type == city]


def _lane_heading(service: MapService, lane_id: str) -> float:
    """
    Chord heading of a lane from its start to its end.

    :param MapService service: Map supplying lane centerlines.
    :param str lane_id: Map lane ID.
    :returns: Direction in radians.
    :rtype: float
    """
    line = service.get_lane_central_curve_by_id(lane_id)
    return chord_heading(line, 0.0, line.length)


def _end_pose(line: LineString) -> Pose:
    """
    The last point of a lane and its heading over the last ``_HEADING_CHORD`` metres.

    :param LineString line: Line geometry in map metres.
    :returns: The last point of a lane and its heading over the last ``_HEADING_CHORD`` metres.
    :rtype: Pose
    """
    return line.coords[-1][:2], chord_heading(line, max(0.0, line.length - _HEADING_CHORD), line.length)


def _start_pose(line: LineString) -> Pose:
    """
    The first point of a lane and its heading over the first ``_HEADING_CHORD`` metres.

    :param LineString line: Line geometry in map metres.
    :returns: The first point of a lane and its heading over the first ``_HEADING_CHORD`` metres.
    :rtype: Pose
    """
    return line.coords[0][:2], chord_heading(line, 0.0, min(_HEADING_CHORD, line.length))


def _connect(end: str, end_pose: Pose, start: str, start_pose: Pose,
             kinds: FrozenSet[ConnectorKind]) -> Iterator[VirtualLane]:
    """
    The connectors of the allowed kinds from ``end`` to ``start``, before length and radius checks.

    :param str end: Source lane ID whose end is connected.
    :param Pose end_pose: Position and heading at the source lane end.
    :param str start: Destination lane ID whose start is connected.
    :param Pose start_pose: Position and heading at the destination lane start.
    :param FrozenSet[ConnectorKind] kinds: Connector kinds allowed for this lane pair.
    :returns: Generated connectors of the allowed kinds before length and
        turning-radius checks.
    :rtype: Iterator[VirtualLane]
    """
    (p0, h0), (p3, h3) = end_pose, start_pose
    if dist(p0, p3) == 0:
        return
    turn = normalize_angle(h3 - h0)
    if abs(turn) < _STRAIGHT_TURN:
        candidates = [ConnectorKind.STRAIGHT]
    elif -_TURN_LIMIT <= turn <= -_STRAIGHT_TURN:
        candidates = [ConnectorKind.RIGHT]
    elif _STRAIGHT_TURN <= turn <= _TURN_LIMIT:
        candidates = [ConnectorKind.LEFT_TWO_STAGE, ConnectorKind.LEFT_DIAGONAL]
    else:
        return
    for kind in candidates:
        if kind not in kinds:
            continue
        lane_id = f"bike_{kind.name.lower()}:{end}>{start}"
        if kind is ConnectorKind.LEFT_TWO_STAGE:
            corner = _corner(p0, h0, p3, h3)
            if corner is not None:
                yield VirtualLane(lane_id, LineString([p0, corner, p3]), (end,), (start,),
                                  pivots=(dist(p0, corner),))
        else:
            yield VirtualLane(lane_id, _bezier(p0, h0, p3, h3), (end,), (start,))


def _bezier(p0: Tuple[float, float], h0: float, p3: Tuple[float, float], h3: float) -> LineString:
    """
    A cubic Bézier curve from ``p0`` to ``p3``, leaving along ``h0`` and arriving along ``h3``.

    :param Tuple[float, float] p0: Starting map coordinate, in metres.
    :param float h0: Departure heading, in radians.
    :param Tuple[float, float] p3: Ending map coordinate, in metres.
    :param float h3: Arrival heading, in radians.
    :returns: A cubic Bézier curve from ``p0`` to ``p3``, leaving along ``h0`` and arriving along ``h3``.
    :rtype: LineString
    """
    reach = dist(p0, p3) / 3
    p1 = (p0[0] + reach * cos(h0), p0[1] + reach * sin(h0))
    p2 = (p3[0] - reach * cos(h3), p3[1] - reach * sin(h3))
    count = max(2, ceil((dist(p0, p1) + dist(p1, p2) + dist(p2, p3)) / _SAMPLE_STEP))
    points = []
    for index in range(count + 1):
        t = index / count
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        points.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                       a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    points[0], points[-1] = p0, p3
    return LineString(points)


def _corner(p0: Tuple[float, float], h0: float, p3: Tuple[float, float], h3: float,
            ) -> Tuple[float, float] | None:
    """
    Where the line through ``p0`` along ``h0`` meets the line through ``p3``
    along ``h3``, if it lies at least ``_CORNER_MARGIN`` beyond ``p0`` and
    before ``p3``; otherwise None.

    :param Tuple[float, float] p0: Starting map coordinate, in metres.
    :param float h0: Departure heading, in radians.
    :param Tuple[float, float] p3: Ending map coordinate, in metres.
    :param float h3: Arrival heading, in radians.
    :returns: The valid corner coordinate, or ``None`` if the lines are parallel
        or the intersection falls inside either endpoint margin.
    :rtype: Tuple[float, float] | None
    """
    u0, u3 = (cos(h0), sin(h0)), (cos(h3), sin(h3))
    determinant = u0[0] * u3[1] - u0[1] * u3[0]
    if abs(determinant) < 1e-9:
        return None
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    ahead = (dx * u3[1] - dy * u3[0]) / determinant
    behind = (u0[0] * dy - u0[1] * dx) / determinant
    if ahead < _CORNER_MARGIN or behind < _CORNER_MARGIN:
        return None
    return p0[0] + ahead * u0[0], p0[1] + ahead * u0[1]


def _too_tight(connector: VirtualLane, min_turn_radius: float) -> bool:
    """
    Whether any heading interval of the connector, pivots aside, is tighter than ``min_turn_radius``.

    :param VirtualLane connector: Virtual bicycle lane whose geometry is checked.
    :param float min_turn_radius: Minimum permitted turning radius, in metres.
    :returns: ``True`` if any heading interval of the connector, pivots aside, is tighter than ``min_turn_radius``; otherwise ``False``.
    :rtype: bool
    """
    return any(curvature > 1 / min_turn_radius + 1e-9
               for _, _, curvature in curvature_spans(connector.line, connector.pivots))
