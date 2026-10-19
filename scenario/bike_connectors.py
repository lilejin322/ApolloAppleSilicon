"""
Virtual connectors that carry bicycles across the gaps between bike lanes.

On the bundled maps each group of bike lanes is a single chain that ends at
a junction or a mid-road break and resumes beyond it. A connector joins a
dead end, a bike lane without a bike successor, to a start, a bike lane
without a bike predecessor. Pairs come from the car lanes beside the bike
lanes, whose topology runs through junctions, and from straight
continuations ahead. Each connector is a ``VirtualLane`` for the bicycles'
``RouteSpace``; the map itself is never changed.
"""

from enum import Enum
from math import ceil, cos, dist, hypot, radians, sin
from typing import Iterable, Iterator

import networkx as nx
from shapely.geometry import LineString

from ..map import MapService
from ..map.geometry import chord_heading, normalize_angle
from ..message.descriptor import DescriptorBundle
from .replay import curvature_spans
from .routing import VirtualLane


class ConnectorKind(Enum):
    """How a connector crosses: by the change of heading from dead end to start."""

    STRAIGHT = "STRAIGHT"
    RIGHT = "RIGHT"
    LEFT_TWO_STAGE = "LEFT_TWO_STAGE"
    """Straight to the far corner, a stop to turn in place, then straight across."""
    LEFT_DIAGONAL = "LEFT_DIAGONAL"
    """One smooth curve across the junction: a left turn ridden diagonally across the junction
    instead of in two stages, which riders do in practice."""


DEFAULT_KINDS = frozenset({ConnectorKind.STRAIGHT, ConnectorKind.RIGHT, ConnectorKind.LEFT_TWO_STAGE})

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
_SAMPLE_STEP = 0.5
"""Approximate spacing in metres of a curved connector's vertices."""
_CORNER_MARGIN = 1.0
"""Metres a two-stage corner must lie beyond the dead end and before the start."""


def bike_connectors(service: MapService, *, min_turn_radius: float,
                    kinds: Iterable[ConnectorKind] = DEFAULT_KINDS,
                    max_length: float = 100.0) -> tuple[VirtualLane, ...]:
    """
    Return the connectors between the map's bike lanes, sorted by ID.

    :param MapService service: A map with its routing graph built; it is only read.
    :param float min_turn_radius: Connectors tighter than this, in metres, are
        dropped; corners at pivots are turned at rest and not measured.
    :param kinds: The kinds of connector to generate.
    :param float max_length: Longest connector in metres.
    :raises RuntimeError: If the map was loaded without its routing graph.
    :rtype: tuple[VirtualLane, ...]
    """
    if service.routing_graph is None:
        raise RuntimeError("bike_connectors needs a map loaded with load_only=False")
    kinds = frozenset(kinds)
    types = DescriptorBundle(service.descriptor_path).message(
        "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name["type"].enum_type.values_by_name
    biking, city = types["BIKING"].number, types["CITY_DRIVING"].number
    lanes, graph = service.lane_table, service.routing_graph
    bikes = {lane_id for lane_id, lane in lanes.items() if lane.type == biking}

    def forward(lane_id: str) -> list[str]:
        return sorted(target for target in graph.successors(lane_id)
                      if graph[lane_id][target]["direction_type"] == "F")

    def car_neighbours(lane_id: str) -> list[str]:
        lane = lanes[lane_id]
        return [other.id for other in (*lane.left_neighbor_forward_lane_id, *lane.right_neighbor_forward_lane_id)
                if other.id in lanes and lanes[other.id].type == city]

    chains = nx.DiGraph()
    chains.add_nodes_from(bikes)
    chains.add_edges_from((lane_id, target) for lane_id in bikes for target in forward(lane_id) if target in bikes)
    chain = {lane_id: index for index, group in enumerate(nx.weakly_connected_components(chains))
             for lane_id in group}
    ends = sorted(lane_id for lane_id in bikes if chains.out_degree(lane_id) == 0)
    starts = sorted(lane_id for lane_id in bikes if chains.in_degree(lane_id) == 0)

    pairs = set()
    beside = {}
    for start in starts:
        for car in car_neighbours(start):
            beside.setdefault(car, []).append(start)
    for end in ends:
        frontier = set(car_neighbours(end))
        for _ in range(_TOPOLOGY_STEPS):
            frontier = {target for car in frontier for target in forward(car) if lanes[target].type == city}
            pairs.update((end, start) for car in frontier for start in beside.get(car, ()))
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

    connectors = []
    for end, start in sorted(pairs):
        if chain[end] != chain[start]:
            connectors.extend(connector for connector in _connect(end, poses[end], start, firsts[start], kinds)
                              if connector.line.length <= max_length
                              and not _too_tight(connector, min_turn_radius)
                              and not any(not mark.virtual and "CURB" in mark.types
                                          for mark in service.get_lane_markings(connector.line)))
    return tuple(sorted(connectors, key=lambda connector: connector.lane_id))


Pose = tuple[tuple[float, float], float]
"""A point and a heading in radians."""


def _end_pose(line: LineString) -> Pose:
    """The last point of a lane and its heading over the last ``_HEADING_CHORD`` metres."""
    return line.coords[-1][:2], chord_heading(line, max(0.0, line.length - _HEADING_CHORD), line.length)


def _start_pose(line: LineString) -> Pose:
    """The first point of a lane and its heading over the first ``_HEADING_CHORD`` metres."""
    return line.coords[0][:2], chord_heading(line, 0.0, min(_HEADING_CHORD, line.length))


def _connect(end: str, end_pose: Pose, start: str, start_pose: Pose,
             kinds: frozenset[ConnectorKind]) -> Iterator[VirtualLane]:
    """The connectors of the allowed kinds from ``end`` to ``start``, before length and radius checks."""
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


def _bezier(p0: tuple[float, float], h0: float, p3: tuple[float, float], h3: float) -> LineString:
    """A cubic Bézier curve from ``p0`` to ``p3``, leaving along ``h0`` and arriving along ``h3``."""
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


def _corner(p0: tuple[float, float], h0: float, p3: tuple[float, float], h3: float,
            ) -> tuple[float, float] | None:
    """
    Where the line through ``p0`` along ``h0`` meets the line through ``p3``
    along ``h3``, if it lies at least ``_CORNER_MARGIN`` beyond ``p0`` and
    before ``p3``; otherwise None.
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
    """Whether any heading interval of the connector, pivots aside, is tighter than ``min_turn_radius``."""
    return any(curvature > 1 / min_turn_radius + 1e-9
               for _, _, curvature in curvature_spans(connector.line, connector.pivots))
