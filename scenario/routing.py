"""
Lane-route chromosomes for the ego vehicle and NPCs.

A route is an ordered sequence of lanes connected by forward steps or lane
changes, with a start ``s`` on the first lane and an end ``s`` on the last.
Routes are immutable, so crossover and mutation produce new ones. A
``RouteSpace`` decides which routes are legal for one agent type on one map
and converts them into the poses Apollo receives. Agents Apollo never routes
may also use ``VirtualLane``s, lanes the map lacks that exist only in their
``RouteSpace``.
"""

import itertools
import weakref
import numpy as np
import networkx as nx
from enum import Enum
from random import Random
from dataclasses import dataclass
from math import ceil, cos, dist, isfinite, sin
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping, NamedTuple, Dict, FrozenSet, List, Tuple, Self
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from ..map import MapService
from ..map.geometry import max_replay_curvature, polygon_to_shape
from ..message.descriptor import DescriptorBundle
from ..utils import require_nonnegative

_WAYPOINT_STEP = 1.0
"""Spacing in metres of the candidate waypoint positions tried on a lane."""
_BLEND_STEPS = 20
"""Minimum points per lane change when blending a path from the source to the target lane."""
_BLEND_SPACING = 1.0
"""Largest metres between blend points of a long lane change."""
_LANDING_MARGIN = 1.0
"""Metres a longer lane change must land before the end of its target-lane stretch."""
_ABEAM_TOLERANCE = 0.5
"""Metres a lane change's start may lie ahead of or behind its projection on the target lane."""
_CHANGE_STEP = 5.0
"""Metres a desired lane-change length is shortened by until it fits."""
_CHANGE_SAMPLES = 64
"""Maximum draws while narrowing a lane-change window around a feasible continuation."""
_JOIN_REACH = 10.0
"""Metres of each lane on either side of a forward step checked for the step's curvature."""
_CURVATURE_CACHE: "weakref.WeakKeyDictionary[MapService, Dict]" = weakref.WeakKeyDictionary()
"""Per map: replayed curvature of each lane and of each forward step between lanes."""
_JUNCTION_CACHE: "weakref.WeakKeyDictionary[MapService, Dict]" = weakref.WeakKeyDictionary()
"""Per map: the union of its junction polygons, its junction lanes, and endpoint ranges per lane and margin."""
ENDPOINT_JUNCTION_MARGIN = 5.0
"""Default ``RouteRules.endpoint_junction_margin`` of ego and NPC rules, in metres."""
JUNCTION_SHARE = 0.5
"""A lane whose centerline lies more than this share inside junction polygons is a junction lane."""
WALKS_PER_TARGET = 10
"""Random walks ``random_route`` tries per drawn target length before drawing another."""
SHORT_WALK_SHARE = 0.8
"""Share of its target length below which ``random_route`` retries a walk (dead ends)."""
_ENDPOINT_STEP = 0.5
"""Metres between the positions checked when finding where on a lane a route may start or end."""
_ENDPOINT_OVERSHOOT = 100.0
"""Metres a random walk may drive past its target length to find an allowed end."""


def _lane_curvature(service: MapService, lane_id: str) -> float:
    """
    Return the cached replay curvature of one map lane.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :param str lane_id: Map lane ID.
    :returns: Maximum replay curvature of the lane, in inverse metres.
    :rtype: float
    """
    cache = _CURVATURE_CACHE.setdefault(service, {})
    key = ("lane", lane_id)
    if key not in cache:
        cache[key] = max_replay_curvature(service.get_lane_central_curve_by_id(lane_id).coords)
    return cache[key]


def _join_curvature(service: MapService, source: str, target: str) -> float:
    """
    Curvature of the last metres of ``source`` joined to the first metres of ``target``.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :param str source: Source lane ID.
    :param str target: Target lane ID.
    :returns: Maximum replay curvature across the source-to-target join, in inverse metres.
    :rtype: float
    """
    cache = _CURVATURE_CACHE.setdefault(service, {})
    key = ("join", source, target)
    if key not in cache:
        before = service.get_lane_central_curve_by_id(source)
        after = service.get_lane_central_curve_by_id(target)
        tail = substring(before, max(before.length - _JOIN_REACH, 0.0), before.length)
        head = substring(after, 0.0, min(_JOIN_REACH, after.length))
        points = [point[:2] for point in tail.coords] + [point[:2] for point in head.coords]
        cache[key] = max_replay_curvature(points)
    return cache[key]


def offset_line(line: LineString, offset: float) -> LineString:
    """
    ``line`` shifted ``offset`` metres to its right (left if negative), in the
    same direction: each vertex moves along the normal of its neighbours'
    chord, so vertices and their order are kept (shapely's ``offset_curve``
    may merge or reorder them on tight bends).

    :param LineString line: Line geometry in map metres.
    :param float offset: Signed lateral shift in metres; positive shifts right.
    :returns: The constructed line geometry in map metres.
    :rtype: LineString
    """
    coords = np.asarray(line.coords, dtype=float)[:, :2]
    if len(coords) < 2 or not offset:
        return line
    tangents = np.empty_like(coords)
    tangents[1:-1] = coords[2:] - coords[:-2]
    tangents[0], tangents[-1] = coords[1] - coords[0], coords[-1] - coords[-2]
    tangents /= np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-12)[:, None]
    right = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    shifted = coords + offset * right
    keep = np.r_[True, np.hypot(*np.diff(shifted, axis=0).T) > 1e-9]
    return LineString(shifted[keep])


def junction_lanes(service: MapService) -> FrozenSet[str]:
    """
    The map's junction lanes (``RouteSpace.is_junction_lane``), without building a space.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :returns: IDs of lanes classified as junction lanes.
    :rtype: FrozenSet[str]
    """
    lane_fields = DescriptorBundle(service.descriptor_path).message("apollo.hdmap.Lane").DESCRIPTOR.fields_by_name
    return frozenset(_junction_info(service, lane_fields["turn"].enum_type.values_by_name["NO_TURN"].number)["lanes"])


def _junction_info(service: MapService, no_turn: int) -> Dict:
    """
    The map's junction polygons as one shape (None without polygons) and its
    junction lanes: lanes with a ``junction_id``, lanes mostly inside junction
    polygons, and turn lanes that enter one (maps often leave ``junction_id`` unset).

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :param int no_turn: Apollo enum number identifying a lane without a turn.
    :returns: The map's junction polygons as one shape (None without polygons) and its junction lanes: lanes with a ``junction_id``, lanes mostly inside junction polygons, and turn lanes that enter one (maps often leave ``junction_id`` unset).
    :rtype: Dict
    """
    info = _JUNCTION_CACHE.get(service)
    if info is None:
        shapes = []
        for junction in service.junction_table.values():
            try:
                shape = polygon_to_shape(junction.polygon)
            except ValueError:
                continue
            if shape is not None:
                shape = shape if shape.is_valid else shape.buffer(0)
                if not shape.is_empty:
                    shapes.append(shape)
        union = unary_union(shapes) if shapes else None
        if union is not None:
            shapely.prepare(union)
        lanes = set()
        for lane_id, lane in service.lane_table.items():
            if lane.HasField("junction_id"):
                lanes.add(lane_id)
                continue
            if union is None:
                continue
            line = service.get_lane_central_curve_by_id(lane_id)
            if not union.intersects(line):
                continue
            inside = line.intersection(union).length
            if inside > JUNCTION_SHARE * line.length or (lane.turn != no_turn and inside > 0):
                lanes.add(lane_id)
        info = {"union": union, "lanes": frozenset(lanes), "ranges": {}}
        _JUNCTION_CACHE[service] = info
    return info


def _quintic(u: float) -> float:
    """
    Minimum-jerk blend weight: 0 to 1 with zero slope and curvature at both ends.

    :param float u: Normalized blend position in [0, 1].
    :returns: Minimum-jerk blend weight in [0, 1].
    :rtype: float
    """
    return u * u * u * (10 - 15 * u + 6 * u * u)


def _ordered_subgraph(graph: nx.DiGraph, lanes: Iterable[str]) -> nx.DiGraph:
    """
    Copy allowed nodes and edges in a stable order, including equal-cost search ties.

    :param nx.DiGraph graph: Directed lane-routing graph.
    :param Iterable[str] lanes: Lane IDs to retain.
    :returns: A copy containing the requested nodes and edges in stable order.
    :rtype: nx.DiGraph
    """
    allowed = set(lanes)
    ordered = nx.DiGraph()
    ordered.add_nodes_from((lane, graph.nodes[lane]) for lane in sorted(allowed))
    ordered.add_edges_from(
        (source, target, graph[source][target])
        for source in sorted(allowed) for target in sorted(graph.successors(source))
        if target in allowed
    )
    return ordered


class InvalidRoute(ValueError):
    """
    The route cannot be driven or sent unambiguously on this map.
    """


class Transition(Enum):
    """
    How a route moves from one lane to the next, as in ``routing_graph``.
    """

    FORWARD = "F"
    LEFT = "L"
    RIGHT = "R"


def lane_changes(transitions: Iterable[Transition]) -> int:
    """
    Count the lane changes among route transitions.

    :param Iterable[Transition] transitions: Transitions connecting consecutive route lanes.
    :returns: The number of non-forward transitions.
    :rtype: int
    """
    return sum(step is not Transition.FORWARD for step in transitions)


@dataclass(frozen=True)
class LaneRoute:
    """
    A route along lanes, from ``start_s`` on the first lane to ``end_s`` on the last.

    ``transitions[i]`` connects ``lanes[i]`` to ``lanes[i + 1]``. Where a lane
    change happens along a lane is decided by Apollo's planning, not here.

    :param Tuple[str, ...] lanes: Lane IDs in driving order.
    :param Tuple[Transition, ...] transitions: One step per consecutive lane pair.
    :param float start_s: Start position in metres along the first lane's centerline.
    :param float end_s: End position in metres along the last lane's centerline.
    """
    lanes: Tuple[str, ...]
    """Lane IDs in driving order."""
    transitions: Tuple[Transition, ...]
    """One step per consecutive lane pair."""
    start_s: float
    """Start position in metres along the first lane's centerline."""
    end_s: float
    """End position in metres along the last lane's centerline."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :raises ValueError: If the route has no lanes, the transition count is wrong, or an endpoint is not finite.
        """
        if not self.lanes:
            raise ValueError("a route needs at least one lane")
        if len(self.transitions) != len(self.lanes) - 1:
            raise ValueError("transitions must have one entry per consecutive lane pair")
        if not (isfinite(self.start_s) and isfinite(self.end_s)):
            raise ValueError("start_s and end_s must be finite")

    @property
    def lane_changes(self) -> int:
        """
        The number of left and right lane changes.

        :returns: The number of non-forward transitions.
        :rtype: int
        """
        return len(self.lane_change_indices)

    @property
    def lane_change_indices(self) -> Tuple[int, ...]:
        """
        The positions in ``transitions`` of the lane changes, in route order.

        :returns: Indices of the non-forward transitions, in route order.
        :rtype: Tuple[int, ...]
        """
        return tuple(index for index, step in enumerate(self.transitions) if step is not Transition.FORWARD)

    @property
    def lane_change_pairs(self) -> Tuple[Tuple[str, str], ...]:
        """
        The (source lane, target lane) of each lane change, in route order.

        :returns: Source/target lane-ID pairs for the lane changes, in route order.
        :rtype: Tuple[Tuple[str, str], ...]
        """
        return tuple((self.lanes[index], self.lanes[index + 1]) for index in self.lane_change_indices)

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A new JSON-compatible mapping of the object fields.
        :rtype: Dict
        """
        return {
            "lanes": list(self.lanes),
            "transitions": [transition.value for transition in self.transitions],
            "start_s": self.start_s,
            "end_s": self.end_s,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Build a route from ``to_dict`` output.

        :param Dict data: Mapping with lanes, transitions, start_s, and end_s.
        :returns: The reconstructed object.
        :rtype: LaneRoute
        :raises KeyError: If a required serialized field is absent.
        :raises ValueError: If a transition value or reconstructed route is invalid.
        """
        return cls(
            tuple(data["lanes"]),
            tuple(Transition(value) for value in data["transitions"]),
            data["start_s"],
            data["end_s"],
        )


@dataclass(frozen=True)
class RoutePose:
    """
    A position on a lane with its map coordinates and lane heading in radians.

    :param str lane_id: Map or virtual lane ID.
    :param float s: Position along the lane, in metres.
    :param float x: Map X coordinate, in metres.
    :param float y: Map Y coordinate, in metres.
    :param float heading: Heading in radians.
    """
    lane_id: str
    """Map or virtual lane ID."""
    s: float
    """Position along the lane, in metres."""
    x: float
    """Map X coordinate, in metres."""
    y: float
    """Map Y coordinate, in metres."""
    heading: float
    """Heading in radians."""

    @property
    def point(self) -> Point:
        """
        The position as a Shapely Point.

        :returns: The map position as a Shapely point.
        :rtype: Point
        """
        return Point(self.x, self.y)


@dataclass(frozen=True)
class RoutePoses:
    """
    The poses sent to Apollo for one route: start, intermediate waypoints, and end.

    :param RoutePose start: Route start pose.
    :param Tuple[RoutePose, ...] waypoints: Intermediate route waypoints.
    :param RoutePose end: Route end pose.
    """
    start: RoutePose
    """Route start pose."""
    waypoints: Tuple[RoutePose, ...]
    """Intermediate route waypoints."""
    end: RoutePose
    """Route end pose."""


class RouteLeg(NamedTuple):
    """
    The stretch ``[start_s, end_s]`` of one lane that a route drives along.

    :param str lane_id: Map or virtual lane ID.
    :param float start_s: Start position along the lane, in metres.
    :param float end_s: End position along the lane, in metres.
    """
    lane_id: str
    """Map or virtual lane ID."""
    start_s: float
    """Start position along the lane, in metres."""
    end_s: float
    """End position along the lane, in metres."""


class PathSection(NamedTuple):
    """
    A stretch of a route's path and the lanes it lies on: one lane, or a
    lane change's source and target lane.

    :param Tuple[str, ...] lanes: Lane IDs spanned by the section, in travel order.
    :param LineString line: Line geometry in map metres.
    """
    lanes: Tuple[str, ...]
    """Lane IDs spanned by the section, in travel order."""
    line: LineString
    """Line geometry in map metres."""


@dataclass(frozen=True)
class VirtualLane:
    """
    A lane the map lacks, joining map lanes for one agent type, such as a
    bicycle crossing a junction from one bike lane to the next.

    ``RouteSpace`` adds it with forward edges from each predecessor and to
    each successor. At a pivot the line has a corner that is only drivable at
    rest: agents stop there, turn, and go on.

    :param str lane_id: Must differ from every map lane ID.
    :param LineString line: Centerline in driving direction, without zero-length segments.
    :param Tuple[str, ...] predecessors: Lanes it continues forward from.
    :param Tuple[str, ...] successors: Lanes it leads forward to.
    :param Tuple[float, ...] pivots: Metres along ``line`` of its corner vertices,
        strictly increasing within ``(0, length)``.
    """
    lane_id: str
    """Must differ from every map lane ID."""
    line: LineString
    """Centerline in driving direction, without zero-length segments."""
    predecessors: Tuple[str, ...] = ()
    """Lanes it continues forward from."""
    successors: Tuple[str, ...] = ()
    """Lanes it leads forward to."""
    pivots: Tuple[float, ...] = ()
    """Metres along ``line`` of its corner vertices, strictly increasing within ``(0, length)``."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :raises ValueError: If the lane ID or line is invalid, or pivots are non-finite, out of order, outside the line, or not at line vertices.
        """
        if not self.lane_id:
            raise ValueError("a virtual lane needs an ID")
        if not self.line.length > 0:
            raise ValueError(f"virtual lane {self.lane_id!r} needs a line of positive length")
        previous = 0.0
        vertices = [self.line.project(Point(point)) for point in self.line.coords]
        for pivot in self.pivots:
            if not (isfinite(pivot) and previous < pivot < self.line.length):
                raise ValueError(f"pivots of virtual lane {self.lane_id!r} must increase strictly "
                                 f"within (0, {self.line.length})")
            if min(abs(pivot - vertex) for vertex in vertices) > 1e-9:
                raise ValueError(f"pivot {pivot} of virtual lane {self.lane_id!r} is not at a vertex")
            previous = pivot


@dataclass(frozen=True)
class RouteRules:
    """
    Which lanes and steps an agent may use, and how routes are sent.

    :param FrozenSet[str] lane_types: Allowed ``Lane.type`` names.
    :param float min_turn_radius: U-turn lanes tighter than this, in metres, are excluded.
    :param float min_lane_change_distance: Metres of a lane consumed by a lane change.
    :param float ambiguity_margin: No other lane may pass within this many metres of a sent pose.
    :param bool unambiguous: Whether ``validate`` requires unambiguous poses. True for
        routes sent to Apollo (the ego); NPC routes are never matched to lanes by Apollo.
    :param bool prune_tight_turns: Also exclude every lane, and every forward step
        between lanes, whose replayed curvature is tighter than ``min_turn_radius``,
        not only U-turn lanes. For vehicles that replay lane geometry (NPCs): a
        random walk then never enters a turn the vehicle cannot drive.
    :param float | None endpoint_junction_margin: If set, routes must start and
        end off junction lanes (``RouteSpace.is_junction_lane``), at least this
        many metres from every junction polygon, and at least this far from a
        lane end that joins a junction lane. None allows any endpoint.
    """
    lane_types: FrozenSet[str] = frozenset({"CITY_DRIVING"})
    """Allowed ``Lane.type`` names."""
    min_turn_radius: float = 0.0
    """U-turn lanes tighter than this, in metres, are excluded."""
    min_lane_change_distance: float = 20.0
    """Metres of a lane consumed by a lane change."""
    ambiguity_margin: float = 0.5
    """No other lane may pass within this many metres of a sent pose."""
    unambiguous: bool = True
    """Whether ``validate`` requires unambiguous poses. True for routes sent to Apollo (the ego); NPC routes are never matched to lanes by Apollo."""
    prune_tight_turns: bool = False
    """Also exclude every lane, and every forward step between lanes, whose replayed curvature is tighter than ``min_turn_radius``, not only U-turn lanes. For vehicles that replay lane geometry (NPCs): a random walk then never enters a turn the vehicle cannot drive."""
    endpoint_junction_margin: float | None = None
    """If set, routes must start and end off junction lanes (``RouteSpace.is_junction_lane``), at least this many metres from every junction polygon, and at least this far from a lane end that joins a junction lane. None allows any endpoint."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :raises ValueError: If a configured distance is negative, or ``ambiguity_margin`` is not finite and positive.
        """
        require_nonnegative(min_turn_radius=self.min_turn_radius,
                            min_lane_change_distance=self.min_lane_change_distance)
        if self.endpoint_junction_margin is not None:
            require_nonnegative(endpoint_junction_margin=self.endpoint_junction_margin)
        # A zero radius would make get_lanes() enumerate every lane.
        if not (isfinite(self.ambiguity_margin) and self.ambiguity_margin > 0):
            raise ValueError("ambiguity_margin must be finite and positive")


def turn_radius(line: LineString) -> float:
    """
    Apollo's U-turn radius estimate: the circle through a centerline's first,
    middle, and last points (``GraphCreator::IsValidUTurn``).

    :param LineString line: Lane centerline.
    :returns: Radius in metres, or infinity for collinear points.
    :rtype: float
    """
    points = list(line.coords)
    a, b, c = points[0], points[len(points) // 2], points[-1]
    twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if twice_area == 0:
        return float("inf")  # Collinear: Apollo treats it as a wide U-turn.
    return dist(a, b) * dist(b, c) * dist(c, a) / (2 * twice_area)


class RouteSpace:
    """
    Legal routes for one agent type on one loaded map.

    Virtual lanes join the allowed map lanes with forward edges; edges to or
    from lanes the rules exclude are dropped. They are added after
    ``prune_tight_turns``, so whoever builds them checks their curvature.
    Routes may pass through virtual lanes but never start or end on one.

    :param MapService service: A map with its lane index and routing graph built.
    :param RouteRules rules: Allowed lanes, lane-change distance, and pose ambiguity margin.
    :param Iterable[VirtualLane] virtual_lanes: Lanes the map lacks; only for rules
        with ``unambiguous=False``, since Apollo cannot route through them.
    :param Iterable[str] extra_lanes: Map lanes allowed besides those of
        ``rules.lane_types``, such as the curb lanes bicycles may ride
        (``bicycle.curb_lanes``). Not pruned for tight turns.
    :param Mapping[str, float] lane_offsets: Metres to shift an allowed map lane's
        centerline to its right (``centerline``): a bicycle rides near the curb
        of a driving lane. Lengths, positions, endpoints and replay paths all use
        the shifted line.
    """
    service: MapService
    """Loaded map service used by this route space."""
    rules: RouteRules
    """Route rules enforced by this space."""
    _offsets: Dict[str, float]
    """Nonzero rightward centerline offsets by map-lane ID."""
    _offset_lines: Dict[str, LineString]
    """Lazily computed shifted centerlines."""
    _virtual: Dict[str, VirtualLane]
    """Virtual lanes by ID."""
    _no_turn: int
    """Apollo enum number for lanes with no turn classification."""
    _lengths: Dict[str, float]
    """Centerline lengths of allowed map and virtual lanes."""
    graph: nx.DiGraph
    """Allowed routing graph with transition and search-cost attributes."""
    _start_lane_list: List[str]
    """Lazily cached sorted map lanes on which a route may start."""

    def __init__(self, service: MapService, rules: RouteRules = RouteRules(),
                 virtual_lanes: Iterable[VirtualLane] = (), *, extra_lanes: Iterable[str] = (),
                 lane_offsets: Mapping[str, float] = MappingProxyType({})) -> None:
        """
        Build a route graph under vehicle rules, including virtual lanes and lateral offsets.

        :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
        :param RouteRules rules: Route validation and geometry rules.
        :param Iterable[VirtualLane] virtual_lanes: Additional immutable lane geometries and their connections.
        :param Iterable[str] extra_lanes: Additional map lanes admitted beyond the configured lane type.
        :param Mapping[str, float] lane_offsets: Rightward centreline shifts by lane ID, in metres.
        :raises RuntimeError: If the map was loaded without its routing graph or lane index.
        :raises ValueError: If lane types are unknown, a virtual lane conflicts or joins an unknown lane, virtual lanes are incompatible with the rules, or an extra/offset lane is unknown.
        """
        if service.routing_graph is None or service.lane_index is None:
            raise RuntimeError("RouteSpace needs a map loaded with load_only=False")
        self.service = service
        self.rules = rules
        extra = frozenset(extra_lanes)
        unknown = sorted((extra | set(lane_offsets)) - set(service.lane_table))
        if unknown:
            raise ValueError(f"unknown extra or offset lanes {unknown[:5]}")
        self._offsets = {lane_id: float(offset) for lane_id, offset in lane_offsets.items() if offset}
        self._offset_lines: Dict[str, LineString] = {}
        self._virtual: Dict[str, VirtualLane] = {}
        for lane in virtual_lanes:
            if lane.lane_id in service.lane_table or lane.lane_id in self._virtual:
                raise ValueError(f"virtual lane ID {lane.lane_id!r} is already in use")
            self._virtual[lane.lane_id] = lane
        if self._virtual and rules.unambiguous:
            raise ValueError("virtual lanes need rules with unambiguous=False: "
                             "Apollo cannot route through lanes its map lacks")
        for lane in self._virtual.values():
            unknown = [other for other in lane.predecessors + lane.successors
                       if other not in service.lane_table and other not in self._virtual]
            if unknown:
                raise ValueError(f"virtual lane {lane.lane_id!r} joins unknown lanes {unknown}")
        lane_fields = DescriptorBundle(service.descriptor_path).message(
            "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name
        types = lane_fields["type"].enum_type.values_by_name
        unknown = rules.lane_types - set(types)
        if unknown:
            raise ValueError(f"unknown lane type {sorted(unknown)}")
        allowed_types = {types[name].number for name in rules.lane_types}
        u_turn = lane_fields["turn"].enum_type.values_by_name["U_TURN"].number
        self._no_turn = lane_fields["turn"].enum_type.values_by_name["NO_TURN"].number
        allowed = {
            lane_id for lane_id, lane in service.lane_table.items()
            if lane.type in allowed_types and not (
                lane.turn == u_turn
                and turn_radius(service.get_lane_central_curve_by_id(lane_id)) < rules.min_turn_radius
            )
        }
        prune = rules.prune_tight_turns and rules.min_turn_radius > 0
        if prune:
            limit = 1 / rules.min_turn_radius
            allowed = {lane_id for lane_id in allowed if _lane_curvature(service, lane_id) <= limit}
        allowed |= extra
        self._lengths = {lane_id: self.centerline(lane_id).length for lane_id in allowed}
        # A concrete copy with precomputed costs: path searches avoid view
        # filters and per-edge geometry calls.
        self.graph = _ordered_subgraph(service.routing_graph, allowed)
        if prune:
            self.graph.remove_edges_from([
                (source, target) for source, target, data in list(self.graph.edges(data=True))
                if data["direction_type"] == Transition.FORWARD.value
                and _join_curvature(service, source, target) > limit
            ])
        if self._virtual:
            self._lengths.update((lane_id, lane.line.length) for lane_id, lane in self._virtual.items())
            topology = nx.DiGraph(self.graph)
            for lane_id, lane in self._virtual.items():
                topology.add_node(lane_id)
                topology.add_edges_from(((source, lane_id) for source in lane.predecessors), direction_type="F")
                topology.add_edges_from(((lane_id, target) for target in lane.successors), direction_type="F")
            self.graph = _ordered_subgraph(topology, allowed | set(self._virtual))
        for _, _, data in self.graph.edges(data=True):
            data["transition"] = Transition(data["direction_type"])
        lateral_targets = {v for _, v, data in self.graph.edges(data=True)
                           if data["transition"] is not Transition.FORWARD}
        for source, _, data in self.graph.edges(data=True):
            forward = data["transition"] is Transition.FORWARD
            data["cost"] = self._lengths[source] if forward else rules.min_lane_change_distance
            # A lane entered by a lane change may be left forward after any part of it.
            data["distance_lower_bound"] = 0.0 if forward and source in lateral_targets else data["cost"]

    def validate(self, route: LaneRoute, *, lane_change_s: Tuple[float, ...] | None = None) -> None:
        """
        Check that the route is legal and, if the rules require it, starts and
        ends away from junctions and is unambiguous to send.

        :param LaneRoute route: The route to check.
        :param Tuple[float, ...] | None lane_change_s: Where each lane change starts on
            its source lane, in route order. By default each starts as soon as the
            route enters the source lane; Apollo's planning decides for the ego.
        :raises InvalidRoute: If route topology, endpoint restrictions, lane-change placement, or pose ambiguity checks fail.
        """
        self._check(route, lane_change_s)
        self._check_endpoints(route)
        if self.rules.unambiguous:
            self.poses(route)

    def is_valid(self, route: LaneRoute, *, lane_change_s: Tuple[float, ...] | None = None) -> bool:
        """
        Return whether ``validate`` accepts the route.

        :param LaneRoute route: Route to validate.
        :param Tuple[float, ...] | None lane_change_s: As in ``validate``.
        :returns: ``True`` if validation accepts the route; ``False`` if it raises ``InvalidRoute``.
        :rtype: bool
        """
        try:
            self.validate(route, lane_change_s=lane_change_s)
        except InvalidRoute:
            return False
        return True

    def legs(self, route: LaneRoute, *,
             lane_change_s: Tuple[float, ...] | None = None) -> List[RouteLeg]:
        """
        Return the stretch of each lane the route drives along.

        A leg ends at the lane's end before a forward step, where the lane
        change starts before a lane change, and at ``end_s`` on the last lane.
        A lane change covers ``min_lane_change_distance`` of the source lane
        after its leg, landing where the next leg starts.

        :param LaneRoute route: The route.
        :param Tuple[float, ...] | None lane_change_s: As in ``validate``.
        :returns: Driven lane stretches in route order.
        :rtype: List[RouteLeg]
        :raises InvalidRoute: If the route is illegal.
        """
        lengths, entries, changes = self._check(route, lane_change_s)
        ends = [lengths[index] if change is None else change for index, change in enumerate(changes)]
        return [RouteLeg(lane, start, end)
                for lane, start, end in zip(route.lanes, entries, ends + [route.end_s])]

    def path_sections(self, route: LaneRoute, *,
                      lane_change_s: Tuple[float, ...] | None = None,
                      lane_change_distances: Tuple[float, ...] | None = None) -> List[PathSection]:
        """
        Return the line of ``path`` in sections by lane.

        Each stretch of a lane is a one-lane section. Each lane change is a
        section on its source and target lane, from the end of the source
        stretch to the landing point on the target lane. Consecutive sections
        share their boundary point; the gap between one lane's end and the next
        lane's start belongs to the later section.

        :param LaneRoute route: The route.
        :param Tuple[float, ...] | None lane_change_s: As in ``validate``.
        :param Tuple[float, ...] | None lane_change_distances: Desired length in
            metres of each lane change, in route order. Each is cut to the room
            the source lane and the target lane's stretch leave, and is never
            shorter than ``min_lane_change_distance``, the default. A longer
            change lands further along the target lane.
        :returns: Path sections in route order.
        :rtype: List[PathSection]
        :raises InvalidRoute: If the route is illegal.
        :raises ValueError: If ``lane_change_distances`` does not contain one value per lane change.
        """
        legs = self.legs(route, lane_change_s=lane_change_s)
        minimum = self.rules.min_lane_change_distance
        if lane_change_distances is not None and len(lane_change_distances) != route.lane_changes:
            raise ValueError(f"lane_change_distances needs one length per lane change "
                             f"({route.lane_changes}), got {len(lane_change_distances)}")
        desired = iter(lane_change_distances or ())
        sections: List[PathSection] = []
        last: Tuple[float, float] | None = None

        def add(lanes: Tuple[str, ...], points: Iterable[Tuple[float, float]]) -> None:
            """
            Append a nondegenerate path section and join its first vertex to the preceding section.

            :param Tuple[str, ...] lanes: Lane IDs in driving order.
            :param Iterable[Tuple[float, float]] points: Ordered polyline vertices in map metres.
            """
            nonlocal last
            line = [] if last is None else [last]
            for point in points:
                if not line or line[-1] != point:
                    line.append(point)
            if len(line) > 1:
                sections.append(PathSection(lanes, LineString(line)))
            last = line[-1]

        resume = 0
        for index, leg in enumerate(legs):
            if index < resume:
                continue                                      # Covered by a long lane change.
            leg = legs[index]
            line = self.centerline(leg.lane_id)
            if leg.end_s > leg.start_s:
                add((leg.lane_id,), [point[:2] for point in substring(line, leg.start_s, leg.end_s).coords])
            else:
                start = line.interpolate(leg.start_s)
                add((leg.lane_id,), [(start.x, start.y)])
            if index < len(legs) - 1 and route.transitions[index] is not Transition.FORWARD:
                wanted = next(desired, minimum)
                if wanted > minimum:
                    change = self._corridor_change(route, legs, index, wanted)
                    if change is not None:
                        points, landed, landing = change
                        legs[landed] = legs[landed]._replace(start_s=landing)
                        add(tuple(leg_.lane_id for leg_ in legs[index:landed + 1]), points)
                        resume = landed
                        continue
                target_id = legs[index + 1].lane_id
                target = self.centerline(target_id)
                distance, landing = minimum, legs[index + 1].start_s
                if wanted > minimum:
                    distance, landing = self._longer_change(line, leg.end_s, target, landing,
                                                            legs[index + 1].end_s, wanted)
                    legs[index + 1] = legs[index + 1]._replace(start_s=landing)
                steps = max(_BLEND_STEPS, ceil(distance / _BLEND_SPACING))
                blend = []
                for step in range(1, steps):
                    u = step / steps
                    weight = _quintic(u)
                    a = line.interpolate(leg.end_s + u * distance)
                    b = target.interpolate(max(landing - (1 - u) * distance, 0.0))
                    blend.append(((1 - weight) * a.x + weight * b.x, (1 - weight) * a.y + weight * b.y))
                end = target.interpolate(landing)
                add((leg.lane_id, target_id), blend + [(end.x, end.y)])
        return sections

    def _corridor_change(self, route: LaneRoute, legs: List[RouteLeg], index: int, wanted: float,
                         ) -> Tuple[List[Tuple[float, float]], int, float] | None:
        """
        A lane change from the end of ``legs[index]`` that offsets laterally
        from the target side's centerline, along the target lane and the
        forward lanes after it up to the next lane change or the route end.
        Long changes need no room on the source lane and may land on a later
        lane than the target. Returns the blend points (landing included),
        the index of the leg it lands on and the landing s there; None if the
        change start is not abeam the target lane or no change longer than
        the minimum fits.

        :param LaneRoute route: Route containing the lane change.
        :param List[RouteLeg] legs: Placed route legs with lane IDs and driven position intervals.
        :param int index: Index of the source leg.
        :param float wanted: Requested lane-change length, in metres.
        :returns: Blend points including the landing, landing-leg index, and landing ``s``; ``None`` if no suitable longer change fits.
        :rtype: Tuple[List[Tuple[float, float]], int, float] | None
        """
        minimum = self.rules.min_lane_change_distance
        source = self.service.get_lane_central_curve_by_id(legs[index].lane_id)
        start = source.interpolate(legs[index].end_s)
        target = self.service.get_lane_central_curve_by_id(legs[index + 1].lane_id)
        t0 = target.project(start)
        if t0 <= 0 or t0 >= legs[index + 1].end_s:
            return None                                       # Not abeam: the target lane starts later.
        pieces: List[Tuple[int, float, float, LineString]] = []   # (leg, start s, end s, line)
        last = index + 1
        while True:
            low = t0 if last == index + 1 else legs[last].start_s
            if legs[last].end_s > low:
                lane = self.service.get_lane_central_curve_by_id(legs[last].lane_id)
                pieces.append((last, low, legs[last].end_s, substring(lane, low, legs[last].end_s)))
            if last + 1 >= len(legs) or route.transitions[last] is not Transition.FORWARD:
                break
            if not self._straight_ahead(legs[last + 1].lane_id):
                break                                         # Never change lanes through a junction or turn.
            last += 1
        corridor_points: List[Tuple[float, float]] = []
        for _, _, _, piece in pieces:
            for point in piece.coords:
                if not corridor_points or corridor_points[-1] != point[:2]:
                    corridor_points.append(point[:2])
        if len(corridor_points) < 2:
            return None
        corridor = LineString(corridor_points)
        distance = min(wanted, corridor.length - _LANDING_MARGIN)
        if distance <= minimum:
            return None
        base = corridor.interpolate(0.0)
        ahead = corridor.interpolate(min(1.0, corridor.length))
        tangent = (ahead.x - base.x, ahead.y - base.y)
        norm = (tangent[0] ** 2 + tangent[1] ** 2) ** 0.5
        if norm == 0:
            return None
        tangent = (tangent[0] / norm, tangent[1] / norm)
        offset = (start.x - base.x, start.y - base.y)
        lateral = tangent[0] * offset[1] - tangent[1] * offset[0]
        if abs(tangent[0] * offset[0] + tangent[1] * offset[1]) > _ABEAM_TOLERANCE:
            return None
        from ..map.geometry import CenterlineProfile
        profile = CenterlineProfile(corridor)
        steps = max(_BLEND_STEPS, ceil(distance / _BLEND_SPACING))
        points = []
        for step in range(1, steps + 1):
            u = step / steps
            q = u * distance
            centre = corridor.interpolate(q)
            heading = profile.heading_at(q)
            shift = (1 - _quintic(u)) * lateral
            points.append((centre.x - shift * sin(heading), centre.y + shift * cos(heading)))
        # Find the leg and s of the landing point along the corridor.
        travelled = 0.0
        for leg_index, low, high, piece in pieces:
            if distance <= travelled + piece.length + 1e-9 or leg_index == pieces[-1][0]:
                landing = min(low + (distance - travelled), high)
                lane = self.service.get_lane_central_curve_by_id(legs[leg_index].lane_id)
                end = lane.interpolate(landing)
                points[-1] = (end.x, end.y)
                return points, leg_index, landing
            travelled += piece.length
        return None

    def _straight_ahead(self, lane_id: str) -> bool:
        """
        Whether a map lane is not a turn lane and not a junction lane (``is_junction_lane``).

        :param str lane_id: Map or virtual-lane ID.
        :returns: ``True`` if a map lane is not a turn lane and not a junction lane (``is_junction_lane``); otherwise ``False``.
        :rtype: bool
        """
        if lane_id in self._virtual:
            return False
        return self.service.get_lane_by_id(lane_id).turn == self._no_turn and not self.is_junction_lane(lane_id)

    def is_junction_lane(self, lane_id: str) -> bool:
        """
        Whether a lane belongs to a junction: it has a ``junction_id``, lies
        mostly (``JUNCTION_SHARE``) inside the map's junction polygons, or is a
        turn lane that enters one.

        :param str lane_id: Map or virtual-lane ID.
        :returns: ``True`` if the lane is classified as a junction lane; otherwise ``False``.
        :rtype: bool
        """
        return lane_id in _junction_info(self.service, self._no_turn)["lanes"]

    @property
    def restricts_endpoints(self) -> bool:
        """
        Whether the rules keep route starts and ends away from junctions
        (``RouteRules.endpoint_junction_margin``).

        :returns: ``True`` if the rules keep route starts and ends away from junctions (``RouteRules.endpoint_junction_margin``); otherwise ``False``.
        :rtype: bool
        """
        return self.rules.endpoint_junction_margin is not None

    def endpoint_ranges(self, lane_id: str) -> Tuple[Tuple[float, float], ...]:
        """
        The stretches ``(low, high)`` of an allowed lane where a route may start
        or end: none on a virtual lane, and the whole lane unless
        ``restricts_endpoints``. Otherwise none on a junction lane, and elsewhere
        the positions at least ``endpoint_junction_margin`` metres from every
        junction polygon and from a lane end that joins a junction lane, checked
        every ``_ENDPOINT_STEP`` metres.

        :param str lane_id: Allowed map or virtual-lane ID.
        :returns: Allowed endpoint intervals ``(low, high)`` in lane metres; empty for a virtual or otherwise ineligible lane.
        :rtype: Tuple[Tuple[float, float], ...]
        """
        if lane_id in self._virtual:
            return ()
        length = self._lengths[lane_id]
        if not self.restricts_endpoints:
            return ((0.0, length),)
        margin = self.rules.endpoint_junction_margin
        info = _junction_info(self.service, self._no_turn)
        key = (lane_id, margin, self._offsets.get(lane_id, 0.0))
        if key not in info["ranges"]:
            info["ranges"][key] = self._endpoint_ranges(lane_id, length, margin, info)
        return info["ranges"][key]

    def _endpoint_ranges(self, lane_id: str, length: float, margin: float, info: Dict,
                         ) -> Tuple[Tuple[float, float], ...]:
        """
        Compute lane intervals whose endpoint centres and margins avoid junctions.

        :param str lane_id: Map lane ID.
        :param float length: Lane centerline length, in metres.
        :param float margin: Minimum clearance from junction geometry, in metres.
        :param Dict info: Cached junction geometry, lane membership, and endpoint intervals.
        :returns: Allowed endpoint intervals ``(low, high)`` in lane metres.
        :rtype: Tuple[Tuple[float, float], ...]
        """
        if lane_id in info["lanes"]:
            return ()
        s = np.linspace(0.0, length, max(2, ceil(length / _ENDPOINT_STEP) + 1))
        ok = np.ones(len(s), dtype=bool)
        if info["union"] is not None:
            line = self.centerline(lane_id)
            points = shapely.line_interpolate_point(line, s)
            ok &= shapely.distance(info["union"], points) >= margin
        lane = self.service.get_lane_by_id(lane_id)
        if any(other.id in info["lanes"] for other in lane.successor_id):
            ok &= s <= length - margin
        if any(other.id in info["lanes"] for other in lane.predecessor_id):
            ok &= s >= margin
        ranges, first = [], None
        for index, allowed in enumerate(ok):
            if allowed and first is None:
                first = index
            if first is not None and (not allowed or index == len(s) - 1):
                last = index if allowed else index - 1
                if s[last] > s[first]:
                    ranges.append((float(s[first]), float(s[last])))
                first = None
        return tuple(ranges)

    def start_lanes(self) -> List[str]:
        """
        Sorted allowed map lanes a route may start on (with an allowed endpoint).

        :returns: Sorted map-lane IDs having at least one allowed route-start position.
        :rtype: List[str]
        """
        return list(self._start_lanes())

    def _start_lanes(self) -> List[str]:
        """
        Sorted allowed lanes a route may start on.

        :returns: Sorted map-lane IDs on which a route may start.
        :rtype: List[str]
        """
        if not hasattr(self, "_start_lane_list"):
            lanes = sorted(lane for lane in self.graph if lane not in self._virtual)
            self._start_lane_list = [lane for lane in lanes if self.endpoint_ranges(lane)] \
                if self.restricts_endpoints else lanes
        return self._start_lane_list

    def endpoint_allowed(self, lane_id: str, s: float) -> bool:
        """
        Whether a route may start or end at ``s`` on an allowed lane (``endpoint_ranges``).

        :param str lane_id: Allowed map or virtual-lane ID.
        :param float s: Position along the lane, in metres.
        :returns: ``True`` if a route may start or end at ``s`` on an allowed lane (``endpoint_ranges``); otherwise ``False``.
        :rtype: bool
        """
        return any(low - 1e-9 <= s <= high + 1e-9 for low, high in self.endpoint_ranges(lane_id))

    def random_endpoint(self, lane_id: str, rng: Random, low: float = 0.0,
                        high: float | None = None) -> float | None:
        """
        A uniformly random allowed endpoint s on an allowed lane within ``[low, high]``
        (default the whole lane), or None if there is none. Without endpoint
        restrictions this is ``rng.uniform(low, high)`` on a map lane.

        :param str lane_id: Allowed map or virtual-lane ID.
        :param Random rng: Random number generator used for sampling and selection.
        :param float low: Lower lane-position bound, in metres.
        :param float | None high: Upper lane-position bound, in metres; ``None`` uses the lane end.
        :returns: A uniformly sampled allowed endpoint in ``[low, high]``, or ``None`` if the interval contains none.
        :rtype: float | None
        """
        high = self._lengths[lane_id] if high is None else high
        if not self.restricts_endpoints and lane_id not in self._virtual:
            return rng.uniform(low, high)
        pieces = [(max(a, low), min(b, high)) for a, b in self.endpoint_ranges(lane_id) if min(b, high) > max(a, low)]
        if not pieces:
            return None
        u = rng.uniform(0.0, sum(b - a for a, b in pieces))
        for a, b in pieces:
            if u <= b - a:
                return a + u
            u -= b - a
        return pieces[-1][1]

    def nearest_endpoint(self, lane_id: str, s: float, low: float = 0.0,
                         high: float | None = None) -> float | None:
        """
        The allowed endpoint within ``[low, high]`` on an allowed lane nearest to
        ``s``, or None if there is none.

        :param str lane_id: Allowed map or virtual-lane ID.
        :param float s: Desired lane position, in metres.
        :param float low: Lower lane-position bound, in metres.
        :param float | None high: Upper lane-position bound, in metres; ``None`` uses the lane end.
        :returns: The allowed endpoint in ``[low, high]`` nearest to ``s``, or ``None`` if none exists.
        :rtype: float | None
        """
        high = self._lengths[lane_id] if high is None else high
        best = None
        for a, b in self.endpoint_ranges(lane_id):
            a, b = max(a, low), min(b, high)
            if b < a:
                continue
            candidate = min(max(s, a), b)
            if best is None or abs(candidate - s) < abs(best - s):
                best = candidate
        return best

    def _check_endpoints(self, route: LaneRoute) -> None:
        """
        Reject route endpoints that violate the route space endpoint restrictions.

        :param LaneRoute route: Route whose endpoints are checked.
        :raises InvalidRoute: If the start or end violates the configured endpoint restrictions.
        """
        if not self.restricts_endpoints:
            return
        margin = self.rules.endpoint_junction_margin
        for name, lane_id, s in (("start_s", route.lanes[0], route.start_s), ("end_s", route.lanes[-1], route.end_s)):
            if not self.endpoint_allowed(lane_id, s):
                raise InvalidRoute(f"{name}={s:.1f} on {lane_id!r} is on a junction lane or within "
                                   f"{margin} m of a junction")

    def _longer_change(self, source: LineString, change_s: float, target: LineString,
                       landing: float, target_end: float, wanted: float) -> Tuple[float, float]:
        """
        The longest lane change up to ``wanted`` metres from ``change_s`` that
        fits the source lane and lands at least ``_LANDING_MARGIN`` before
        ``target_end``, with its landing; the minimum change otherwise.

        :param LineString source: Source-lane centerline.
        :param float change_s: Lane-change start position on the source lane, in metres.
        :param LineString target: Target-lane centerline.
        :param float landing: Position where the lane change reaches the target lane, in metres.
        :param float target_end: End of the usable target-lane interval, in metres.
        :param float wanted: Requested lane-change length, in metres.
        :returns: Selected lane-change length and target-lane landing ``s``.
        :rtype: Tuple[float, float]
        """
        minimum = self.rules.min_lane_change_distance
        room = source.length - change_s
        distance = min(wanted, room)
        while distance > minimum:
            far = target.project(source.interpolate(change_s + distance))
            if landing <= far <= target_end - _LANDING_MARGIN:
                return distance, far
            distance = max(minimum, distance - _CHANGE_STEP)
        return minimum, landing

    def path(self, route: LaneRoute, *,
             lane_change_s: Tuple[float, ...] | None = None,
             lane_change_distances: Tuple[float, ...] | None = None) -> LineString:
        """
        Return the line the route drives along: the stretch of each lane's
        centerline, joined by lane changes that blend from the source to the
        target lane over ``min_lane_change_distance`` with a quintic (minimum-jerk) weight.

        :param LaneRoute route: The route.
        :param Tuple[float, ...] | None lane_change_s: As in ``validate``.
        :param Tuple[float, ...] | None lane_change_distances: As in ``path_sections``.
        :returns: Route path geometry in map metres.
        :rtype: LineString
        :raises InvalidRoute: If the route is illegal.
        """
        points: List[Tuple[float, float]] = []
        for section in self.path_sections(route, lane_change_s=lane_change_s,
                                          lane_change_distances=lane_change_distances):
            coords = list(section.line.coords)
            points.extend(coords[1:] if points else coords)
        return LineString(points)

    def lane_change_positions(
        self, route: LaneRoute, rng: Random, *,
        keep: Mapping[Tuple[str, str], float] | None = None,
    ) -> Tuple[float, ...]:
        """
        Choose where each lane change starts, for agents that drive their own changes.

        Reuse a position in ``keep`` when it leaves a feasible continuation.
        Otherwise draw within the source window, shrinking rejected ranges
        toward a verified earliest continuation. This also handles very narrow
        windows imposed by the end position or consecutive lane changes.

        :param LaneRoute route: The route.
        :param Random rng: Source of randomness.
        :param Mapping[Tuple[str, str], float] | None keep: Positions by (source lane, target lane), e.g. from a parent route.
        :returns: Lane-change start positions on source lanes, in route order.
        :rtype: Tuple[float, ...]
        :raises InvalidRoute: If no feasible continuation can be found.
        """
        self._route_lengths(route)
        keep = keep or {}

        def choose(index: int, earliest: float, latest: float) -> float:
            """
            Choose a lane-change start that permits the remaining route to be completed.

            :param int index: Index of the lane-change transition.
            :param float earliest: Earliest feasible lane-change start position, in metres.
            :param float latest: Latest feasible lane-change start position, in metres.
            :returns: A feasible lane-change start position, in metres.
            :rtype: float
            :raises InvalidRoute: If no feasible lane-change start can complete the remaining route.
            """
            def can_finish(change: float) -> bool:
                """
                Check whether a proposed lane-change start leaves a feasible route continuation.

                :param float change: Proposed lane-change start position, in metres.
                :returns: ``True`` if the proposed change leaves a feasible route continuation.
                :rtype: bool
                """
                entry = self.entry_after_change(route.lanes[index], change, route.lanes[index + 1])
                suffix = LaneRoute(route.lanes[index + 1:], route.transitions[index + 1:], entry, route.end_s)
                try:
                    self._check(suffix)
                except InvalidRoute:
                    # A curved lane's projection can run backwards. The
                    # parent's later changes may work even when early ones do not.
                    if not keep:
                        return False

                    def preferred(i: int, low: float, high: float) -> float:
                        """
                        Reuse a preferred lane-change start when it fits the continuation window.

                        :param int i: Index into the ordered input sequence.
                        :param float low: Earliest feasible lane-change start, in metres.
                        :param float high: Latest feasible lane-change start, in metres.
                        :returns: The preferred start if it lies in the window; otherwise ``low``.
                        :rtype: float
                        """
                        value = keep.get((suffix.lanes[i], suffix.lanes[i + 1]))
                        return value if value is not None and low <= value <= high else low

                    try:
                        _, later = self._entries(suffix, preferred)
                        self._check(suffix, tuple(s for s in later if s is not None))
                    except InvalidRoute:
                        return False
                return True

            kept = keep.get((route.lanes[index], route.lanes[index + 1]))
            if kept is not None and earliest <= kept <= latest and can_finish(kept):
                return kept
            fallback = earliest if can_finish(earliest) else None
            for _ in range(_CHANGE_SAMPLES):
                change = rng.uniform(earliest, latest)
                if can_finish(change):
                    return change
                if fallback is not None:
                    latest = change
            # Do not assume projections between curved lanes are monotone:
            # every accepted draw and the fallback have an explicit witness.
            if fallback is not None:
                return fallback
            raise InvalidRoute(f"no lane-change position from {route.lanes[index]!r} "
                               "leaves a feasible continuation")

        _, changes = self._entries(route, choose)
        positions = tuple(change for change in changes if change is not None)
        self._check(route, positions)
        return positions

    def transition(self, source: str, target: str) -> Transition:
        """
        Return the legal step from ``source`` to ``target``.

        :param str source: Source lane ID.
        :param str target: Target lane ID.
        :returns: Transition represented by the legal edge.
        :rtype: Transition
        :raises KeyError: If there is no such edge for these rules.
        """
        return self.graph[source][target]["transition"]

    def lane_length(self, lane_id: str) -> float:
        """
        Return an allowed map or virtual lane's centerline length in metres.

        :param str lane_id: Allowed map or virtual-lane ID.
        :returns: Centerline length, in metres.
        :rtype: float
        """
        return self._lengths[lane_id]

    def centerline(self, lane_id: str) -> LineString:
        """
        Return the centerline of a map or virtual lane, shifted right by its
        ``lane_offsets`` entry if it has one.

        :param str lane_id: Allowed map or virtual-lane ID.
        :returns: The constructed line geometry in map metres.
        :rtype: LineString
        """
        lane = self._virtual.get(lane_id)
        if lane is not None:
            return lane.line
        offset = self._offsets.get(lane_id)
        if not offset:
            return self.service.get_lane_central_curve_by_id(lane_id)
        line = self._offset_lines.get(lane_id)
        if line is None:
            line = self._offset_lines[lane_id] = offset_line(self.service.get_lane_central_curve_by_id(lane_id),
                                                             offset)
        return line

    def lane_offset(self, lane_id: str) -> float:
        """
        Metres the lane's centerline is shifted to its right (``lane_offsets``); 0 by default.

        :param str lane_id: Allowed map or virtual-lane ID.
        :returns: Rightward centerline offset in metres, or 0 if none.
        :rtype: float
        """
        return self._offsets.get(lane_id, 0.0)

    def virtual_lane(self, lane_id: str) -> VirtualLane | None:
        """
        Return the virtual lane with this ID, or None for a map lane.

        :param str lane_id: Allowed map or virtual-lane ID.
        :returns: The matching virtual lane, or ``None`` for a map lane.
        :rtype: VirtualLane | None
        """
        return self._virtual.get(lane_id)

    def entry_after_change(self, source: str, change_s: float, target: str) -> float:
        """
        Return the target-lane s reached by a lane change starting at ``change_s`` on ``source``.

        The change covers ``min_lane_change_distance`` along the source lane;
        the point there is projected onto the target lane.

        :param str source: Source lane ID.
        :param float change_s: Lane-change start position on the source lane, in metres.
        :param str target: Target lane ID.
        :returns: Projected entry position on the target lane, in metres.
        :rtype: float
        """
        point = self._pose(source, change_s + self.rules.min_lane_change_distance).point
        return self.centerline(target).project(point)

    def poses(self, route: LaneRoute) -> RoutePoses:
        """
        Return the start pose, waypoints, and end pose that make Apollo follow the route.

        Every lane after the first and before the last gets a waypoint where an
        unambiguous position exists; decision lanes, entered by a lane change or
        chosen among several outgoing edges, must have one. Before a lane change
        the waypoint is placed as early as possible, otherwise mid-window.
        Each waypoint delays the following change as needed; the resulting
        entries must still leave room for all later changes and the end pose.

        :param LaneRoute route: The route to convert.
        :returns: Start pose, intermediate waypoints, and end pose for Apollo routing.
        :rtype: RoutePoses
        :raises InvalidRoute: If the route is illegal or cannot be sent unambiguously.
        """
        lengths, entries, _ = self._check(route)
        self._check_endpoints(route)
        start = self._pose(route.lanes[0], route.start_s)
        if not self._unambiguous(start):
            raise InvalidRoute(f"start pose is ambiguous at {route.lanes[0]!r} s={route.start_s}")
        end = self._pose(route.lanes[-1], route.end_s)
        if not self._unambiguous(end):
            raise InvalidRoute(f"end pose is ambiguous at {route.lanes[-1]!r} s={route.end_s}")

        distance = self.rules.min_lane_change_distance
        waypoints = []
        changes = []
        for index in range(len(route.lanes) - 1):
            lane_id = route.lanes[index]
            changes_next = route.transitions[index] is not Transition.FORWARD
            low = entries[index]
            high = lengths[index] - distance if changes_next else lengths[index]
            if changes_next:
                self._change_window(lane_id, low)
            if index:
                pose = self._waypoint(lane_id, low, high, early=changes_next)
                if pose is not None:
                    waypoints.append(pose)
                    low = pose.s
                elif (route.transitions[index - 1] is not Transition.FORWARD
                      or self.graph.out_degree(route.lanes[index - 1]) > 1):
                    raise InvalidRoute(f"no unambiguous waypoint on decision lane {lane_id!r}")
            if changes_next:
                changes.append(low)
                entries[index + 1] = self.entry_after_change(lane_id, low, route.lanes[index + 1])
            else:
                entries[index + 1] = 0.0
        self._check(route, tuple(changes))
        return RoutePoses(start, tuple(waypoints), end)

    def length(self, route: LaneRoute, *, lane_change_s: Tuple[float, ...] | None = None) -> float:
        """
        Return the driven distance in metres; each lane change counts as
        ``min_lane_change_distance``.

        :param LaneRoute route: Route whose driven length is measured.
        :param Tuple[float, ...] | None lane_change_s: As in ``validate``.
        :returns: Total driven route length, in metres.
        :rtype: float
        :raises InvalidRoute: If the route is illegal.
        """
        legs = self.legs(route, lane_change_s=lane_change_s)
        return (sum(leg.end_s - leg.start_s for leg in legs)
                + route.lane_changes * self.rules.min_lane_change_distance)

    def random_route(
        self, rng: Random, *, min_length: float = 50.0, max_length: float = 500.0,
        lane_change_probability: float = 0.2, max_attempts: int = 100,
        start_lanes: Iterable[str] | None = None,
    ) -> LaneRoute:
        """
        Draw a valid route by a random walk of a random target length.

        Walks start on map lanes, never on a virtual lane. The target length is
        uniform in the bounds and kept for ``WALKS_PER_TARGET`` walks, so that
        lengths stay close to uniform where long walks often fail. The driven distance
        (``length``) is within ``[min_length, max_length]``: walking on to an
        allowed endpoint can overshoot the target, and such walks are redrawn.

        :param Random rng: Source of randomness; equal seeds give equal routes.
        :param float min_length: Minimum driven distance in metres.
        :param float max_length: Maximum driven distance in metres.
        :param float lane_change_probability: Chance of a lane change where one is possible.
        :param int max_attempts: Walks to try before giving up.
        :param Iterable[str] | None start_lanes: If given, walks start only on these lanes (among
            those a route may start on), e.g. a bicycle's bike lanes.
        :returns: A randomly generated valid route satisfying the requested length bounds.
        :rtype: LaneRoute
        :raises RuntimeError: If no valid route was found.
        """
        lanes = self._start_lanes()
        if start_lanes is not None:
            wanted = set(start_lanes)
            lanes = [lane for lane in lanes if lane in wanted]
        if lanes:
            target = 0.0
            for attempt in range(max_attempts):
                # Each target length gets several walks: long walks fail more
                # often (dead ends, endpoint rules), and redrawing the target
                # after every failure skewed lengths short of uniform. Genetic
                # operators then drifted them back towards the middle of the
                # bounds (borregas_ave NPCs: median 269 -> 335 m in 15 generations).
                if attempt % WALKS_PER_TARGET == 0:
                    target = rng.uniform(min_length, max_length)
                route = self._walk(rng, rng.choice(lanes), target, lane_change_probability)
                if route is not None and self.is_valid(route) and min_length <= self.length(route) <= max_length:
                    # A walk that ran into a dead end well short of its target
                    # is kept only on the target's last walk.
                    if (self.length(route) >= SHORT_WALK_SHARE * target
                            or attempt % WALKS_PER_TARGET == WALKS_PER_TARGET - 1):
                        return route
        raise RuntimeError(f"no valid route found in {max_attempts} attempts")

    def alternative_routes(
        self, start: Tuple[str, float], end: Tuple[str, float], k: int = 5,
    ) -> List[LaneRoute]:
        """
        Return up to ``k`` valid routes between two lane positions, shortest first.

        Candidates are simple lane paths ordered by lane lengths, with each lane
        change costing ``min_lane_change_distance``.

        :param Tuple[str, float] start: Start lane ID and s.
        :param Tuple[str, float] end: End lane ID and s.
        :param int k: Maximum number of routes.
        :returns: Up to ``k`` valid alternatives, shortest first.
        :rtype: List[LaneRoute]
        """
        (start_lane, start_s), (end_lane, end_s) = start, end
        routes = []
        for lanes, transitions in self.lane_paths(start_lane, end_lane, 10 * k):
            route = LaneRoute(lanes, transitions, start_s, end_s)
            if self.is_valid(route):
                routes.append(route)
                if len(routes) == k:
                    break
        return routes

    def _weight(self, source: str, entry_s: float | None, attribute: str = "distance_lower_bound") -> Callable[[str, str, Mapping[str, float | Transition]], float]:
        """
        Edge weights for a search from ``source``, entered at ``entry_s``.

        Without ``entry_s`` every edge costs its ``cost``. With it, the source's
        forward edges cost its remaining length and other edges ``attribute``.
        The default, ``distance_lower_bound``, is optimistic: a later lane may be
        entered partway through by a lane change, and charging its full length
        would wrongly prune feasible routes. Zero is a safe lower bound for its
        forward remainder; complete children are validated with their actual
        entries after candidate selection.

        :param str source: Source lane ID.
        :param float | None entry_s: Entry position on the source lane, in metres; ``None`` uses full edge costs.
        :param str attribute: Edge attribute used as the search cost away from the initial source.
        :returns: An edge-cost callback for the supplied source entry and edge attribute.
        :rtype: Callable[[str, str, Mapping[str, float | Transition]], float]
        :raises ValueError: If ``entry_s`` is not finite within the source lane.
        """
        if entry_s is None:
            return lambda u, v, data: data["cost"]
        if not isfinite(entry_s) or not 0 <= entry_s <= self._lengths[source]:
            raise ValueError("entry_s must be within the source lane")
        remaining = self._lengths[source] - entry_s

        def weight(u: str, v: str, data: Mapping[str, float | Transition]) -> float:
            """
            Compute the search edge cost using the source remainder or the selected edge attribute.

            :param str u: Source lane ID of the edge.
            :param str v: Destination lane ID of the edge.
            :param Mapping[str, float | Transition] data: Edge attributes supplying transition kind and search costs.
            :returns: The edge search cost, in metres.
            :rtype: float
            """
            if u == source and data["transition"] is Transition.FORWARD:
                return remaining
            return data[attribute]

        return weight

    def reachable(self, source: str, max_cost: float, *, entry_s: float | None = None) -> Dict[str, float]:
        """
        Return the lanes reachable from ``source`` within ``max_cost``, with their costs.

        Costs are as in ``lane_paths``; ``source`` itself costs 0.

        :param str source: Starting lane.
        :param float max_cost: Largest path cost to include.
        :param float | None entry_s: If supplied, use lower bounds on driven
            distance from this source position, including partial lane entries.
        :returns: Lane ID to cost; empty if ``source`` is not allowed.
        :rtype: Dict[str, float]
        """
        if source not in self.graph:
            return {}
        return nx.single_source_dijkstra_path_length(
            self.graph, source, cutoff=max_cost, weight=self._weight(source, entry_s))

    def lane_paths(
        self, source: str, target: str, k: int = 5, *, max_cost: float | None = None,
        entry_s: float | None = None,
    ) -> Iterator[Tuple[Tuple[str, ...], Tuple[Transition, ...]]]:
        """
        Yield up to ``k`` simple lane paths from ``source`` to ``target``, cheapest first.

        A path's cost sums the lengths of the lanes it leaves forward, plus
        ``min_lane_change_distance`` per lane change. Paths are not validated
        as routes.

        :param str source: First lane of every path.
        :param str target: Last lane of every path.
        :param int k: Maximum number of paths.
        :param float | None max_cost: Only yield paths costing at most this; the
            search is then confined to lanes on such paths, which keeps it local.
        :param float | None entry_s: Prune using driven-distance lower bounds
            from this position. Rank the candidates by lane cost with the unused
            source prefix removed; validate complete routes before accepting them.
        :returns: (lanes, transitions) pairs; nothing if either lane is not allowed
            or the target is unreachable.
        :rtype: Iterator[Tuple[Tuple[str, ...], Tuple[Transition, ...]]]
        """
        if source not in self.graph or target not in self.graph or k <= 0:
            return
        if source == target:
            yield (source,), ()
            return
        graph = self.graph
        cost = self._weight(source, entry_s)
        if max_cost is not None:
            ahead = nx.single_source_dijkstra_path_length(graph, source, cutoff=max_cost, weight=cost)
            if target not in ahead:
                return
            behind = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), target,
                                                           cutoff=max_cost,
                                                           weight=lambda u, v, data: cost(v, u, data))
            graph = _ordered_subgraph(graph, [lane for lane, cost in ahead.items()
                                             if lane in behind and cost + behind[lane] <= max_cost])
        # Lower bounds are for pruning only. Ranking by them would favour long
        # detours through zero-cost forward edges on roads with parallel lanes.
        rank = "cost" if entry_s is None else self._weight(source, entry_s, "cost")
        paths = nx.shortest_simple_paths(graph, source, target, weight=rank)
        try:
            for path in itertools.islice(paths, k):
                if max_cost is not None and sum(cost(u, v, graph[u][v])
                                               for u, v in zip(path, path[1:])) > max_cost:
                    if entry_s is None:
                        break
                    continue
                yield tuple(path), self._transitions(path)
        except nx.NetworkXNoPath:
            return

    def _route_lengths(self, route: LaneRoute) -> List[float]:
        """
        Check topology and endpoint bounds without choosing lane-change positions.

        :param LaneRoute route: Route whose topology and endpoint bounds are checked.
        :returns: Centerline length of each route lane, in route order.
        :rtype: List[float]
        :raises InvalidRoute: If a lane or edge is illegal, a lane repeats, a virtual lane is an endpoint, or an endpoint lies outside its lane.
        """
        for lane_id in route.lanes:
            if lane_id not in self.graph:
                raise InvalidRoute(f"lane {lane_id!r} is not allowed for these rules")
        if len(set(route.lanes)) != len(route.lanes):
            raise InvalidRoute("a route may not use a lane more than once")
        for lane_id in (route.lanes[0], route.lanes[-1]):
            if lane_id in self._virtual:
                raise InvalidRoute(f"a route may not start or end on virtual lane {lane_id!r}")
        for (source, target), transition in zip(zip(route.lanes, route.lanes[1:]), route.transitions):
            data = self.graph.get_edge_data(source, target)
            if data is None or data["transition"] is not transition:
                raise InvalidRoute(f"no {transition.name} edge {source!r} -> {target!r}")

        lengths = [self._lengths[lane] for lane in route.lanes]
        if not 0 <= route.start_s <= lengths[0]:
            raise InvalidRoute(f"start_s={route.start_s} is outside lane {route.lanes[0]!r}")
        if not 0 <= route.end_s <= lengths[-1]:
            raise InvalidRoute(f"end_s={route.end_s} is outside lane {route.lanes[-1]!r}")
        return lengths

    def _check(self, route: LaneRoute, lane_change_s: Tuple[float, ...] | None = None,
               ) -> Tuple[List[float], List[float], List[float | None]]:
        """
        Validate topology and progress; return lane lengths, entry positions,
        and each transition's lane-change start (None for forward steps).

        :param LaneRoute route: Route to validate.
        :param Tuple[float, ...] | None lane_change_s: Lane-change start positions on source lanes, in metres; ``None`` uses default placement.
        :returns: Lane lengths, lane-entry positions, and per-transition lane-change starts.
        :rtype: Tuple[List[float], List[float], List[float | None]]
        :raises InvalidRoute: If route topology, endpoints, lane-change count, progress, or a lane-change position is invalid.
        """
        lengths = self._route_lengths(route)
        lateral = route.lane_change_indices
        if lane_change_s is not None and len(lane_change_s) != len(lateral):
            raise InvalidRoute(f"lane_change_s needs one position per lane change ({len(lateral)}), "
                               f"got {len(lane_change_s)}")
        given = dict(zip(lateral, lane_change_s or ()))

        def choose(index: int, earliest: float, latest: float) -> float:
            """
            Select a supplied or earliest lane-change start and validate its feasible window.

            :param int index: Index of the lane-change transition.
            :param float earliest: Earliest feasible lane-change start position, in metres.
            :param float latest: Latest feasible lane-change start position, in metres.
            :returns: Supplied lane-change start, or the earliest feasible start if none was supplied.
            :rtype: float
            :raises InvalidRoute: If the chosen start is non-finite or outside its feasible window.
            """
            change = given.get(index, earliest)
            if not (isfinite(change) and earliest <= change <= latest):
                raise InvalidRoute(f"lane change from {route.lanes[index]!r} must start within "
                                   f"s=[{earliest:.1f}, {latest:.1f}], not {change}")
            return change

        entries, changes = self._entries(route, choose)
        if route.end_s <= entries[-1]:
            raise InvalidRoute(f"end_s={route.end_s} must be beyond s={entries[-1]:.1f}, "
                               f"where the route enters {route.lanes[-1]!r}")
        return lengths, entries, changes

    def _entries(self, route: LaneRoute, choose: Callable[[int, float, float], float],
                 ) -> Tuple[List[float], List[float | None]]:
        """
        Walk the route: where it enters each lane, and where each transition's
        lane change starts (None for forward steps). ``choose(index, earliest,
        latest)`` places the lane change at ``route.transitions[index]``.

        :param LaneRoute route: Route to traverse.
        :param Callable[[int, float, float], float] choose: Callback choosing a lane-change start within the supplied feasible window.
        :returns: Lane-entry positions and per-transition lane-change starts.
        :rtype: Tuple[List[float], List[float | None]]
        """
        entries, changes = [route.start_s], []
        for index, transition in enumerate(route.transitions):
            if transition is Transition.FORWARD:
                entries.append(0.0)
                changes.append(None)
                continue
            source, target = route.lanes[index], route.lanes[index + 1]
            change = choose(index, *self._change_window(source, entries[index]))
            changes.append(change)
            entries.append(self.entry_after_change(source, change, target))
        return entries, changes

    def _change_window(self, source: str, entry: float) -> Tuple[float, float]:
        """
        Earliest and latest start of a lane change from ``source``, entered at ``entry``.

        :param str source: Source lane ID.
        :param float entry: Position at which the route enters the source lane, in metres.
        :returns: Earliest and latest feasible lane-change start positions.
        :rtype: Tuple[float, float]
        :raises InvalidRoute: If the source lane has insufficient room for the minimum lane change.
        """
        distance = self.rules.min_lane_change_distance
        latest = self._lengths[source] - distance
        if latest < entry:
            raise InvalidRoute(f"lane change from {source!r} needs {distance} m "
                               f"but only {self._lengths[source] - entry:.1f} m remain")
        return entry, latest

    def _pose(self, lane_id: str, s: float) -> RoutePose:
        """
        Convert a lane position to its map coordinates and heading.

        :param str lane_id: Map lane ID.
        :param float s: Position along the lane, in metres.
        :returns: Lane pose at the requested position.
        :rtype: RoutePose
        """
        point, heading = self.service.get_lane_coord_and_heading(lane_id, s)
        return RoutePose(lane_id, s, point.x, point.y, heading)

    def _unambiguous(self, pose: RoutePose) -> bool:
        """
        No lane of any type other than the pose's own lies within the margin.

        :param RoutePose pose: Lane position with its map coordinates and heading.
        :returns: ``True`` if no other map lane lies within the ambiguity margin.
        :rtype: bool
        """
        return set(self.service.get_lanes(pose.point, self.rules.ambiguity_margin)) == {pose.lane_id}

    def _waypoint(self, lane_id: str, low: float, high: float, *, early: bool) -> RoutePose | None:
        """
        Find an unambiguous waypoint within a lane interval.

        :param str lane_id: Map lane ID.
        :param float low: Earliest lane position to consider, in metres.
        :param float high: Latest lane position to consider, in metres.
        :param bool early: Whether to prefer the earliest allowed position.
        :returns: An unambiguous waypoint in the interval, or ``None`` if none exists.
        :rtype: RoutePose | None
        """
        if high < low:
            return None
        middle = (low + high) / 2
        candidates = sorted(self._candidate_s(low, high),
                            key=(lambda s: s) if early else (lambda s: abs(s - middle)))
        for s in candidates:
            pose = self._pose(lane_id, s)
            if self._unambiguous(pose):
                return pose
        return None

    @staticmethod
    def _candidate_s(low: float, high: float) -> Iterator[float]:
        """
        Yield regularly spaced candidate positions and the interval end.

        :param float low: Interval start, in metres.
        :param float high: Interval end, in metres.
        :returns: Candidate lane positions from ``low`` through ``high``.
        :rtype: Iterator[float]
        """
        steps = int((high - low) // _WAYPOINT_STEP)
        yield from (low + index * _WAYPOINT_STEP for index in range(steps + 1))
        if low + steps * _WAYPOINT_STEP < high:
            yield high

    def _transitions(self, path: List[str]) -> Tuple[Transition, ...]:
        """
        Collect the legal transitions between consecutive lanes in a path.

        :param List[str] path: Lane IDs in path order.
        :returns: Legal transitions between each consecutive lane pair.
        :rtype: Tuple[Transition, ...]
        """
        return tuple(self.transition(source, target) for source, target in zip(path, path[1:]))

    def _walk(self, rng: Random, lane_id: str, target: float,
              lane_change_probability: float) -> LaneRoute | None:
        """
        One random walk of about ``target`` metres; validated by the caller.

        :param Random rng: Random number generator used for sampling and selection.
        :param str lane_id: Allowed map lane on which the walk starts.
        :param float target: Target driven distance, in metres.
        :param float lane_change_probability: Probability of selecting a lane change during random route generation.
        :returns: A candidate route, or ``None`` if the walk cannot finish validly.
        :rtype: LaneRoute | None
        """
        distance = self.rules.min_lane_change_distance
        start_s = self.random_endpoint(lane_id, rng)
        if start_s is None:
            return None
        lanes, transitions = [lane_id], []
        entry, travelled = start_s, 0.0
        while True:
            current = lanes[-1]
            remaining = self._lengths[current] - entry
            if target - travelled <= remaining:
                end_s = entry + target - travelled
                if not self.restricts_endpoints and current not in self._virtual:
                    return LaneRoute(tuple(lanes), tuple(transitions), start_s, end_s)
                # The nearest allowed end after the entry, else drive on to a lane with one.
                if travelled > target + _ENDPOINT_OVERSHOOT:
                    return None
                end_s = self.nearest_endpoint(current, end_s, low=entry + _ENDPOINT_STEP)
                if end_s is not None:
                    return LaneRoute(tuple(lanes), tuple(transitions), start_s, end_s)
            forward, lateral = [], []
            for successor in sorted(self.graph.successors(current)):
                if successor in lanes:
                    continue
                if self.transition(current, successor) is Transition.FORWARD:
                    forward.append(successor)
                elif remaining >= distance:
                    lateral.append(successor)
            if lateral and (not forward or rng.random() < lane_change_probability):
                successor = rng.choice(lateral)
                entry = self.entry_after_change(current, entry, successor)
                travelled += distance
            elif forward:
                successor = rng.choice(forward)
                entry, travelled = 0.0, travelled + remaining
            else:
                # Dead end: finish somewhere on this lane.
                end_s = self.random_endpoint(current, rng, entry, entry + remaining)
                if end_s is None:
                    return None
                return LaneRoute(tuple(lanes), tuple(transitions), start_s, end_s)
            lanes.append(successor)
            transitions.append(self.transition(current, successor))
