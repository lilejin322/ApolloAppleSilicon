"""
Lane-route chromosomes for the ego vehicle and NPCs.

A route is an ordered sequence of lanes connected by forward steps or lane
changes, with a start ``s`` on the first lane and an end ``s`` on the last.
Routes are immutable, so crossover and mutation produce new ones. A
``RouteSpace`` decides which routes are legal for one agent type on one map
and converts them into the poses Apollo receives.
"""

import itertools
from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite
from random import Random
from typing import Callable, Iterable, Iterator, Mapping, NamedTuple

import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import substring

from ..map import MapService
from ..message.descriptor import DescriptorBundle
from ..utils import require_nonnegative

_WAYPOINT_STEP = 1.0
"""Spacing in metres of the candidate waypoint positions tried on a lane."""
_BLEND_STEPS = 20
"""Points per lane change when blending a path from the source to the target lane."""
_CHANGE_SAMPLES = 64
"""Maximum draws while narrowing a lane-change window around a feasible continuation."""


def _ordered_subgraph(graph: nx.DiGraph, lanes: Iterable[str]) -> nx.DiGraph:
    """Copy allowed nodes and edges in a stable order, including equal-cost search ties."""
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
    """The route cannot be driven or sent unambiguously on this map."""


class Transition(Enum):
    """How a route moves from one lane to the next, as in ``routing_graph``."""

    FORWARD = "F"
    LEFT = "L"
    RIGHT = "R"


def lane_changes(transitions: Iterable[Transition]) -> int:
    """
    Count the lane changes among route transitions.

    :rtype: int
    """
    return sum(step is not Transition.FORWARD for step in transitions)


@dataclass(frozen=True)
class LaneRoute:
    """
    A route along lanes, from ``start_s`` on the first lane to ``end_s`` on the last.

    ``transitions[i]`` connects ``lanes[i]`` to ``lanes[i + 1]``. Where a lane
    change happens along a lane is decided by Apollo's planning, not here.

    :param tuple[str, ...] lanes: Lane IDs in driving order.
    :param tuple[Transition, ...] transitions: One step per consecutive lane pair.
    :param float start_s: Start position in metres along the first lane's centerline.
    :param float end_s: End position in metres along the last lane's centerline.
    """

    lanes: tuple[str, ...]
    transitions: tuple[Transition, ...]
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
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

        :rtype: int
        """
        return len(self.lane_change_indices)

    @property
    def lane_change_indices(self) -> tuple[int, ...]:
        """
        The positions in ``transitions`` of the lane changes, in route order.

        :rtype: tuple[int, ...]
        """
        return tuple(index for index, step in enumerate(self.transitions) if step is not Transition.FORWARD)

    @property
    def lane_change_pairs(self) -> tuple[tuple[str, str], ...]:
        """
        The (source lane, target lane) of each lane change, in route order.

        :rtype: tuple[tuple[str, str], ...]
        """
        return tuple((self.lanes[index], self.lanes[index + 1]) for index in self.lane_change_indices)

    def to_dict(self) -> dict:
        """
        Return a JSON-compatible representation.

        :rtype: dict
        """
        return {
            "lanes": list(self.lanes),
            "transitions": [transition.value for transition in self.transitions],
            "start_s": self.start_s,
            "end_s": self.end_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LaneRoute":
        """
        Build a route from ``to_dict`` output.

        :param dict data: Mapping with lanes, transitions, start_s, and end_s.
        :rtype: LaneRoute
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
    """

    lane_id: str
    s: float
    x: float
    y: float
    heading: float

    @property
    def point(self) -> Point:
        """
        The position as a Shapely Point.

        :rtype: Point
        """
        return Point(self.x, self.y)


@dataclass(frozen=True)
class RoutePoses:
    """
    The poses sent to Apollo for one route: start, intermediate waypoints, and end.
    """

    start: RoutePose
    waypoints: tuple[RoutePose, ...]
    end: RoutePose


class RouteLeg(NamedTuple):
    """The stretch ``[start_s, end_s]`` of one lane that a route drives along."""

    lane_id: str
    start_s: float
    end_s: float


class PathSection(NamedTuple):
    """A stretch of a route's path and the lanes it lies on: one lane, or a
    lane change's source and target lane."""

    lanes: tuple[str, ...]
    line: LineString


@dataclass(frozen=True)
class RouteRules:
    """
    Which lanes and steps an agent may use, and how routes are sent.

    :param frozenset[str] lane_types: Allowed ``Lane.type`` names.
    :param float min_turn_radius: U-turn lanes tighter than this, in metres, are excluded.
    :param float min_lane_change_distance: Metres of a lane consumed by a lane change.
    :param float ambiguity_margin: No other lane may pass within this many metres of a sent pose.
    :param bool unambiguous: Whether ``validate`` requires unambiguous poses. True for
        routes sent to Apollo (the ego); NPC routes are never matched to lanes by Apollo.
    """

    lane_types: frozenset[str] = frozenset({"CITY_DRIVING"})
    min_turn_radius: float = 0.0
    min_lane_change_distance: float = 20.0
    ambiguity_margin: float = 0.5
    unambiguous: bool = True

    def __post_init__(self) -> None:
        require_nonnegative(min_turn_radius=self.min_turn_radius,
                            min_lane_change_distance=self.min_lane_change_distance)
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

    :param MapService service: A map with its lane index and routing graph built.
    :param RouteRules rules: Allowed lanes, lane-change distance, and pose ambiguity margin.
    """

    def __init__(self, service: MapService, rules: RouteRules = RouteRules()) -> None:
        if service.routing_graph is None or service.lane_index is None:
            raise RuntimeError("RouteSpace needs a map loaded with load_only=False")
        self.service = service
        self.rules = rules
        lane_fields = DescriptorBundle(service.descriptor_path).message(
            "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name
        types = lane_fields["type"].enum_type.values_by_name
        unknown = rules.lane_types - set(types)
        if unknown:
            raise ValueError(f"unknown lane type {sorted(unknown)}")
        allowed_types = {types[name].number for name in rules.lane_types}
        u_turn = lane_fields["turn"].enum_type.values_by_name["U_TURN"].number
        allowed = {
            lane_id for lane_id, lane in service.lane_table.items()
            if lane.type in allowed_types and not (
                lane.turn == u_turn
                and turn_radius(service.get_lane_central_curve_by_id(lane_id)) < rules.min_turn_radius
            )
        }
        self._lengths = {lane_id: service.get_lane_central_curve_by_id(lane_id).length
                         for lane_id in allowed}
        # A concrete copy with precomputed costs: path searches avoid view
        # filters and per-edge geometry calls.
        self.graph = _ordered_subgraph(service.routing_graph, allowed)
        for _, _, data in self.graph.edges(data=True):
            data["transition"] = Transition(data["direction_type"])
        lateral_targets = {v for _, v, data in self.graph.edges(data=True)
                           if data["transition"] is not Transition.FORWARD}
        for source, _, data in self.graph.edges(data=True):
            forward = data["transition"] is Transition.FORWARD
            data["cost"] = self._lengths[source] if forward else rules.min_lane_change_distance
            # A lane entered by a lane change may be left forward after any part of it.
            data["distance_lower_bound"] = 0.0 if forward and source in lateral_targets else data["cost"]

    def validate(self, route: LaneRoute, *, lane_change_s: tuple[float, ...] | None = None) -> None:
        """
        Check that the route is legal and, if the rules require it, unambiguous to send.

        :param LaneRoute route: The route to check.
        :param tuple[float, ...] | None lane_change_s: Where each lane change starts on
            its source lane, in route order. By default each starts as soon as the
            route enters the source lane; Apollo's planning decides for the ego.
        :raises InvalidRoute: With the first problem found.
        """
        self._check(route, lane_change_s)
        if self.rules.unambiguous:
            self.poses(route)

    def is_valid(self, route: LaneRoute, *, lane_change_s: tuple[float, ...] | None = None) -> bool:
        """
        Return whether ``validate`` accepts the route.

        :rtype: bool
        """
        try:
            self.validate(route, lane_change_s=lane_change_s)
        except InvalidRoute:
            return False
        return True

    def legs(self, route: LaneRoute, *,
             lane_change_s: tuple[float, ...] | None = None) -> list[RouteLeg]:
        """
        Return the stretch of each lane the route drives along.

        A leg ends at the lane's end before a forward step, where the lane
        change starts before a lane change, and at ``end_s`` on the last lane.
        A lane change covers ``min_lane_change_distance`` of the source lane
        after its leg, landing where the next leg starts.

        :param LaneRoute route: The route.
        :param tuple[float, ...] | None lane_change_s: As in ``validate``.
        :raises InvalidRoute: If the route is illegal.
        :rtype: list[RouteLeg]
        """
        lengths, entries, changes = self._check(route, lane_change_s)
        ends = [lengths[index] if change is None else change for index, change in enumerate(changes)]
        return [RouteLeg(lane, start, end)
                for lane, start, end in zip(route.lanes, entries, ends + [route.end_s])]

    def path_sections(self, route: LaneRoute, *,
                      lane_change_s: tuple[float, ...] | None = None) -> list[PathSection]:
        """
        Return the line of ``path`` in sections by lane.

        Each stretch of a lane is a one-lane section. Each lane change is a
        section on its source and target lane, from the end of the source
        stretch to the landing point on the target lane. Consecutive sections
        share their boundary point; the gap between one lane's end and the next
        lane's start belongs to the later section.

        :param LaneRoute route: The route.
        :param tuple[float, ...] | None lane_change_s: As in ``validate``.
        :raises InvalidRoute: If the route is illegal.
        :rtype: list[PathSection]
        """
        legs = self.legs(route, lane_change_s=lane_change_s)
        distance = self.rules.min_lane_change_distance
        sections: list[PathSection] = []
        last: tuple[float, float] | None = None

        def add(lanes: tuple[str, ...], points: Iterable[tuple[float, float]]) -> None:
            nonlocal last
            line = [] if last is None else [last]
            for point in points:
                if not line or line[-1] != point:
                    line.append(point)
            if len(line) > 1:
                sections.append(PathSection(lanes, LineString(line)))
            last = line[-1]

        for index, leg in enumerate(legs):
            line = self.service.get_lane_central_curve_by_id(leg.lane_id)
            if leg.end_s > leg.start_s:
                add((leg.lane_id,), [point[:2] for point in substring(line, leg.start_s, leg.end_s).coords])
            else:
                start = line.interpolate(leg.start_s)
                add((leg.lane_id,), [(start.x, start.y)])
            if index < len(legs) - 1 and route.transitions[index] is not Transition.FORWARD:
                target_id = legs[index + 1].lane_id
                target = self.service.get_lane_central_curve_by_id(target_id)
                landing = legs[index + 1].start_s
                blend = []
                for step in range(1, _BLEND_STEPS):
                    u = step / _BLEND_STEPS
                    weight = u * u * (3 - 2 * u)              # Smoothstep: no kink at either end.
                    a = line.interpolate(leg.end_s + u * distance)
                    b = target.interpolate(max(landing - (1 - u) * distance, 0.0))
                    blend.append(((1 - weight) * a.x + weight * b.x, (1 - weight) * a.y + weight * b.y))
                end = target.interpolate(landing)
                add((leg.lane_id, target_id), blend + [(end.x, end.y)])
        return sections

    def path(self, route: LaneRoute, *,
             lane_change_s: tuple[float, ...] | None = None) -> LineString:
        """
        Return the line the route drives along: the stretch of each lane's
        centerline, joined by lane changes that blend from the source to the
        target lane over ``min_lane_change_distance`` with a smoothstep weight.

        :param LaneRoute route: The route.
        :param tuple[float, ...] | None lane_change_s: As in ``validate``.
        :raises InvalidRoute: If the route is illegal.
        :rtype: LineString
        """
        points: list[tuple[float, float]] = []
        for section in self.path_sections(route, lane_change_s=lane_change_s):
            coords = list(section.line.coords)
            points.extend(coords[1:] if points else coords)
        return LineString(points)

    def lane_change_positions(
        self, route: LaneRoute, rng: Random, *,
        keep: Mapping[tuple[str, str], float] | None = None,
    ) -> tuple[float, ...]:
        """
        Choose where each lane change starts, for agents that drive their own changes.

        Reuse a position in ``keep`` when it leaves a feasible continuation.
        Otherwise draw within the source window, shrinking rejected ranges
        toward a verified earliest continuation. This also handles very narrow
        windows imposed by the end position or consecutive lane changes.

        :param LaneRoute route: The route.
        :param Random rng: Source of randomness.
        :param keep: Positions by (source lane, target lane), e.g. from a parent route.
        :raises InvalidRoute: If no feasible continuation can be found.
        :rtype: tuple[float, ...]
        """
        self._route_lengths(route)
        keep = keep or {}

        def choose(index: int, earliest: float, latest: float) -> float:
            def can_finish(change: float) -> bool:
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

        :raises KeyError: If there is no such edge for these rules.
        :rtype: Transition
        """
        return self.graph[source][target]["transition"]

    def lane_length(self, lane_id: str) -> float:
        """
        Return an allowed lane's centerline length in metres.

        :rtype: float
        """
        return self._lengths[lane_id]

    def entry_after_change(self, source: str, change_s: float, target: str) -> float:
        """
        Return the target-lane s reached by a lane change starting at ``change_s`` on ``source``.

        The change covers ``min_lane_change_distance`` along the source lane;
        the point there is projected onto the target lane.

        :rtype: float
        """
        point = self._pose(source, change_s + self.rules.min_lane_change_distance).point
        return self.service.get_lane_central_curve_by_id(target).project(point)

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
        :raises InvalidRoute: If the route is illegal or cannot be sent unambiguously.
        :rtype: RoutePoses
        """
        lengths, entries, _ = self._check(route)
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

    def length(self, route: LaneRoute, *, lane_change_s: tuple[float, ...] | None = None) -> float:
        """
        Return the driven distance in metres; each lane change counts as
        ``min_lane_change_distance``.

        :param tuple[float, ...] | None lane_change_s: As in ``validate``.
        :raises InvalidRoute: If the route is illegal.
        :rtype: float
        """
        legs = self.legs(route, lane_change_s=lane_change_s)
        return (sum(leg.end_s - leg.start_s for leg in legs)
                + route.lane_changes * self.rules.min_lane_change_distance)

    def random_route(
        self, rng: Random, *, min_length: float = 50.0, max_length: float = 500.0,
        lane_change_probability: float = 0.2, max_attempts: int = 100,
    ) -> LaneRoute:
        """
        Draw a valid route by a random walk of a random target length.

        :param Random rng: Source of randomness; equal seeds give equal routes.
        :param float min_length: Minimum driven distance in metres.
        :param float max_length: Maximum driven distance in metres.
        :param float lane_change_probability: Chance of a lane change where one is possible.
        :param int max_attempts: Walks to try before giving up.
        :raises RuntimeError: If no valid route was found.
        :rtype: LaneRoute
        """
        lanes = sorted(self.graph)
        if lanes:
            for _ in range(max_attempts):
                route = self._walk(rng, rng.choice(lanes), rng.uniform(min_length, max_length),
                                   lane_change_probability)
                if route is not None and self.is_valid(route) and self.length(route) >= min_length:
                    return route
        raise RuntimeError(f"no valid route found in {max_attempts} attempts")

    def alternative_routes(
        self, start: tuple[str, float], end: tuple[str, float], k: int = 5,
    ) -> list[LaneRoute]:
        """
        Return up to ``k`` valid routes between two lane positions, shortest first.

        Candidates are simple lane paths ordered by lane lengths, with each lane
        change costing ``min_lane_change_distance``.

        :param tuple[str, float] start: Start lane ID and s.
        :param tuple[str, float] end: End lane ID and s.
        :param int k: Maximum number of routes.
        :rtype: list[LaneRoute]
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

    def _weight(self, source: str, entry_s: float | None, attribute: str = "distance_lower_bound"):
        """Edge weights for a search from ``source``, entered at ``entry_s``.

        Without ``entry_s`` every edge costs its ``cost``. With it, the source's
        forward edges cost its remaining length and other edges ``attribute``.
        The default, ``distance_lower_bound``, is optimistic: a later lane may be
        entered partway through by a lane change, and charging its full length
        would wrongly prune feasible routes. Zero is a safe lower bound for its
        forward remainder; complete children are validated with their actual
        entries after candidate selection.
        """
        if entry_s is None:
            return lambda u, v, data: data["cost"]
        if not isfinite(entry_s) or not 0 <= entry_s <= self._lengths[source]:
            raise ValueError("entry_s must be within the source lane")
        remaining = self._lengths[source] - entry_s

        def weight(u, v, data):
            if u == source and data["transition"] is Transition.FORWARD:
                return remaining
            return data[attribute]

        return weight

    def reachable(self, source: str, max_cost: float, *, entry_s: float | None = None) -> dict[str, float]:
        """
        Return the lanes reachable from ``source`` within ``max_cost``, with their costs.

        Costs are as in ``lane_paths``; ``source`` itself costs 0.

        :param str source: Starting lane.
        :param float max_cost: Largest path cost to include.
        :param float | None entry_s: If supplied, use lower bounds on driven
            distance from this source position, including partial lane entries.
        :returns: Lane ID to cost; empty if ``source`` is not allowed.
        :rtype: dict[str, float]
        """
        if source not in self.graph:
            return {}
        return nx.single_source_dijkstra_path_length(
            self.graph, source, cutoff=max_cost, weight=self._weight(source, entry_s))

    def lane_paths(
        self, source: str, target: str, k: int = 5, *, max_cost: float | None = None,
        entry_s: float | None = None,
    ) -> Iterator[tuple[tuple[str, ...], tuple[Transition, ...]]]:
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

    def _route_lengths(self, route: LaneRoute) -> list[float]:
        """Check topology and endpoint bounds without choosing lane-change positions."""
        for lane_id in route.lanes:
            if lane_id not in self.graph:
                raise InvalidRoute(f"lane {lane_id!r} is not allowed for these rules")
        if len(set(route.lanes)) != len(route.lanes):
            raise InvalidRoute("a route may not use a lane more than once")
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

    def _check(self, route: LaneRoute, lane_change_s: tuple[float, ...] | None = None,
               ) -> tuple[list[float], list[float], list[float | None]]:
        """Validate topology and progress; return lane lengths, entry positions,
        and each transition's lane-change start (None for forward steps)."""
        lengths = self._route_lengths(route)
        lateral = route.lane_change_indices
        if lane_change_s is not None and len(lane_change_s) != len(lateral):
            raise InvalidRoute(f"lane_change_s needs one position per lane change ({len(lateral)}), "
                               f"got {len(lane_change_s)}")
        given = dict(zip(lateral, lane_change_s or ()))

        def choose(index: int, earliest: float, latest: float) -> float:
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
                 ) -> tuple[list[float], list[float | None]]:
        """Walk the route: where it enters each lane, and where each transition's
        lane change starts (None for forward steps). ``choose(index, earliest,
        latest)`` places the lane change at ``route.transitions[index]``."""
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

    def _change_window(self, source: str, entry: float) -> tuple[float, float]:
        """Earliest and latest start of a lane change from ``source``, entered at ``entry``."""
        distance = self.rules.min_lane_change_distance
        latest = self._lengths[source] - distance
        if latest < entry:
            raise InvalidRoute(f"lane change from {source!r} needs {distance} m "
                               f"but only {self._lengths[source] - entry:.1f} m remain")
        return entry, latest

    def _pose(self, lane_id: str, s: float) -> RoutePose:
        point, heading = self.service.get_lane_coord_and_heading(lane_id, s)
        return RoutePose(lane_id, s, point.x, point.y, heading)

    def _unambiguous(self, pose: RoutePose) -> bool:
        """No lane of any type other than the pose's own lies within the margin."""
        return set(self.service.get_lanes(pose.point, self.rules.ambiguity_margin)) == {pose.lane_id}

    def _waypoint(self, lane_id: str, low: float, high: float, *, early: bool) -> RoutePose | None:
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
        steps = int((high - low) // _WAYPOINT_STEP)
        yield from (low + index * _WAYPOINT_STEP for index in range(steps + 1))
        if low + steps * _WAYPOINT_STEP < high:
            yield high

    def _transitions(self, path: list[str]) -> tuple[Transition, ...]:
        return tuple(self.transition(source, target) for source, target in zip(path, path[1:]))

    def _walk(self, rng: Random, lane_id: str, target: float,
              lane_change_probability: float) -> LaneRoute | None:
        """One random walk of about ``target`` metres; validated by the caller."""
        distance = self.rules.min_lane_change_distance
        start_s = rng.uniform(0, self._lengths[lane_id])
        lanes, transitions = [lane_id], []
        entry, travelled = start_s, 0.0
        while True:
            current = lanes[-1]
            remaining = self._lengths[current] - entry
            if target - travelled <= remaining:
                return LaneRoute(tuple(lanes), tuple(transitions), start_s, entry + target - travelled)
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
                return LaneRoute(tuple(lanes), tuple(transitions), start_s,
                                 rng.uniform(entry, entry + remaining))
            lanes.append(successor)
            transitions.append(self.transition(current, successor))
