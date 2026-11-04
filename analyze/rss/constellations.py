"""
RSS constellations on our HD map: what ad-rss-lib leaves to its map
integration (digest §8), rebuilt on ``MapService`` because ``ad_map`` is not
available.

For a pair of road users, from the first one's side: the constellation type
(same direction, opposite direction, intersection with priority, or
unstructured), and the relative constellation the checkers take: both bodies
in the lane coordinates of the first one's lane chain (its current lane, the
lanes behind it and its route ahead), distances, velocity ranges along and
across that chain, whether each is in its correct lane, and for intersections
the distances to enter and leave the area both may occupy.

Simplifications against the library: the connecting route is replaced by lane
relations (one road user's lane on the other's chain, or a forward or reverse
neighbour of it); the other's route is the lanes it actually drives next in the
record; priority follows the spec's rules (signals, then stop signs, then same
priority) instead of ad_map's right-of-way model.
"""

from dataclasses import dataclass
from functools import cached_property
from math import atan2, cos, pi, sin
from typing import Dict, List, Sequence, Tuple
from weakref import WeakKeyDictionary
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring
from shapely.strtree import STRtree

from ...map import MapService
from ...map.geometry import curve_to_line
from .types import (ConstellationType, Dynamics, LatPosition, LonPosition, ObjectType, RelativeConstellation,
                    RelativeObjectState, RssParameters, StructuredState, UnstructuredObjectState, Velocity)
from .unstructured import normalize_angle

__all__ = ["BACK_LENGTH", "LaneInfo", "Party", "RssMap", "relative_constellation", "unstructured_constellation"]

BACK_LENGTH = 100.0
"""Metres of lanes behind a road user its chain keeps, so that what follows it projects onto it."""
AHEAD_LENGTH = 150.0
"""Metres of lanes ahead its chain keeps when it has no route that far."""
_HALF_WIDTH = 1.75
"""Half lane width where a lane has neither boundary curves nor width samples."""


@dataclass(frozen=True)
class Party:
    """
    One road user at one frame.

    :param int id: Its id; the ego's is -1.
    :param ObjectType object_type: ``EGO_VEHICLE``, ``OTHER_VEHICLE``, ``BICYCLE``, ``PEDESTRIAN`` or
        ``OTHER_OBJECT``.
    :param float x: Body centre, map frame.
    :param float y: Body centre.
    :param float heading: Radians.
    :param float speed: m/s.
    :param float length: Metres, along the heading.
    :param float width: Metres, across it.
    :param Polygon body: Its footprint.
    :param Tuple[str, ...] route: The lanes it drives, in order (the ego's planned route; an obstacle's lanes as
        recorded).
    :param float yaw_rate: rad/s, counter-clockwise; the record has no steering angle, which stays 0.
    """
    id: int
    """Its id; the ego's is -1."""
    object_type: ObjectType
    """``EGO_VEHICLE``, ``OTHER_VEHICLE``, ``BICYCLE``, ``PEDESTRIAN`` or ``OTHER_OBJECT``."""
    x: float
    """Body centre, map frame."""
    y: float
    """Body centre."""
    heading: float
    """Radians."""
    speed: float
    """m/s."""
    length: float
    """Metres, along the heading."""
    width: float
    """Metres, across it."""
    body: Polygon
    """Its footprint."""
    route: Tuple[str, ...] = ()
    """The lanes it drives, in order (the ego's planned route; an obstacle's lanes as recorded)."""
    yaw_rate: float = 0.0
    """rad/s, counter-clockwise; the record has no steering angle, which stays 0."""


@dataclass(frozen=True)
class LaneInfo:
    """
    Cached lane geometry, connectivity, speed limit and traffic controls used by RSS map queries.

    :param str id: The lane's id.
    :param LineString line: Its centreline, in driving direction.
    :param Polygon surface: The ground it covers (``_surface``).
    :param Tuple[str, ...] successors: The lanes it leads into.
    :param Tuple[str, ...] predecessors: The lanes leading into it.
    :param Tuple[str, ...] forward_neighbors: Its left and right neighbours driving its way.
    :param Tuple[str, ...] reverse_neighbors: Its left and right neighbours driving against it.
    :param float speed_limit: m/s; 0 where the map has none.
    :param str | None junction: The junction covering at least half its centreline; None off junctions.
    :param str turn: The map's turn type: ``NO_TURN``, ``LEFT_TURN``, ``RIGHT_TURN`` or ``U_TURN``.
    :param Tuple[str, ...] signals: The signals controlling it, by explicit overlaps.
    :param bool stop_sign: Whether a stop sign overlaps it.
    """
    id: str
    """The lane's id."""
    line: LineString
    """Its centreline, in driving direction."""
    surface: Polygon
    """The ground it covers (``_surface``)."""
    successors: Tuple[str, ...]
    """The lanes it leads into."""
    predecessors: Tuple[str, ...]
    """The lanes leading into it."""
    forward_neighbors: Tuple[str, ...]
    """Its left and right neighbours driving its way."""
    reverse_neighbors: Tuple[str, ...]
    """Its left and right neighbours driving against it."""
    speed_limit: float
    """m/s; 0 where the map has none."""
    junction: str | None
    """The junction covering at least half its centreline; None off junctions."""
    turn: str
    """The map's turn type: ``NO_TURN``, ``LEFT_TURN``, ``RIGHT_TURN`` or ``U_TURN``."""
    signals: Tuple[str, ...]
    """The signals controlling it, by explicit overlaps."""
    stop_sign: bool
    """Whether a stop sign overlaps it."""

    def heading_at(self, s: float) -> float:
        """
        Estimate the lane heading at an arc-length position.

        :param float s: Metres along the centreline.
        :returns: Heading in radians, using a centred sample clipped to the lane endpoints.
        :rtype: float
        """
        a = self.line.interpolate(max(s - 0.5, 0.0))
        b = self.line.interpolate(min(s + 0.5, self.line.length))
        return atan2(b.y - a.y, b.x - a.x)


class RssMap:
    """
    The lanes of a ``MapService`` as RSS needs them: surfaces, links, junctions, control.

    :param MapService service: The map.
    """
    _built: "WeakKeyDictionary[MapService, RssMap]" = WeakKeyDictionary()
    """Each map service's lanes, built once (``of``)."""
    service: MapService
    """The map."""
    lanes: Dict[str, LaneInfo]
    """Every lane, by id."""
    _ids: List[str]
    """The lane ids, in the order of ``_tree``'s surfaces."""
    _tree: STRtree
    """A spatial index of the lane surfaces."""
    junction_areas: Dict[str, BaseGeometry | None]
    """Each junction's area, by id; None where it has none."""

    @classmethod
    def of(cls, service: MapService) -> "RssMap":
        """
        The map's lanes, built once per ``MapService`` (maps are read-only).

        :param MapService service: The map.
        :returns: Its lanes, built on the first call.
        :rtype: RssMap
        """
        rss_map = cls._built.get(service)
        if rss_map is None:
            rss_map = cls._built[service] = cls(service)
        return rss_map

    def __init__(self, service: MapService) -> None:
        """
        Cache lane surfaces, traffic controls and a spatial index from the supplied map service.

        :param MapService service: The map.
        """
        self.service = service
        junctions = service.relations.junction_areas
        stop_overlaps = {o.id for sign in service.stop_sign_table.values() for o in sign.overlap_id}
        lanes: Dict[str, LaneInfo] = {}
        for lane_id, lane in service.lane_table.items():
            line = service.get_lane_central_curve_by_id(lane_id)
            lanes[lane_id] = LaneInfo(
                lane_id, line, _surface(lane, line),
                tuple(i.id for i in lane.successor_id), tuple(i.id for i in lane.predecessor_id),
                tuple(i.id for i in (*lane.left_neighbor_forward_lane_id, *lane.right_neighbor_forward_lane_id)),
                tuple(i.id for i in (*lane.left_neighbor_reverse_lane_id, *lane.right_neighbor_reverse_lane_id)),
                lane.speed_limit, _junction_of(line, junctions),
                lane.DESCRIPTOR.fields_by_name["turn"].enum_type.values_by_number[lane.turn].name,
                tuple(service.get_signals_for_lane(lane_id)), bool({o.id for o in lane.overlap_id} & stop_overlaps))
        self.lanes = lanes
        self._ids = list(lanes)
        self._tree = STRtree([lanes[i].surface for i in self._ids])
        self.junction_areas = junctions

    def match(self, body: Polygon) -> List[str]:
        """
        The lanes whose surface the body overlaps.

        :param Polygon body: A footprint.
        :returns: Their ids.
        :rtype: List[str]
        """
        return [self._ids[i] for i in self._tree.query(body, predicate="intersects")]

    def main_lane(self, party: Party) -> str | None:
        """
        The lane a road user drives in: of the lanes it overlaps, one on its
        route, then one along its heading, then the most overlapped; a wrong-way
        road user is matched to the lane it drives against, as ad_map does.

        :param Party party: The road user.
        :returns: The lane's id; None when it overlaps no lane.
        :rtype: str | None
        """
        candidates = []
        for lane_id in self.match(party.body):
            lane = self.lanes[lane_id]
            s = lane.line.project(Point(party.x, party.y))
            aligned = abs(normalize_angle(lane.heading_at(s) - party.heading)) < pi / 2
            candidates.append((lane_id in party.route, aligned, lane.surface.intersection(party.body).area, lane_id))
        return max(candidates)[3] if candidates else None

    def against(self, party: Party, lane_id: str) -> bool:
        """
        Whether the road user drives against its lane.

        :param Party party: The road user.
        :param str lane_id: Its lane (``main_lane``).
        :returns: True if its heading is at least 90 degrees off the lane's where it is.
        :rtype: bool
        """
        lane = self.lanes[lane_id]
        s = lane.line.project(Point(party.x, party.y))
        return abs(normalize_angle(lane.heading_at(s) - party.heading)) >= pi / 2

    def chain(self, lane_id: str, route: Sequence[str], against: bool = False) -> Tuple[str, ...]:
        """
        The lanes behind ``lane_id`` (BACK_LENGTH), it, and ahead along ``route``
        or single successors, in driving order; ``against`` its lanes, behind are
        the successors and ahead the single predecessors.

        :param str lane_id: The road user's lane (``main_lane``).
        :param Sequence[str] route: Its route; ignored driving against its lanes.
        :param bool against: Whether it drives against its lanes (``against``).
        :returns: Lane ids, in driving order.
        :rtype: Tuple[str, ...]
        """
        def back(lane: LaneInfo) -> Tuple[str, ...]:
            """
            Return links behind the current lane, reversing connectivity when travelling against it.

            :param LaneInfo lane: The lane.
            :returns: Lane ids: its predecessors, or its successors driving against it.
            :rtype: Tuple[str, ...]
            """
            return lane.successors if against else lane.predecessors

        def front(lane: LaneInfo) -> Tuple[str, ...]:
            """
            Return links ahead of the current lane, reversing connectivity when travelling against it.

            :param LaneInfo lane: The lane.
            :returns: Lane ids: its successors, or its predecessors driving against it.
            :rtype: Tuple[str, ...]
            """
            return lane.predecessors if against else lane.successors
        behind: List[str] = []
        length, current = 0.0, lane_id
        while length < BACK_LENGTH:
            predecessors = [p for p in back(self.lanes[current]) if p in self.lanes]
            if not predecessors or predecessors[0] in behind or predecessors[0] == lane_id:
                break
            current = predecessors[0]
            behind.insert(0, current)
            length += self.lanes[current].line.length
        ahead, length = [lane_id], 0.0
        rest = list(route[route.index(lane_id) + 1:]) if lane_id in route and not against else []
        while length < AHEAD_LENGTH:
            last = self.lanes[ahead[-1]]
            successors = front(last)
            # A route lists parallel lanes too, and changes lanes: the chain takes the
            # next route lane that continues the lane it reached, or, changing lanes
            # first, one that continues a neighbour of it, as a car moving over to turn.
            neighbours = [n for n in last.forward_neighbors if n in self.lanes] if not against else []
            on_route = [l for l in rest if l in successors and l in self.lanes]
            changed = [l for l in rest if any(l in self.lanes[n].successors for n in neighbours)]
            if on_route or changed:
                nxt = (on_route or changed)[0]
                rest = rest[rest.index(nxt) + 1:]
            elif len(successors) == 1 and successors[0] in self.lanes:
                nxt = successors[0]
            else:
                break
            if nxt in ahead or nxt in behind:
                break
            ahead.append(nxt)
            length += self.lanes[nxt].line.length
        return tuple(behind + ahead)

    def chain_line(self, lanes: Sequence[str], against: bool = False) -> LineString:
        """
        The chain's centerline in driving order.

        :param Sequence[str] lanes: The chain (``chain``).
        :param bool against: Whether it is driven against its lanes: each lane's points then run backwards.
        :returns: Its lanes' centrelines joined, repeated points dropped.
        :rtype: LineString
        """
        coords: List[Tuple[float, float]] = []
        for lane_id in lanes:
            points = list(self.lanes[lane_id].line.coords)
            for point in points[::-1] if against else points:
                if not coords or Point(coords[-1]).distance(Point(point)) > 1e-6:
                    coords.append(tuple(point[:2]))
        return LineString(coords)

    @cached_property
    def _family(self) -> Dict[str, Tuple[frozenset, frozenset]]:
        """
        Each lane's forward neighbours (transitively) and reverse neighbours of those.

        :returns: Per lane id, the lanes driving its way across the road (itself included) and those driving
            against it, each with their forward neighbours.
        :rtype: Dict[str, Tuple[frozenset, frozenset]]
        """
        family = {}
        for lane_id in self.lanes:
            forward, frontier = {lane_id}, [lane_id]
            while frontier:
                for n in self.lanes[frontier.pop()].forward_neighbors:
                    if n in self.lanes and n not in forward:
                        forward.add(n)
                        frontier.append(n)
            reverse = {r for f in forward for r in self.lanes[f].reverse_neighbors if r in self.lanes}
            for r in list(reverse):
                frontier = [r]
                while frontier:
                    for n in self.lanes[frontier.pop()].forward_neighbors:
                        if n in self.lanes and n not in reverse:
                            reverse.add(n)
                            frontier.append(n)
            family[lane_id] = (frozenset(forward), frozenset(reverse))
        return family

    def road(self, chain: Sequence[str], against: bool = False) -> Tuple[frozenset, frozenset]:
        """
        The lanes along a chain driving its way, and those driving against it.

        :param Sequence[str] chain: The chain (``chain``).
        :param bool against: Whether it is driven against its lanes: the two sets then swap.
        :returns: ``(along, against)``: lane ids driving the chain's way, and driving against it.
        :rtype: Tuple[frozenset, frozenset]
        """
        forward = frozenset().union(*(self._family[l][0] for l in chain))
        reverse = frozenset().union(*(self._family[l][1] for l in chain))
        return (reverse, forward) if against else (forward, reverse)

    def dynamics(self, party: Party, params: RssParameters, lanes: Sequence[str]) -> Dynamics:
        """
        Select road-user dynamics and apply the configured lane speed-limit factor.

        :param Party party: Road user whose type selects the base dynamics.
        :param RssParameters params: Dynamics and optional speed-limit multiplier.
        :param Sequence[str] lanes: Known lane IDs whose positive speed limits are considered.
        :returns: Base dynamics or a copy with the acceleration speed cap adjusted; pedestrians keep the base.
        :rtype: Dynamics
        """
        base = {ObjectType.EGO_VEHICLE: params.ego, ObjectType.PEDESTRIAN: params.pedestrian,
                ObjectType.BICYCLE: params.bicycle}.get(party.object_type, params.vehicle)
        limits = [self.lanes[l].speed_limit for l in lanes if self.lanes[l].speed_limit > 0]
        if params.speed_limit_factor and limits and party.object_type != ObjectType.PEDESTRIAN:
            return base.with_(max_speed_on_acceleration=max(limits) * params.speed_limit_factor)
        return base


def _surface(lane, line: LineString) -> Polygon:
    """
    Build the lane surface from its boundaries, falling back to a buffered centreline.

    :param Lane lane: The map's lane message.
    :param LineString line: Its centreline.
    :returns: The polygon between its boundaries; else its centreline buffered by its first width samples'
        mean, or ``_HALF_WIDTH`` without them.
    :rtype: Polygon
    """
    try:
        left, right = curve_to_line(lane.left_boundary.curve), curve_to_line(lane.right_boundary.curve)
        polygon = Polygon(list(left.coords) + list(right.coords)[::-1])
        if polygon.is_valid and polygon.area > 0:
            return polygon
    except ValueError:
        pass
    half = (lane.left_sample[0].width + lane.right_sample[0].width) / 2 if lane.left_sample and lane.right_sample \
        else _HALF_WIDTH
    return line.buffer(half, cap_style="flat")


def _junction_of(line: LineString, junctions) -> str | None:
    """
    Return the first junction covering at least half the centreline length, or None.

    :param LineString line: A lane's centreline.
    :param Dict[str, BaseGeometry | None] junctions: Each junction's area, by id (``junction_areas``).
    :returns: The junction's id; None when none covers that much.
    :rtype: str | None
    """
    for junction_id, area in junctions.items():
        if area is not None and line.intersection(area).length >= 0.5 * line.length:
            return junction_id
    return None


# Lane coordinates

def _coordinates(line: LineString, body: Polygon) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    ``(s range, lateral range)`` of a body along ``line``; lateral grows to the right.

    :param LineString line: A chain's centreline (``chain_line``).
    :param Polygon body: A footprint.
    :returns: ``((s_min, s_max), (l_min, l_max))`` over its corners, metres.
    :rtype: Tuple[Tuple[float, float], Tuple[float, float]]
    """
    s_values, l_values = [], []
    for x, y in list(body.exterior.coords)[:-1]:
        s = line.project(Point(x, y))
        a, b = line.interpolate(max(s - 0.05, 0.0)), line.interpolate(min(s + 0.05, line.length))
        tx, ty = b.x - a.x, b.y - a.y
        norm = (tx * tx + ty * ty) ** 0.5 or 1.0
        on = line.interpolate(s)
        left = (tx * (y - on.y) - ty * (x - on.x)) / norm
        s_values.append(s)
        l_values.append(-left)
    return (min(s_values), max(s_values)), (min(l_values), max(l_values))


def _heading(line: LineString, s: float) -> float:
    """
    Estimate heading in radians from centreline points up to half a metre either side of s.

    :param LineString line: A centreline.
    :param float s: Metres along it.
    :returns: Radians.
    :rtype: float
    """
    a, b = line.interpolate(max(s - 0.5, 0.0)), line.interpolate(min(s + 0.5, line.length))
    return atan2(b.y - a.y, b.x - a.x)


def _velocity(line: LineString, party: Party) -> Velocity:
    """
    Speed along and across the chain at the road user (RssObjectConversion.cpp:241-278).

    :param LineString line: The chain's centreline.
    :param Party party: The road user.
    :returns: Its speed along the chain, never negative, and across it, positive to the right; each range a
        single value.
    :rtype: Velocity
    """
    delta = normalize_angle(_heading(line, line.project(Point(party.x, party.y))) - party.heading)
    lon, lat = abs(cos(delta)) * party.speed, sin(delta) * party.speed
    return Velocity(lon, lon, lat, lat)


def _lon_relation(ego: Tuple[float, float], other: Tuple[float, float]) -> Tuple[LonPosition, float]:
    """
    Classify ego longitudinal interval placement and return the nonnegative separation in metres.

    :param Tuple[float, float] ego: The first party's ``(s_min, s_max)`` along the chain.
    :param Tuple[float, float] other: The other's.
    :returns: Where the first is against the other, ``IN_FRONT`` when further along, and the gap; 0 overlapping.
    :rtype: Tuple[LonPosition, float]
    """
    if ego[0] > other[1]:
        return LonPosition.IN_FRONT, ego[0] - other[1]
    if other[0] > ego[1]:
        return LonPosition.AT_BACK, other[0] - ego[1]
    if ego[0] > other[0] and ego[1] > other[1]:
        return LonPosition.OVERLAP_FRONT, 0.0
    if ego[0] < other[0] and ego[1] < other[1]:
        return LonPosition.OVERLAP_BACK, 0.0
    return LonPosition.OVERLAP, 0.0


def _lat_relation(ego: Tuple[float, float], other: Tuple[float, float]) -> Tuple[LatPosition, float]:
    """
    Classify lateral interval placement in the RSS convention and return separation in metres.

    :param Tuple[float, float] ego: The first party's ``(l_min, l_max)`` across the chain, growing to the right.
    :param Tuple[float, float] other: The other's.
    :returns: Where the first is against the other, ``AT_RIGHT`` when to its right, and the gap; 0 overlapping.
    :rtype: Tuple[LatPosition, float]
    """
    if ego[0] > other[1]:
        return LatPosition.AT_RIGHT, ego[0] - other[1]
    if other[0] > ego[1]:
        return LatPosition.AT_LEFT, other[0] - ego[1]
    if ego[0] > other[0] and ego[1] > other[1]:
        return LatPosition.OVERLAP_RIGHT, 0.0
    if ego[0] < other[0] and ego[1] < other[1]:
        return LatPosition.OVERLAP_LEFT, 0.0
    return LatPosition.OVERLAP, 0.0


def _unstructured_state(party: Party) -> UnstructuredObjectState:
    """
    Convert a road user to an unstructured state, retaining its speed and yaw rate.

    :param Party party: The road user.
    :returns: Its pose, size, speed as a single-value range and yaw rate.
    :rtype: UnstructuredObjectState
    """
    return UnstructuredObjectState(party.x, party.y, party.heading, party.length, party.width, party.speed,
                                   party.speed, yaw_rate=party.yaw_rate)


def _wrong_way(rss_map: RssMap, line: LineString, party: Party) -> Tuple[bool, bool]:
    """
    Whether the road user occupies a lane against the chain (negative) and one along it (positive).

    :param RssMap rss_map: The map's lanes.
    :param LineString line: The chain's centreline.
    :param Party party: The road user.
    :returns: ``(negative, positive)``: a lane it overlaps runs more than 90 degrees off the chain; one does not.
    :rtype: Tuple[bool, bool]
    """
    negative = positive = False
    for lane_id in rss_map.match(party.body):
        lane = rss_map.lanes[lane_id]
        s = lane.line.project(Point(party.x, party.y))
        chain_heading = _heading(line, line.project(Point(party.x, party.y)))
        if abs(normalize_angle(lane.heading_at(s) - chain_heading)) > pi / 2:
            negative = True
        else:
            positive = True
    return negative, positive


def relative_constellation(rss_map: RssMap, first: Party, second: Party, lights: Dict[str, str],
                           params: RssParameters, constellation_id: int = 0) -> RelativeConstellation | None:
    """
    The constellation of ``second`` from ``first``'s side (``first`` takes the ego's role).

    :param RssMap rss_map: The map's lanes.
    :param Party first: The party in the ego's role.
    :param Party second: The other.
    :param Dict[str, str] lights: Colour name of each reported signal, e.g. ``{"signal_9": "RED"}``.
    :param RssParameters params: RSS dynamics.
    :param int constellation_id: The id the checkers keep its history by.
    :returns: The relative constellation, unstructured with a pedestrian or off lanes; None when their lanes
        are not related and their routes do not cross in a junction.
    :rtype: RelativeConstellation | None
    """
    first_type = ObjectType.EGO_VEHICLE  # The checkers' "ego" is always the first party.
    first_lane, second_lane = rss_map.main_lane(first), rss_map.main_lane(second)
    if (first.object_type == ObjectType.PEDESTRIAN or second.object_type == ObjectType.PEDESTRIAN
            or first_lane is None or second_lane is None):
        return unstructured_constellation(rss_map, first, second, params, constellation_id)
    # A wrong-way road user's chain runs the way it drives (ad_map orients its route so).
    first_against, second_against = rss_map.against(first, first_lane), rss_map.against(second, second_lane)
    chain = rss_map.chain(first_lane, first.route, first_against)
    forward, reverse = rss_map.road(chain, first_against)
    line = rss_map.chain_line(chain, first_against)
    other_chain = rss_map.chain(second_lane, second.route, second_against)
    first_dynamics = rss_map.dynamics(first, params, [first_lane])
    second_dynamics = rss_map.dynamics(second, params, [second_lane])
    heading_gap = abs(normalize_angle(second.heading - first.heading))
    if heading_gap < pi / 2 and (second_lane in forward
                                 or first_lane in rss_map.road(other_chain, second_against)[0]):
        kind = ConstellationType.SAME_DIRECTION
    elif (second_lane in reverse or second_lane in forward) and heading_gap >= pi / 2 \
            and not _junction_between(rss_map, chain, line, first, second):
        kind = ConstellationType.OPPOSITE_DIRECTION
    else:
        return _intersection(rss_map, first, second, line, other_chain,
                             rss_map.chain_line(other_chain, second_against), chain, lights, first_dynamics,
                             second_dynamics, constellation_id)
    first_s, first_l = _coordinates(line, first.body)
    second_s, second_l = _coordinates(line, second.body)
    lon_position, lon_distance = _lon_relation(first_s, second_s)
    lat_position, lat_distance = _lat_relation(first_l, second_l)
    first_negative, _ = _wrong_way(rss_map, line, first)
    second_negative, second_positive = _wrong_way(rss_map, line, second)
    second_correct = not second_positive if kind == ConstellationType.OPPOSITE_DIRECTION else not second_negative
    return RelativeConstellation(
        constellation_id, kind,
        RelativeObjectState(first_type, first_dynamics,
                            StructuredState(_velocity(line, first), is_in_correct_lane=not first_negative)),
        RelativeObjectState(second.object_type, second_dynamics,
                            StructuredState(_velocity(line, second), is_in_correct_lane=second_correct)),
        lon_position, lon_distance, lat_position, lat_distance, first.id, second.id)


def unstructured_constellation(rss_map: RssMap, first: Party, second: Party, params: RssParameters,
                               constellation_id: int = 0) -> RelativeConstellation:
    """
    The pair in the plane, from ``first``'s side; a pedestrian keeps its own role and trajectory sets.

    :param RssMap rss_map: The map's lanes.
    :param Party first: The party in the ego's role, unless a pedestrian.
    :param Party second: The other.
    :param RssParameters params: RSS dynamics.
    :param int constellation_id: The id the checkers keep its history by.
    :returns: The unstructured constellation, each party's dynamics under its lane's speed limit.
    :rtype: RelativeConstellation
    """
    lanes = [rss_map.main_lane(p) for p in (first, second)]
    first_type = ObjectType.PEDESTRIAN if first.object_type == ObjectType.PEDESTRIAN else ObjectType.EGO_VEHICLE
    return RelativeConstellation(
        constellation_id, ConstellationType.UNSTRUCTURED,
        RelativeObjectState(first_type, rss_map.dynamics(first, params, [l for l in lanes[:1] if l]),
                            unstructured=_unstructured_state(first)),
        RelativeObjectState(second.object_type, rss_map.dynamics(second, params, [l for l in lanes[1:] if l]),
                            unstructured=_unstructured_state(second)),
        ego_id=first.id, object_id=second.id)


def _junction_between(rss_map: RssMap, chain: Sequence[str], line: LineString, first: Party, second: Party) -> bool:
    """
    Return whether a junction lane intersects the path segment between the two road users.

    :param RssMap rss_map: The map's lanes.
    :param Sequence[str] chain: The first party's chain.
    :param LineString line: Its centreline.
    :param Party first: The party in the ego's role.
    :param Party second: The other.
    :returns: True if a junction lane of the chain comes within 0.5 m of the stretch between their projections.
    :rtype: bool
    """
    a = line.project(Point(first.x, first.y))
    b = line.project(Point(second.x, second.y))
    between = substring(line, min(a, b), max(a, b))
    return any(rss_map.lanes[l].junction and rss_map.lanes[l].line.intersects(between.buffer(0.5))
               for l in chain if rss_map.lanes[l].junction)


def _intersection(rss_map: RssMap, first: Party, second: Party, first_line: LineString, other_chain: Sequence[str],
                  second_line: LineString, chain: Sequence[str], lights: Dict[str, str], first_dynamics: Dynamics,
                  second_dynamics: Dynamics, constellation_id: int) -> RelativeConstellation | None:
    """
    Both routes through one junction on lanes whose surfaces overlap: an intersection constellation.

    :param RssMap rss_map: The map's lanes.
    :param Party first: The party in the ego's role.
    :param Party second: The other.
    :param LineString first_line: The first's chain's centreline.
    :param Sequence[str] other_chain: The other's chain.
    :param LineString second_line: Its centreline.
    :param Sequence[str] chain: The first's chain.
    :param Dict[str, str] lights: Colour name of each reported signal, e.g. ``{"signal_9": "RED"}``.
    :param Dynamics first_dynamics: The first's dynamics.
    :param Dynamics second_dynamics: The other's.
    :param int constellation_id: The id the checkers keep its history by.
    :returns: The constellation at the conflict area the first enters soonest that neither has left; None
        without one.
    :rtype: RelativeConstellation | None
    """
    best = None
    for a in chain:
        lane_a = rss_map.lanes[a]
        if lane_a.junction is None:
            continue
        for b in other_chain:
            lane_b = rss_map.lanes[b]
            if lane_b.junction != lane_a.junction or a == b:
                continue
            area = lane_a.surface.intersection(lane_b.surface)
            if area.is_empty or area.area < 0.5:
                continue
            first_range = _area_range(first_line, area, first.width)
            second_range = _area_range(second_line, area, second.width)
            if first_range is None or second_range is None:
                continue
            first_s, _ = _coordinates(first_line, first.body)
            second_s, _ = _coordinates(second_line, second.body)
            if first_s[0] > first_range[1] or second_s[0] > second_range[1]:
                continue  # One has left this conflict already.
            enter_first = first_range[0] - first_s[1]
            if best is None or enter_first < best[0]:
                best = (enter_first, a, b, first_range, second_range, first_s, second_s)
    if best is None:
        return None
    _, a, b, first_range, second_range, first_s, second_s = best
    kind = _priority(rss_map, chain, a, other_chain, b, lights)
    first_i = (first_range[0] - first_s[1], first_range[0] - first_s[0])
    second_i = (second_range[0] - second_s[1], second_range[0] - second_s[0])
    position, distance = _intersection_order(first_i, second_i)

    def state(party: Party, dynamics: Dynamics, line: LineString, s_range, area_range, priority: bool):
        """
        Build intersection-relative velocity, priority and nonnegative entry/exit distances for one party.

        :param Party party: The party.
        :param Dynamics dynamics: Its dynamics.
        :param LineString line: Its chain's centreline.
        :param Tuple[float, float] s_range: Its body's ``(s_min, s_max)`` along it.
        :param Tuple[float, float] area_range: The conflict area's.
        :param bool priority: Whether it has priority.
        :returns: Its state, in the ego's role for ``first``.
        :rtype: RelativeObjectState
        """
        return RelativeObjectState(
            ObjectType.EGO_VEHICLE if party is first else party.object_type, dynamics,
            StructuredState(_velocity(line, party), has_priority=priority,
                            distance_to_enter_intersection=max(0.0, area_range[0] - s_range[1]),
                            distance_to_leave_intersection=max(0.0, area_range[1] - s_range[0])))
    return RelativeConstellation(
        constellation_id, kind,
        state(first, first_dynamics, first_line, first_s, first_range,
              kind == ConstellationType.INTERSECTION_EGO_HAS_PRIORITY),
        state(second, second_dynamics, second_line, second_s, second_range,
              kind == ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY),
        position, distance, LatPosition.OVERLAP, 0.0, first.id, second.id)


def _area_range(line: LineString, area, width: float) -> Tuple[float, float] | None:
    """
    Project the area intersecting a width-metre path band to an arc-length range, or None.

    :param LineString line: A chain's centreline.
    :param BaseGeometry area: The conflict area.
    :param float width: Metres: the road user's width.
    :returns: ``(s_min, s_max)`` of the area within the band; None when the band misses it or holds no polygon.
    :rtype: Tuple[float, float] | None
    """
    band = line.buffer(width / 2, cap_style="flat").intersection(area)
    if band.is_empty:
        return None
    points = [p for geom in getattr(band, "geoms", (band,)) if hasattr(geom, "exterior")
              for p in geom.exterior.coords]
    if not points:
        return None
    s_values = [line.project(Point(p)) for p in points]
    return min(s_values), max(s_values)


def _intersection_order(ego: Tuple[float, float], other: Tuple[float, float]) -> Tuple[LonPosition, float]:
    """
    Order by distance to the entry, smaller first (RssSituationExtraction.cpp:61-93).

    :param Tuple[float, float] ego: The first party's distances to the area's entry, from its front and its rear.
    :param Tuple[float, float] other: The other's.
    :returns: ``IN_FRONT`` when the first is nearer the entry, and the gap; 0 overlapping.
    :rtype: Tuple[LonPosition, float]
    """
    if ego[1] < other[0]:
        return LonPosition.IN_FRONT, other[0] - ego[1]
    if other[1] < ego[0]:
        return LonPosition.AT_BACK, ego[0] - other[1]
    if ego[0] < other[0] and ego[1] < other[1]:
        return LonPosition.OVERLAP_FRONT, 0.0
    if ego[0] > other[0] and ego[1] > other[1]:
        return LonPosition.OVERLAP_BACK, 0.0
    return LonPosition.OVERLAP, 0.0


def _approach(rss_map: RssMap, chain: Sequence[str], junction_lane: str) -> List[str]:
    """
    The junction lane and the lane leading into it on the chain.

    :param RssMap rss_map: The map's lanes; unused.
    :param Sequence[str] chain: A chain.
    :param str junction_lane: A junction lane on it.
    :returns: Lane ids: the junction lane, then the one before it on the chain, if any.
    :rtype: List[str]
    """
    index = list(chain).index(junction_lane)
    return [junction_lane] + ([chain[index - 1]] if index > 0 else [])


def _priority(rss_map: RssMap, chain: Sequence[str], a: str, other_chain: Sequence[str], b: str,
              lights: Dict[str, str]) -> ConstellationType:
    """
    The spec's rules: signals (green over not green; both green: an unprotected
    left or U turn yields to straight and right), else a stop sign on one
    approach only yields, else same priority.

    :param RssMap rss_map: The map's lanes.
    :param Sequence[str] chain: The first party's chain.
    :param str a: Its junction lane at the conflict.
    :param Sequence[str] other_chain: The other's chain.
    :param str b: Its junction lane at the conflict.
    :param Dict[str, str] lights: Colour name of each reported signal, e.g. ``{"signal_9": "RED"}``.
    :returns: ``INTERSECTION_EGO_HAS_PRIORITY``, ``INTERSECTION_OBJECT_HAS_PRIORITY`` or
        ``INTERSECTION_SAME_PRIORITY``, from the first party's side.
    :rtype: ConstellationType
    """
    first_signals = {s for l in _approach(rss_map, chain, a) for s in rss_map.lanes[l].signals}
    second_signals = {s for l in _approach(rss_map, other_chain, b) for s in rss_map.lanes[l].signals}
    if first_signals or second_signals:
        first_green = bool(first_signals) and all(lights.get(s) == "GREEN" for s in first_signals)
        second_green = bool(second_signals) and all(lights.get(s) == "GREEN" for s in second_signals)
        if first_green != second_green:
            return (ConstellationType.INTERSECTION_EGO_HAS_PRIORITY if first_green
                    else ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY)
        if first_green:
            first_turns = rss_map.lanes[a].turn in ("LEFT_TURN", "U_TURN")
            second_turns = rss_map.lanes[b].turn in ("LEFT_TURN", "U_TURN")
            if first_turns != second_turns:
                return (ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY if first_turns
                        else ConstellationType.INTERSECTION_EGO_HAS_PRIORITY)
        return ConstellationType.INTERSECTION_SAME_PRIORITY
    first_stop = any(rss_map.lanes[l].stop_sign for l in _approach(rss_map, chain, a))
    second_stop = any(rss_map.lanes[l].stop_sign for l in _approach(rss_map, other_chain, b))
    if first_stop != second_stop:
        return (ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY if first_stop
                else ConstellationType.INTERSECTION_EGO_HAS_PRIORITY)
    return ConstellationType.INTERSECTION_SAME_PRIORITY
