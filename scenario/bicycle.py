"""
Bicycle chromosomes and the trajectories they replay.

A ``BicycleAgent`` rides bike lanes and the virtual connectors that join them
across junctions (``bike_connectors``), and the road where there is no bike
lane: the curb-side driving lane (``curb_lanes``), near its right edge
(``CURB_CLEARANCE``). On maps with bike lanes most routes start on one
(``BIKE_LANE_START_SHARE``) and may carry on along the road where it ends;
on maps without, every route rides the road. Its route, the waits of its two-stage
left turns, its rider type, its speed profile and its start time are genes.
Like an NPC vehicle it ignores signals and the ego: it waits at its start
pose until ``start_t``, follows its speed profile within the rider's limits,
stops at each pivot of its connectors for its wait, and stops at the end of
its route, where it stays. A bicycle manager will publish ``state_at(t)``.
"""

from dataclasses import dataclass, replace
from enum import Enum
from random import Random
from types import MappingProxyType
from typing import Iterable, Mapping, Dict, FrozenSet, List, Set, Tuple, Self

from .bike_connectors import DEFAULT_KINDS, ConnectorKind, bike_connectors
from .replay import (
    ReplayLayout, ReplayTrajectory, RouteFrame, Span, SpeedPoint, anchored, check_profile, check_turning_radius,
    curvature_spans, drawn_speeds, placed_stops, random_draws, rebased, refit_end, replay_layout, replay_path, speed_grid,
)
from .traffic_constraints import check_footprint, check_stops
from .realism import RIDER_MIX, comfort_limits, draw_type
from .routing import ENDPOINT_JUNCTION_MARGIN, InvalidRoute, LaneRoute, RouteRules, RouteSpace, junction_lanes
from ..map import MapService
from ..message.descriptor import DescriptorBundle
from ..utils import require_nonnegative

_LENGTH_FITS = 4
"""Times ``random`` refits a route's end to bring its replayed length within bounds."""

CURB_CLEARANCE = 0.8
"""Metres from a driving lane's right edge to a cyclist's centreline: riders keep
about half a metre between their right side and the curb or parked cars
(e.g. AASHTO Guide for the Development of Bicycle Facilities, 2012, §4.2)."""

BIKE_LANE_START_SHARE = 0.8
"""Default chance of trying bike-lane starts first when both bike and curb starts are available."""

PivotKey = Tuple[str, int]
"""A pivot by its connector's lane ID and its index among that connector's pivots."""


class RiderType(Enum):
    """
    Rider categories selecting bicycle dimensions, motion limits, and speed priors.
    """
    BICYCLE = "BICYCLE"
    """Normal bicycle"""
    E_BIKE = "E_BIKE"
    """E-bicycle"""


def random_rider_type(rng: Random, weights: Mapping[str, float] = RIDER_MIX) -> RiderType:
    """
    A rider type drawn from a composition (type name -> weight); by default
    about one e-bike in six (``realism.RIDER_MIX``).

    :param Random rng: Random number generator used for sampling and selection.
    :param Mapping[str, float] weights: Relative nonnegative sampling weights keyed by type name.
    :returns: The sampled rider category.
    :rtype: RiderType
    """
    return RiderType[draw_type(rng, weights)]


@dataclass(frozen=True)
class RiderSpec:
    """
    Fixed properties of a rider type; Apollo's obstacle type is BICYCLE.

    :param float length: Metres.
    :param float width: Metres.
    :param float height: Metres.
    :param str sub_type: Apollo ``PerceptionObstacle.SubType`` name.
    :param float min_turn_radius: Metres, on the path apart from pivots.
    :param float max_speed: m/s.
    :param float max_acceleration: m/s^2.
    :param float max_deceleration: m/s^2, positive.
    :param float max_lateral_acceleration: m/s^2; cap on speed squared times
        the curvature of the replayed heading profile.
    :param float comfortable_deceleration: m/s^2, positive; how hard the rider
        brakes for stops, pivots, curves and the route end.
    """

    length: float
    """Metres."""
    width: float
    """Metres."""
    height: float
    """Metres."""
    sub_type: str
    """Apollo ``PerceptionObstacle.SubType`` name."""
    min_turn_radius: float
    """Metres, on the path apart from pivots."""
    max_speed: float
    """m/s."""
    max_acceleration: float
    """m/s^2."""
    max_deceleration: float
    """m/s^2, positive."""
    max_lateral_acceleration: float
    """m/s^2; cap on speed squared times the curvature of the replayed heading profile."""
    comfortable_deceleration: float
    """m/s^2, positive; how hard the rider brakes for stops, pivots, curves and the route end."""


RIDER_SPECS = MappingProxyType({
    RiderType.BICYCLE: RiderSpec(1.8, 0.6, 1.7, "ST_CYCLIST", 2.0, 8.0, 1.0, 3.0, 2.0, 1.5),
    RiderType.E_BIKE: RiderSpec(1.9, 0.7, 1.7, "ST_CYCLIST", 2.5, 11.0, 1.5, 3.5, 2.5, 2.0),
})
"""Dimensions and replay limits per rider type; not calibrated rider dynamics."""


def _lane_type_numbers(service: MapService) -> Dict[str, int]:
    """
    Read Apollo lane-type enum numbers from the map descriptor.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :returns: Apollo lane-type names mapped to their enum numbers.
    :rtype: Dict[str, int]
    """
    values = DescriptorBundle(service.descriptor_path).message(
        "apollo.hdmap.Lane").DESCRIPTOR.fields_by_name["type"].enum_type.values_by_name
    return {name: value.number for name, value in values.items()}


def curb_lanes(service: MapService) -> FrozenSet[str]:
    """
    The driving lanes a bicycle may ride where there is no bike lane: the
    curb-side lane of each road, i.e. a CITY_DRIVING lane outside junctions
    with no driving lane and no bike lane on its right (a rider beside a bike
    lane uses the bike lane), and the junction lanes that join two of them
    (no U-turns).

    :param MapService service: A map with its routing graph built.
    :returns: IDs of curb road lanes and eligible junction lanes joining them.
    :rtype: FrozenSet[str]
    """
    numbers = _lane_type_numbers(service)
    city, biking = numbers["CITY_DRIVING"], numbers["BIKING"]
    lanes, graph = service.lane_table, service.routing_graph
    junction = junction_lanes(service)
    u_turn = DescriptorBundle(service.descriptor_path).message("apollo.hdmap.Lane").DESCRIPTOR.fields_by_name[
        "turn"].enum_type.values_by_name["U_TURN"].number
    # A bike lane beside a car lane may be declared on either lane.
    beside_bike = {other.id for lane in lanes.values() if lane.type == biking
                   for other in lane.left_neighbor_forward_lane_id}
    curb = {lane_id for lane_id, lane in lanes.items()
            if lane.type == city and lane_id not in junction and lane_id not in beside_bike
            and not any(other.id in lanes and lanes[other.id].type in (city, biking)
                        for other in lane.right_neighbor_forward_lane_id)}

    def forward(lane_id: str, reverse: bool = False) -> Set[str]:
        """
        Find adjacent forward driving lanes, optionally following edges in reverse.

        :param str lane_id: Map lane ID.
        :param bool reverse: Whether to follow predecessor edges rather than successor edges.
        :returns: Forward-connected lane IDs.
        :rtype: Set[str]
        """
        edges = graph.in_edges(lane_id, data=True) if reverse else graph.out_edges(lane_id, data=True)
        return {(source if reverse else target) for source, target, data in edges if data["direction_type"] == "F"}

    joins = {lane_id for lane_id, lane in lanes.items()
             if lane.type == city and lane_id in junction and lane.turn != u_turn and lane_id in graph
             and forward(lane_id, reverse=True) & curb and forward(lane_id) & curb}
    return frozenset(curb | joins)


def curb_offsets(service: MapService, lanes: Iterable[str]) -> Dict[str, float]:
    """
    How far right of each curb lane's centreline a cyclist rides: the lane's
    right half-width less ``CURB_CLEARANCE``, never negative. A junction lane
    takes the mean of the curb lanes it joins, so the line stays continuous.

    :param MapService service: Loaded map service supplying geometry, descriptors, and spatial indexes.
    :param Iterable[str] lanes: Lane IDs in driving order.
    :returns: Rightward riding offsets by curb-lane ID, in metres.
    :rtype: Dict[str, float]
    """
    lanes = set(lanes)
    junction = junction_lanes(service)
    table, graph = service.lane_table, service.routing_graph

    def own(lane_id: str) -> float:
        """
        Compute the usable rightward riding offset on one curb lane.

        :param str lane_id: Map lane ID.
        :returns: The usable rightward riding offset in metres.
        :rtype: float
        """
        samples = table[lane_id].right_sample
        half = sum(sample.width for sample in samples) / len(samples) if samples else 0.0
        return max(half - CURB_CLEARANCE, 0.0)

    offsets = {lane_id: own(lane_id) for lane_id in lanes if lane_id not in junction}
    for lane_id in lanes & junction:
        ends = [other for other in (*graph.predecessors(lane_id), *graph.successors(lane_id)) if other in offsets]
        offsets[lane_id] = sum(offsets[other] for other in ends) / len(ends) if ends else own(lane_id)
    return offsets


def is_bike_lane(space: RouteSpace, lane_id: str) -> bool:
    """
    Whether a lane of the bicycles' space is a map bike lane (not a curb lane or connector).

    :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
    :param str lane_id: Map-lane or virtual-connector ID.
    :returns: ``True`` for a real map bike lane; ``False`` for curb lanes and connectors.
    :rtype: bool
    """
    lane = space.service.lane_table.get(lane_id)
    return lane is not None and space.virtual_lane(lane_id) is None \
        and lane.type == _lane_type_numbers(space.service)["BIKING"]


def starts_on_bike_lane(space: RouteSpace, route: LaneRoute) -> bool:
    """
    Whether the route starts on a bike lane rather than on the road. Genetic
    operators keep this from parent to child (``genetic.bicycle``).


    :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
    :param LaneRoute route: Route to inspect, validate, or convert.
    :returns: ``True`` if the first route lane is a real bike lane.
    :rtype: bool
    """
    return is_bike_lane(space, route.lanes[0])


def pivot_keys(space: RouteSpace, route: LaneRoute) -> Tuple[PivotKey, ...]:
    """
    Return the route's pivots in riding order.

    :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
    :param LaneRoute route: Route to inspect, validate, or convert.
    :returns: Pivot identifiers in riding order.
    :rtype: Tuple[PivotKey, ...]
    """
    keys = []
    for lane_id in route.lanes:
        lane = space.virtual_lane(lane_id)
        if lane is not None:
            keys.extend((lane_id, index) for index in range(len(lane.pivots)))
    return tuple(keys)


@dataclass(frozen=True)
class BicycleAgent:
    """
    Genetic representation of one bicycle.

    :param LaneRoute route: Bike lanes, curb lanes, and connectors, with start and end positions.
    :param Tuple[float, ...] turn_dwell: Seconds to wait at each pivot of the
        route, in riding order; ``pivot_keys`` names them.
    :param Tuple[SpeedPoint, ...] speed_profile: One to four target speeds at
        (lane, s) positions on the route, in riding order; the last speed must
        not be zero. The first point's speed also applies before it.
    :param float start_t: Seconds after scenario start to move off.
    :param RiderType rider_type: Dimensions and limits from ``RIDER_SPECS``.
    :param float drive_style: In [0, 1], from gentle to brisk: the comfortable
        acceleration and braking the replay uses (``realism.COMFORT``).
    """

    route: LaneRoute
    """Bike lanes, curb lanes, and connectors, with start and end positions."""
    turn_dwell: Tuple[float, ...]
    """Seconds to wait at each pivot of the route, in riding order; ``pivot_keys`` names them."""
    speed_profile: Tuple[SpeedPoint, ...]
    """One to four target speeds at (lane, s) positions on the route, in riding order; the last speed must not be zero. The first point's speed also applies before it."""
    start_t: float
    """Seconds after scenario start to move off."""
    rider_type: RiderType
    """Dimensions and limits from ``RIDER_SPECS``."""
    drive_style: float = 0.5
    """In [0, 1], from gentle to brisk: the comfortable acceleration and braking the replay uses (``realism.COMFORT``)."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If input fields or bounds violate the validation conditions of this operation.
        """
        check_profile(self.route, self.speed_profile, self.spec.max_speed, self.rider_type.name)
        for dwell in self.turn_dwell:
            require_nonnegative(turn_dwell=dwell)
        require_nonnegative(start_t=self.start_t)
        if not 0 <= self.drive_style <= 1:
            raise ValueError(f"drive_style must be within [0, 1], got {self.drive_style}")

    @property
    def limits(self) -> RiderSpec:
        """
        The spec with this rider's comfortable acceleration and braking (``drive_style``).

        :returns: The type specification with style-dependent comfortable acceleration and braking.
        :rtype: RiderSpec
        """
        return comfort_limits(self.spec, self.rider_type.name, self.drive_style)

    @property
    def spec(self) -> RiderSpec:
        """
        The rider type's dimensions and limits.

        :returns: The immutable dimensions and hard motion limits of the agent type.
        :rtype: RiderSpec
        """
        return RIDER_SPECS[self.rider_type]

    @property
    def lane_change_s(self) -> Tuple[float, ...]:
        """
        Always empty: bicycles never change lanes. Lets code shared with NPC
        vehicles read lane-change positions from either agent.

        :returns: An empty tuple; this agent does not perform lane changes.
        :rtype: Tuple[float, ...]
        """
        return ()

    @staticmethod
    def rules() -> RouteRules:
        """
        Route rules for bicycles: bike lanes, starts and ends away from
        junctions (``ENDPOINT_JUNCTION_MARGIN``), no ambiguity checks (Apollo
        never matches bicycle poses to lanes).

        :returns: Immutable route rules for this agent category.
        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"BIKING"}), unambiguous=False,
                          endpoint_junction_margin=ENDPOINT_JUNCTION_MARGIN)

    @staticmethod
    def space(service: MapService, *, kinds: Iterable[ConnectorKind] = DEFAULT_KINDS,
              max_connector_length: float = 100.0, roads: bool = True) -> RouteSpace:
        """
        Build the route space every rider type shares: bike lanes and their
        connectors and, if ``roads``, the curb lanes (``curb_lanes``) ridden
        ``curb_offsets`` right of their centrelines, with connectors from bike
        lanes' dead ends onto them.

        Connectors are built for the largest turning radius in ``RIDER_SPECS``,
        so every rider type can ride all of them. Curb lanes are checked per
        agent (``validate``: turning radius, footprint).

        :param MapService service: A map with its lane index and routing graph built.
        :param Iterable[ConnectorKind] kinds: The connector kinds to generate.
        :param float max_connector_length: Longest connector in metres.
        :param bool roads: Whether bicycles may ride curb lanes.
        :returns: The route space shared by the eligible agent types.
        :rtype: RouteSpace
        """
        radius = max(spec.min_turn_radius for spec in RIDER_SPECS.values())
        curb = curb_lanes(service) if roads else frozenset()
        offsets = curb_offsets(service, curb)
        road_space = RouteSpace(service, BicycleAgent.rules(), extra_lanes=curb, lane_offsets=offsets) if curb else None
        road_lines = {lane_id: road_space.centerline(lane_id) for lane_id in curb} if curb else None
        connectors = bike_connectors(service, min_turn_radius=radius, kinds=kinds, max_length=max_connector_length,
                                     road_lanes=road_lines)
        return RouteSpace(service, BicycleAgent.rules(), connectors, extra_lanes=curb, lane_offsets=offsets)

    def turn_waits(self, space: RouteSpace) -> Dict[PivotKey, float]:
        """
        Return the wait at each pivot of the route, by pivot.

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :returns: Pivot identifiers mapped to their dwell times in seconds.
        :rtype: Dict[PivotKey, float]
        """
        return dict(zip(pivot_keys(space, self.route), self.turn_dwell))

    def validate(self, space: RouteSpace) -> None:
        """
        Check the route, the turn waits, the speed points, and the path's turning radius.

        :param RouteSpace space: Built with ``BicycleAgent.space``.
        :returns: ``None`` if all applicable validation checks pass.
        :rtype: None
        :raises InvalidRoute: With the first problem found.
        """
        self._validated_layout(space)

    def _validated_layout(self, space: RouteSpace) -> Tuple[ReplayLayout, List[Span], List[float]]:
        """
        The replay layout, its curvature spans, and the path metres of each
        speed point, once the path is known to be rideable.

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :returns: The replay layout, its curvature spans, and the path metres of each speed point, once the path is known to be rideable.
        :rtype: Tuple[ReplayLayout, List[Span], List[float]]
        :raises InvalidRoute: If route geometry, topology, point placement, or map constraints fail the checks performed here.
        """
        space.validate(self.route)
        if self.route.lane_changes:
            raise InvalidRoute("bicycles do not change lanes")
        pivots = len(pivot_keys(space, self.route))
        if len(self.turn_dwell) != pivots:
            raise InvalidRoute(f"turn_dwell needs one wait per pivot ({pivots}), got {len(self.turn_dwell)}")
        RouteFrame(space, self.route, ()).check_windows(self.speed_profile)
        layout = replay_layout(space, self.route, ())
        spans = curvature_spans(layout.path, layout.pivots)
        check_turning_radius(spans, self.spec.min_turn_radius, self.rider_type.name)
        positions = layout.positions(space, self.speed_profile)
        check_stops(space, layout, self.speed_profile, positions)
        check_footprint(space, layout.path, self.spec.length, self.spec.width, pivots=layout.pivots)
        return layout, spans, positions

    def route_distance(self, space: RouteSpace, lane_id: str, s: float) -> float:
        """
        Driven distance from the route start to ``s`` on a route lane, clamped
        to the lane's driven stretch (as ``RouteSpace.length``).

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :param str lane_id: ID of a map lane or virtual connector on the route.
        :param float s: Position along the lane or connector, in metres.
        :returns: Travelled distance from the route start, in metres, clamped to the route element window.
        :rtype: float
        :raises InvalidRoute: If the route is invalid.
        :raises KeyError: If the lane is not on the route.
        """
        return RouteFrame(space, self.route, ()).distance(lane_id, s)

    def route_point(self, space: RouteSpace, distance: float) -> Tuple[str, float]:
        """
        The (lane, s) at a driven distance along the route (``route_distance``).

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :param float distance: Travelled distance, in metres.
        :returns: The map-lane or virtual-connector ID and position in metres at the requested travelled distance.
        :rtype: Tuple[str, float]
        :raises InvalidRoute: If the route is invalid.
        """
        return RouteFrame(space, self.route, ()).at(distance)

    def trajectory(self, space: RouteSpace) -> ReplayTrajectory:
        """
        Compute the motion this agent replays.

        :param RouteSpace space: As in ``validate``.
        :returns: The trajectory describing the agent motion over scenario time.
        :rtype: ReplayTrajectory
        :raises InvalidRoute: If the agent is invalid in this space.
        """
        layout, curvature, positions = self._validated_layout(space)
        s, v, dwell = speed_grid(layout.path.length, self.speed_profile, positions, self.limits,
                                 curvature_spans=curvature, speed_limits=layout.limits,
                                 forced_stops=dict(zip(layout.pivots, self.turn_dwell)))
        return ReplayTrajectory(layout.path, self.start_t, s, v, dwell, layout.pivots)

    def with_route(self, route: LaneRoute, space: RouteSpace, rng: Random, *,
                   keep: Mapping[PivotKey, float] | None = None,
                   max_turn_dwell: float = 30.0) -> Self:
        """
        Return this agent on another route, e.g. after crossover or mutation.

        Pivots in ``keep`` keep their wait; others get one drawn uniformly in
        ``[0, max_turn_dwell]``. Speed points move as for NPC vehicles
        (``replay.rebased``): a point whose lane is still on the route keeps its
        ``s``, clamped to the lane's new driven stretch; others move to the
        same share of the new route's driven distance.

        :param LaneRoute route: Route to inspect, validate, or convert.
        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :param Random rng: Random number generator used for sampling and selection.
        :param Mapping[PivotKey, float] | None keep: Waits by pivot; this agent's
            own by default. A crossover child passes both parents'.
        :param float max_turn_dwell: Maximum sampled pivot-turn dwell, in seconds.
        :returns: An agent on the proposed route with its profile and dependent
            route genes adjusted.
        :rtype: BicycleAgent
        :raises InvalidRoute: If the agent is invalid on the new route.
        """
        keep = self.turn_waits(space) if keep is None else keep
        waits = tuple(keep[key] if key in keep else rng.uniform(0, max_turn_dwell)
                      for key in pivot_keys(space, route))
        candidate = replace(self, route=route, turn_dwell=waits, speed_profile=self._rebased(space, route))
        candidate.validate(space)
        return candidate

    @classmethod
    def random(cls, space: RouteSpace, rng: Random, *, rider_type: RiderType, max_start_t: float = 10.0,
               min_length: float = 30.0, max_length: float = 300.0, stop_probability: float = 0.0,
               max_dwell: float = 5.0, max_turn_dwell: float = 30.0, min_speed: float = 1.0,
               max_attempts: int = 200, min_arrival_t: float = 0.0,
               bike_lane_start_share: float = BIKE_LANE_START_SHARE) -> Self:
        """
        Draw a random valid bicycle of the given rider type.

        :param RouteSpace space: Built with ``BicycleAgent.space``.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param RiderType rider_type: Rider category selecting motion limits and speed priors.
        :param float max_start_t: Maximum sampled start time, in scenario seconds.
        :param float min_length: Minimum length of the final replay path in metres.
        :param float max_length: Maximum length of the final replay path in metres.
        :param float stop_probability: Chance that an inner speed point is a stop.
            Speed points are drawn at random shares of the route's driven
            distance and anchored to the lane there; stops move off connectors
            and junction lanes where their neighbours leave room.
        :param float max_dwell: Longest stop in seconds.
        :param float max_turn_dwell: Longest wait at a pivot in seconds.
        :param float min_speed: Lowest nonzero target speed in m/s.
        :param int max_attempts: Maximum generation attempts for each selected start-lane pool,
            including candidates rejected for invalid path geometry.
        :param float min_arrival_t: Earliest allowed route-end time in scenario seconds;
            set to the scenario duration to prevent early terminal parking.
            Zero disables it.
        :param float bike_lane_start_share: Chance, drawn once per call, of trying
            bike-lane starts before curb-lane starts when both are available.
            The other start kind is tried if the preferred one yields no valid agent.
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: BicycleAgent
        :raises ValueError: If input fields or bounds violate the validation
            conditions of this operation.
        :raises RuntimeError: If the space has no lanes or no valid agent was found.
        """
        require_nonnegative(min_arrival_t=min_arrival_t)
        if not 0 <= bike_lane_start_share <= 1:
            raise ValueError(f"bike_lane_start_share must be within [0, 1], got {bike_lane_start_share}")
        if not len(space.graph):
            raise RuntimeError("the map has no lanes a bicycle may ride")
        starts = space.start_lanes()
        bike = [lane_id for lane_id in starts if is_bike_lane(space, lane_id)]
        road = [lane_id for lane_id in starts if not is_bike_lane(space, lane_id)]
        pools: List[List[str] | None] = [None]
        if bike and road:
            # One draw per call: redrawing per attempt would favour whichever
            # kind of start yields valid agents more often. The other kind is
            # tried only if the drawn one yields no agent at all.
            pools = [bike, road] if rng.random() < bike_lane_start_share else [road, bike]
        last_problem = "no candidates drawn"
        for start_lanes in pools:
            try:
                return cls._random_from(space, rng, rider_type, start_lanes, max_start_t, min_length, max_length,
                                        stop_probability, max_dwell, max_turn_dwell, min_speed, max_attempts,
                                        min_arrival_t)
            except RuntimeError as problem:
                last_problem = str(problem)
        raise RuntimeError(last_problem)

    @classmethod
    def _random_from(cls, space: RouteSpace, rng: Random, rider_type: RiderType, start_lanes: List[str] | None,
                     max_start_t: float, min_length: float, max_length: float, stop_probability: float,
                     max_dwell: float, max_turn_dwell: float, min_speed: float, max_attempts: int,
                     min_arrival_t: float) -> Self:
        """
        ``random`` with routes starting on ``start_lanes`` (any start lane if None).

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :param Random rng: Random number generator used for sampling and selection.
        :param RiderType rider_type: Rider category selecting motion limits and speed priors.
        :param List[str] | None start_lanes: Optional allowed starting lane IDs; ``None`` uses all eligible starts.
        :param float max_start_t: Maximum sampled start time, in scenario seconds.
        :param float min_length: Minimum accepted replay path or route length, in metres.
        :param float max_length: Maximum accepted replay path or route length, in metres.
        :param float stop_probability: Probability of drawing an ordinary stop at an eligible inner speed point.
        :param float max_dwell: Maximum sampled stop dwell, in seconds.
        :param float max_turn_dwell: Maximum sampled pivot-turn dwell, in seconds.
        :param float min_speed: Minimum moving target speed, in m/s.
        :param int max_attempts: Maximum number of generation attempts before failure.
        :param float min_arrival_t: Earliest accepted route-end time in scenario seconds; zero disables the horizon check.
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: BicycleAgent
        :raises RuntimeError: If required map data is unavailable or no valid candidate is found within the attempt limit.
        """
        spec = RIDER_SPECS[rider_type]
        last_problem = "no candidates drawn"
        for _ in range(max_attempts):
            try:
                route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                           lane_change_probability=0.0, start_lanes=start_lanes)
            except RuntimeError as problem:
                last_problem = str(problem)
                continue
            waits = tuple(rng.uniform(0, max_turn_dwell) for _ in pivot_keys(space, route))
            draws = random_draws(rng, spec.max_speed, stop_probability, max_dwell, min_speed)
            profile = drawn_speeds(space, anchored(RouteFrame(space, route, ()), draws), rng, rider_type.name,
                                   spec.max_speed, min_speed)
            agent = cls(route, waits, profile, rng.uniform(0, max_start_t), rider_type, rng.random())
            try:
                # Fit the last leg to the length bounds, as for NPC vehicles,
                # then validate the complete path.
                path = replay_path(space, route, ()).path
                target = min(max(space.length(route), min_length), max_length)
                for _ in range(_LENGTH_FITS):
                    if min_length <= path.length <= max_length:
                        break
                    route = refit_end(space, route, path.length, target)
                    agent = replace(agent, route=route, speed_profile=agent._rebased(space, route))
                    path = replay_path(space, route, ()).path
                agent = agent._placed_stops(space)
                path = agent._validated_layout(space)[0].path
            except InvalidRoute as problem:
                last_problem = str(problem)
                continue
            if min_length <= path.length <= max_length:
                if min_arrival_t and (agent.start_t >= min_arrival_t
                                      or agent.trajectory(space).arrival_t < min_arrival_t):
                    last_problem = f"route must start before and finish at or after {min_arrival_t} s"
                    continue
                return agent
            last_problem = f"final path length {path.length} is outside [{min_length}, {max_length}]"
        raise RuntimeError(f"no valid bicycle found in {max_attempts} attempts; last candidate: {last_problem}")

    def _rebased(self, space: RouteSpace, route: LaneRoute) -> Tuple[SpeedPoint, ...]:
        """
        This agent's speed points on another route of the same space (``rebased``).

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :param LaneRoute route: Route to inspect, validate, or convert.
        :returns: The speed profile rebased onto ``route``.
        :rtype: Tuple[SpeedPoint, ...]
        :raises InvalidRoute: If ``route`` is invalid.
        """
        return rebased(space, self.route, (), self.speed_profile, route, ())

    def _placed_stops(self, space: RouteSpace) -> Self:
        """
        This agent with each stop moved, if it lies on a connector or a
        junction lane, to the nearest riding position off them between its
        neighbours (``placed_stops``). Stops without such a position stay.

        :param RouteSpace space: Route space supplying bicycle routing geometry and topology.
        :returns: A copy of this agent with ordinary stops moved to valid riding positions.
        :rtype: BicycleAgent
        """
        if not any(point.speed == 0 for point in self.speed_profile):
            return self
        frame = RouteFrame(space, self.route, ())

        def blocked(lane_id: str) -> bool:
            """
            Check whether a lane is a connector or a junction where ordinary stops are forbidden.

            :param str lane_id: ID of a map lane or virtual connector.
            :returns: ``True`` if the lane is a virtual connector or junction lane.
            :rtype: bool
            """
            return space.virtual_lane(lane_id) is not None or space.is_junction_lane(lane_id)

        try:
            return replace(self, speed_profile=placed_stops(frame, self.speed_profile, (), blocked))
        except ValueError:
            return self

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A new JSON-compatible mapping of the object fields.
        :rtype: Dict
        """
        return {
            "route": self.route.to_dict(),
            "turn_dwell": list(self.turn_dwell),
            "speed_profile": [{"lane_id": point.lane_id, "s": point.s, "speed": point.speed, "dwell": point.dwell}
                              for point in self.speed_profile],
            "start_t": self.start_t,
            "rider_type": self.rider_type.name,
            "drive_style": self.drive_style,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Build an agent from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed object.
        :rtype: BicycleAgent
        :raises KeyError: If a required serialized field is missing or ``rider_type``
            is not a known rider type.
        """
        return cls(
            LaneRoute.from_dict(data["route"]),
            tuple(data["turn_dwell"]),
            tuple(SpeedPoint(**point) for point in data["speed_profile"]),
            data["start_t"],
            RiderType[data["rider_type"]],
            data.get("drive_style", 0.5),
        )
