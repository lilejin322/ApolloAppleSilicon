"""
Bicycle chromosomes and the trajectories they replay.

A ``BicycleAgent`` rides bike lanes and the virtual connectors that join them
across junctions (``bike_connectors``). Its route, the waits of its two-stage
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
from typing import Iterable, Mapping

from .bike_connectors import DEFAULT_KINDS, ConnectorKind, bike_connectors
from .replay import (
    ReplayLayout, ReplayTrajectory, RouteFrame, Span, SpeedPoint, anchored, check_profile, check_turning_radius,
    curvature_spans, placed_stops, random_draws, rebased, refit_end, replay_layout, replay_path, speed_grid,
)
from .traffic_constraints import check_footprint, check_stops
from .routing import ENDPOINT_JUNCTION_MARGIN, InvalidRoute, LaneRoute, RouteRules, RouteSpace
from ..map import MapService
from ..utils import require_nonnegative

_LENGTH_FITS = 4
"""Times ``random`` refits a route's end to bring its replayed length within bounds."""

PivotKey = tuple[str, int]
"""A pivot by its connector's lane ID and its index among that connector's pivots."""


class RiderType(Enum):
    BICYCLE = "BICYCLE"
    E_BIKE = "E_BIKE"


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
    width: float
    height: float
    sub_type: str
    min_turn_radius: float
    max_speed: float
    max_acceleration: float
    max_deceleration: float
    max_lateral_acceleration: float
    comfortable_deceleration: float


RIDER_SPECS = MappingProxyType({
    RiderType.BICYCLE: RiderSpec(1.8, 0.6, 1.7, "ST_CYCLIST", 2.0, 8.0, 1.0, 3.0, 2.0, 1.5),
    RiderType.E_BIKE: RiderSpec(1.9, 0.7, 1.7, "ST_CYCLIST", 2.5, 11.0, 1.5, 3.5, 2.5, 2.0),
})
"""Dimensions and replay limits per rider type; not calibrated rider dynamics."""


def pivot_keys(space: RouteSpace, route: LaneRoute) -> tuple[PivotKey, ...]:
    """
    Return the route's pivots in riding order.

    :rtype: tuple[tuple[str, int], ...]
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

    :param LaneRoute route: Bike lanes and connectors, with start and end positions.
    :param tuple[float, ...] turn_dwell: Seconds to wait at each pivot of the
        route, in riding order; ``pivot_keys`` names them.
    :param tuple[SpeedPoint, ...] speed_profile: One to four target speeds at
        (lane, s) positions on the route, in riding order; the last speed must
        not be zero. The first point's speed also applies before it.
    :param float start_t: Seconds after scenario start to move off.
    :param RiderType rider_type: Dimensions and limits from ``RIDER_SPECS``.
    """

    route: LaneRoute
    turn_dwell: tuple[float, ...]
    speed_profile: tuple[SpeedPoint, ...]
    start_t: float
    rider_type: RiderType

    def __post_init__(self) -> None:
        check_profile(self.route, self.speed_profile, self.spec.max_speed, self.rider_type.name)
        for dwell in self.turn_dwell:
            require_nonnegative(turn_dwell=dwell)
        require_nonnegative(start_t=self.start_t)

    @property
    def spec(self) -> RiderSpec:
        """
        The rider type's dimensions and limits.

        :rtype: RiderSpec
        """
        return RIDER_SPECS[self.rider_type]

    @property
    def lane_change_s(self) -> tuple[float, ...]:
        """
        Always empty: bicycles never change lanes. Lets code shared with NPC
        vehicles read lane-change positions from either agent.

        :rtype: tuple[float, ...]
        """
        return ()

    @staticmethod
    def rules() -> RouteRules:
        """
        Route rules for bicycles: bike lanes, starts and ends away from
        junctions (``ENDPOINT_JUNCTION_MARGIN``), no ambiguity checks (Apollo
        never matches bicycle poses to lanes).

        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"BIKING"}), unambiguous=False,
                          endpoint_junction_margin=ENDPOINT_JUNCTION_MARGIN)

    @staticmethod
    def space(service: MapService, *, kinds: Iterable[ConnectorKind] = DEFAULT_KINDS,
              max_connector_length: float = 100.0) -> RouteSpace:
        """
        Build the route space every rider type shares: bike lanes and their connectors.

        Connectors are built for the largest turning radius in ``RIDER_SPECS``,
        so every rider type can ride all of them.

        :param MapService service: A map with its lane index and routing graph built.
        :param kinds: The connector kinds to generate.
        :param float max_connector_length: Longest connector in metres.
        :rtype: RouteSpace
        """
        radius = max(spec.min_turn_radius for spec in RIDER_SPECS.values())
        connectors = bike_connectors(service, min_turn_radius=radius, kinds=kinds, max_length=max_connector_length)
        return RouteSpace(service, BicycleAgent.rules(), connectors)

    def turn_waits(self, space: RouteSpace) -> dict[PivotKey, float]:
        """
        Return the wait at each pivot of the route, by pivot.

        :rtype: dict[tuple[str, int], float]
        """
        return dict(zip(pivot_keys(space, self.route), self.turn_dwell))

    def validate(self, space: RouteSpace) -> None:
        """
        Check the route, the turn waits, the speed points, and the path's turning radius.

        :param RouteSpace space: Built with ``BicycleAgent.space``.
        :raises InvalidRoute: With the first problem found.
        """
        self._validated_layout(space)

    def _validated_layout(self, space: RouteSpace) -> tuple[ReplayLayout, list[Span], list[float]]:
        """The replay layout, its curvature spans, and the path metres of each
        speed point, once the path is known to be rideable."""
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

        :raises InvalidRoute: If the route is invalid.
        :raises KeyError: If the lane is not on the route.
        :rtype: float
        """
        return RouteFrame(space, self.route, ()).distance(lane_id, s)

    def route_point(self, space: RouteSpace, distance: float) -> tuple[str, float]:
        """
        The (lane, s) at a driven distance along the route (``route_distance``).

        :raises InvalidRoute: If the route is invalid.
        :rtype: tuple[str, float]
        """
        return RouteFrame(space, self.route, ()).at(distance)

    def trajectory(self, space: RouteSpace) -> ReplayTrajectory:
        """
        Compute the motion this agent replays.

        :param RouteSpace space: As in ``validate``.
        :raises InvalidRoute: If the agent is invalid in this space.
        :rtype: ReplayTrajectory
        """
        layout, curvature, positions = self._validated_layout(space)
        s, v, dwell = speed_grid(layout.path.length, self.speed_profile, positions, self.spec,
                                 curvature_spans=curvature, speed_limits=layout.limits,
                                 forced_stops=dict(zip(layout.pivots, self.turn_dwell)))
        return ReplayTrajectory(layout.path, self.start_t, s, v, dwell, layout.pivots)

    def with_route(self, route: LaneRoute, space: RouteSpace, rng: Random, *,
                   keep: Mapping[PivotKey, float] | None = None,
                   max_turn_dwell: float = 30.0) -> "BicycleAgent":
        """
        Return this agent on another route, e.g. after crossover or mutation.

        Pivots in ``keep`` keep their wait; others get one drawn uniformly in
        ``[0, max_turn_dwell]``. Speed points move as for NPC vehicles
        (``replay.rebased``): a point whose lane is still on the route keeps its
        ``s``, clamped to the lane's new driven stretch; others move to the
        same share of the new route's driven distance.

        :param keep: Waits by pivot; this agent's own by default. A crossover
            child passes both parents'.
        :raises InvalidRoute: If the agent is invalid on the new route.
        :rtype: BicycleAgent
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
               max_attempts: int = 200, min_arrival_t: float = 0.0) -> "BicycleAgent":
        """
        Draw a random valid bicycle of the given rider type.

        :param RouteSpace space: Built with ``BicycleAgent.space``.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param float stop_probability: Chance that an inner speed point is a stop.
            Speed points are drawn at random shares of the route's driven
            distance and anchored to the lane there; stops move off connectors
            and junction lanes where their neighbours leave room.
        :param float max_dwell: Longest stop in seconds.
        :param float max_turn_dwell: Longest wait at a pivot in seconds.
        :param float min_speed: Lowest nonzero target speed in m/s.
        :param float min_length: Minimum length of the final replay path in metres.
        :param float max_length: Maximum length of the final replay path in metres.
        :param float min_arrival_t: Earliest allowed route-end time in scenario seconds;
            set to the scenario duration to prevent early terminal parking. Zero disables it.
        :param int max_attempts: Candidate limit, including rejected path geometry.
        :raises RuntimeError: If the map has no bike lanes or no valid agent was found.
        :rtype: BicycleAgent
        """
        require_nonnegative(min_arrival_t=min_arrival_t)
        if not len(space.graph):
            raise RuntimeError("the map has no bike lanes")
        spec = RIDER_SPECS[rider_type]
        last_problem = "no candidates drawn"
        for _ in range(max_attempts):
            try:
                route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                           lane_change_probability=0.0)
            except RuntimeError as problem:
                last_problem = str(problem)
                continue
            waits = tuple(rng.uniform(0, max_turn_dwell) for _ in pivot_keys(space, route))
            draws = random_draws(rng, spec.max_speed, stop_probability, max_dwell, min_speed)
            agent = cls(route, waits, anchored(RouteFrame(space, route, ()), draws), rng.uniform(0, max_start_t),
                        rider_type)
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

    def _rebased(self, space: RouteSpace, route: LaneRoute) -> tuple[SpeedPoint, ...]:
        """
        This agent's speed points on another route of the same space (``rebased``).

        :raises InvalidRoute: If ``route`` is invalid.
        """
        return rebased(space, self.route, (), self.speed_profile, route, ())

    def _placed_stops(self, space: RouteSpace) -> "BicycleAgent":
        """
        This agent with each stop moved, if it lies on a connector or a
        junction lane, to the nearest riding position off them between its
        neighbours (``placed_stops``). Stops without such a position stay.
        """
        if not any(point.speed == 0 for point in self.speed_profile):
            return self
        frame = RouteFrame(space, self.route, ())

        def blocked(lane_id: str) -> bool:
            return space.virtual_lane(lane_id) is not None or space.is_junction_lane(lane_id)

        try:
            return replace(self, speed_profile=placed_stops(frame, self.speed_profile, (), blocked))
        except ValueError:
            return self

    def to_dict(self) -> dict:
        """
        Return a JSON-compatible representation.

        :rtype: dict
        """
        return {
            "route": self.route.to_dict(),
            "turn_dwell": list(self.turn_dwell),
            "speed_profile": [{"lane_id": point.lane_id, "s": point.s, "speed": point.speed, "dwell": point.dwell}
                              for point in self.speed_profile],
            "start_t": self.start_t,
            "rider_type": self.rider_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BicycleAgent":
        """
        Build an agent from ``to_dict`` output.

        :rtype: BicycleAgent
        """
        return cls(
            LaneRoute.from_dict(data["route"]),
            tuple(data["turn_dwell"]),
            tuple(SpeedPoint(**point) for point in data["speed_profile"]),
            data["start_t"],
            RiderType[data["rider_type"]],
        )
