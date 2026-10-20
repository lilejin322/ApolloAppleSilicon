"""
NPC vehicle chromosomes and the trajectories they replay.

An ``NPCVehicleAgent`` drives a lane route that Apollo never sees: where it
changes lanes, how fast it goes, and its vehicle type are genes. The NPC does
not react to the ego. It waits at its start pose until ``start_t``, follows
its speed profile within the vehicle's acceleration limits, and stops at the
end of its route, where it stays. The motion itself is computed by
``replay``; an NPC manager publishes ``state_at(t)``.
"""

from bisect import bisect_right
from dataclasses import dataclass, replace
from enum import Enum
from math import hypot
from random import Random
from types import MappingProxyType
from typing import Mapping, Sequence
from shapely.geometry import LineString

from .replay import (
    STOP_CLEARANCE, ReplayTrajectory, RouteFrame, SpeedPoint, anchored, check_profile, check_turning_radius,
    curvature_spans, ordered_points, placed_stops, random_draws, rebased, refit_end, replay_layout, replay_path,
    speed_grid, speed_limit, drawn_speeds,
)
from .traffic_constraints import check_footprint, check_stops
from .realism import VEHICLE_MIX, comfort_limits, draw_type
from .routing import ENDPOINT_JUNCTION_MARGIN, InvalidRoute, LaneRoute, RouteRules, RouteSpace
from ..utils import require_nonnegative
from typing import Dict, List, Tuple

LANE_CHANGE_TIME = 3.5
"""Seconds a lane change takes at the NPC's desired speed; sets its length."""
MAX_LANE_CHANGE_DISTANCE = 80.0
"""Longest lane change in metres; the shortest is the route rules' minimum."""
_LENGTH_FITS = 4
"""Times ``random`` refits a route's end to bring its replayed length within bounds."""


class VehicleType(Enum):
    """
    NPC vehicle categories selecting body dimensions, motion limits, and speed priors.
    """
    CAR = "CAR"
    VAN = "VAN"
    TRUCK = "TRUCK"
    BUS = "BUS"


def random_vehicle_type(rng: Random, weights: Mapping[str, float] = VEHICLE_MIX) -> "VehicleType":
    """
    A vehicle type drawn from a traffic composition (type name -> weight);
    by default urban US vehicle-miles (``realism.VEHICLE_MIX``).

    :param Random rng: Random number generator used for sampling and selection.
    :param Mapping[str, float] weights: Relative nonnegative sampling weights keyed by type name.
    :returns: The sampled vehicle category.
    :rtype: VehicleType
    """
    return VehicleType[draw_type(rng, weights)]


@dataclass(frozen=True)
class VehicleSpec:
    """
    Fixed properties of a vehicle type.

    :param float length: Metres.
    :param float width: Metres.
    :param float height: Metres.
    :param str sub_type: Apollo ``PerceptionObstacle.SubType`` name; the type is VEHICLE.
    :param float min_turn_radius: Metres; tighter U-turn lanes are excluded.
    :param float max_speed: m/s.
    :param float max_acceleration: m/s^2.
    :param float max_deceleration: m/s^2, positive.
    :param float max_lateral_acceleration: m/s^2; cap on speed squared times
        the curvature of the replayed heading profile.
    :param float comfortable_deceleration: m/s^2, positive; how hard the NPC
        brakes for stops, curves, limits and the route end. Never above
        ``max_deceleration``, which stays the hard limit.
    """

    length: float
    """Metres."""
    width: float
    """Metres."""
    height: float
    """Metres."""
    sub_type: str
    """Apollo ``PerceptionObstacle.SubType`` name; the type is VEHICLE."""
    min_turn_radius: float
    """Metres; tighter U-turn lanes are excluded."""
    max_speed: float
    """m/s."""
    max_acceleration: float
    """m/s^2."""
    max_deceleration: float
    """m/s^2, positive."""
    max_lateral_acceleration: float = 2.5
    """m/s^2; cap on speed squared times the curvature of the replayed heading profile."""
    comfortable_deceleration: float = 2.5
    """m/s^2, positive; how hard the NPC brakes for stops, curves, limits and the route end. Never above ``max_deceleration``, which stays the hard limit."""


VEHICLE_SPECS = MappingProxyType({
    VehicleType.CAR: VehicleSpec(4.7, 1.9, 1.5, "ST_CAR", 5.5, 20.0, 3.0, 6.0),
    VehicleType.VAN: VehicleSpec(5.5, 2.0, 2.2, "ST_VAN", 6.5, 18.0, 2.5, 5.5),
    VehicleType.TRUCK: VehicleSpec(9.0, 2.5, 3.5, "ST_TRUCK", 10.0, 15.0, 1.5, 4.5),
    VehicleType.BUS: VehicleSpec(12.0, 2.55, 3.2, "ST_BUS", 10.5, 15.0, 1.5, 4.5),
})
"""Default dimensions and limits per vehicle type."""


SPEED_PROFILE_VERSION = 2
"""``to_dict`` format version: speed points anchored to (lane, s) on the route.
Version 1 (no ``version`` key) placed them by fraction of the replay path."""


@dataclass(frozen=True)
class NPCVehicleAgent:
    """
    Genetic representation of one NPC vehicle.

    :param LaneRoute route: City lanes to drive, with start and end positions.
    :param Tuple[float, ...] lane_change_s: Where each lane change starts on its
        source lane, one per lateral transition, in route order.
    :param Tuple[SpeedPoint, ...] speed_profile: One to four target speeds at
        (lane, s) positions on the route, in driving order; the last speed must
        not be zero. The first point's speed also applies before it.
    :param float start_t: Seconds after scenario start to move off.
    :param VehicleType vehicle_type: Dimensions and limits from ``VEHICLE_SPECS``.
    :param float drive_style: In [0, 1], from gentle to brisk: the comfortable
        acceleration and braking the replay uses (``realism.COMFORT``).
    """

    route: LaneRoute
    """City lanes to drive, with start and end positions."""
    lane_change_s: Tuple[float, ...]
    """Where each lane change starts on its source lane, one per lateral transition, in route order."""
    speed_profile: Tuple[SpeedPoint, ...]
    """One to four target speeds at (lane, s) positions on the route, in driving order; the last speed must not be zero. The first point's speed also applies before it."""
    start_t: float
    """Seconds after scenario start to move off."""
    vehicle_type: VehicleType
    """Dimensions and limits from ``VEHICLE_SPECS``."""
    drive_style: float = 0.5
    """In [0, 1], from gentle to brisk: the comfortable acceleration and braking the replay uses (``realism.COMFORT``)."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If input fields or bounds violate the validation conditions of this operation.
        """
        if len(self.lane_change_s) != self.route.lane_changes:
            raise ValueError(f"lane_change_s needs one position per lane change "
                             f"({self.route.lane_changes}), got {len(self.lane_change_s)}")
        check_profile(self.route, self.speed_profile, self.spec.max_speed, self.vehicle_type.name)
        require_nonnegative(start_t=self.start_t)
        if not 0 <= self.drive_style <= 1:
            raise ValueError(f"drive_style must be within [0, 1], got {self.drive_style}")

    @property
    def spec(self) -> VehicleSpec:
        """
        The vehicle type's dimensions and limits.

        :returns: The immutable dimensions and hard motion limits of the agent type.
        :rtype: VehicleSpec
        """
        return VEHICLE_SPECS[self.vehicle_type]

    @property
    def limits(self) -> VehicleSpec:
        """
        The spec with this agent's comfortable acceleration and braking (``drive_style``).

        :returns: The type specification with style-dependent comfortable acceleration and braking.
        :rtype: VehicleSpec
        """
        return comfort_limits(self.spec, self.vehicle_type.name, self.drive_style)

    @staticmethod
    def rules(vehicle_type: VehicleType) -> RouteRules:
        """
        Route rules for an NPC vehicle: city lanes, no lane or lane joint
        tighter than the type can drive, starts and ends away from junctions
        (``ENDPOINT_JUNCTION_MARGIN``), no ambiguity checks (Apollo never
        matches NPC poses).

        :param VehicleType vehicle_type: The NPC's type.
        :returns: Immutable route rules for this agent category.
        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"CITY_DRIVING"}),
                          min_turn_radius=VEHICLE_SPECS[vehicle_type].min_turn_radius,
                          unambiguous=False, prune_tight_turns=True,
                          endpoint_junction_margin=max(ENDPOINT_JUNCTION_MARGIN,
                                                       hypot(VEHICLE_SPECS[vehicle_type].length,
                                                             VEHICLE_SPECS[vehicle_type].width) / 2))

    def validate(self, space: RouteSpace) -> None:
        """
        Check the route, lane-change positions, and final path's turning radius.

        :param RouteSpace space: Built with ``NPCVehicleAgent.rules`` for this type.
        :returns: ``None`` if all applicable validation checks pass.
        :rtype: None
        :raises ValueError: If the space allows U-turns too tight for this type.
        :raises InvalidRoute: With the first problem found.
        """
        self._validated_path(space)

    def _validated_path(self, space: RouteSpace) -> Tuple[
            LineString, List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        The replay path, its curvature spans and its speed-limit spans, once the path is known to be drivable.

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :returns: The replay path, its curvature spans, and its speed-limit spans.
        :rtype: Tuple[LineString, List[Tuple[float, float, float]], List[Tuple[float, float, float]]]
        :raises ValueError: If the route space is incompatible with this vehicle type.
        :raises InvalidRoute: If route geometry, lane-change placement, speed points,
            stops, turning radius, or footprint checks fail.
        """
        path, spans, limits, _ = self._validated_layout(space)
        return path, spans, limits

    def _validated_layout(self, space: RouteSpace) -> Tuple[
            LineString, List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[float]]:
        """
        ``_validated_path`` and the path metres of each speed point.

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :returns: The replay path, curvature spans, speed-limit spans, and path
            positions of the speed-profile points.
        :rtype: Tuple[LineString, List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[float]]
        :raises ValueError: If the route space is incompatible with this vehicle type.
        :raises InvalidRoute: If route geometry, lane-change placement, speed points,
            stops, turning radius, or footprint checks fail.
        """
        if space.rules.min_turn_radius < self.spec.min_turn_radius:
            raise ValueError(f"the space allows U-turns down to {space.rules.min_turn_radius} m; a "
                             f"{self.vehicle_type.name} needs NPCVehicleAgent.rules({self.vehicle_type})")
        space.validate(self.route, lane_change_s=self.lane_change_s)
        frame = RouteFrame(space, self.route, self.lane_change_s)
        frame.check_windows(self.speed_profile)
        layout = replay_layout(space, self.route, self.lane_change_s, self._lane_change_distances(space, frame))
        spans = curvature_spans(layout.path)
        check_turning_radius(spans, self.spec.min_turn_radius, self.vehicle_type.name)
        positions = layout.positions(space, self.speed_profile)
        check_stops(space, layout, self.speed_profile, positions)
        check_footprint(space, layout.path, self.spec.length, self.spec.width)
        return layout.path, spans, layout.limits, positions

    def route_distance(self, space: RouteSpace, lane_id: str, s: float) -> float:
        """
        Driven distance from the route start to ``s`` on a route lane, clamped
        to the lane's driven stretch; each lane change counts as the rules'
        minimum (as ``RouteSpace.length``).

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :param str lane_id: ID of a lane on the route.
        :param float s: Position along that lane, in metres.
        :returns: Travelled distance from the route start, in metres, clamped to
            the lane's driven window.
        :rtype: float
        :raises InvalidRoute: If the route or lane-change positions are invalid.
        :raises KeyError: If the lane is not on the route.
        """
        return RouteFrame(space, self.route, self.lane_change_s).distance(lane_id, s)

    def route_point(self, space: RouteSpace, distance: float) -> Tuple[str, float]:
        """
        The (lane, s) at a driven distance along the route (``route_distance``);
        inside a lane change's minimum stretch, the nearer end of that stretch.

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :param float distance: Travelled distance, in metres.
        :returns: The lane ID and position in metres at the requested travelled distance.
        :rtype: Tuple[str, float]
        :raises InvalidRoute: If the route or lane-change positions are invalid.
        """
        return RouteFrame(space, self.route, self.lane_change_s).at(distance)

    def speed_point_positions(self, space: RouteSpace) -> List[float]:
        """
        Where each speed point lies on the replay path, in path metres.

        :param RouteSpace space: As in ``validate``.
        :returns: Path distances of the speed-profile points, in metres.
        :rtype: List[float]
        :raises InvalidRoute: If the agent is invalid.
        """
        return self._validated_layout(space)[3]

    def trajectory(self, space: RouteSpace) -> ReplayTrajectory:
        """
        Compute the motion this agent replays.

        :param RouteSpace space: As in ``validate``.
        :returns: The trajectory describing the agent motion over scenario time.
        :rtype: ReplayTrajectory
        :raises InvalidRoute: If the route or lane-change positions are invalid.
        """
        path, curvature, limits, positions = self._validated_layout(space)
        s, v, dwell = speed_grid(path.length, self.speed_profile, positions, self.limits,
                                 curvature_spans=curvature, speed_limits=limits)
        return ReplayTrajectory(path, self.start_t, s, v, dwell)

    def with_route(self, route: LaneRoute, space: RouteSpace, rng: Random, *,
                   keep: Mapping[Tuple[str, str], float] | None = None,
                   attempts: int = 10) -> "NPCVehicleAgent":
        """
        Return this agent on another route, e.g. after crossover or mutation.

        Lane changes between the same lanes keep positions that leave a feasible
        continuation; other positions are sampled with the remaining route checked.
        Speed points whose lane is still on the route keep their ``s``, clamped
        to the lane's new driven stretch; points whose lane left the route move
        to the same share of the new route's driven distance (``_rebased``).

        :param LaneRoute route: Route to inspect, validate, or convert.
        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :param Random rng: Random number generator used for sampling and selection.
        :param Mapping[Tuple[str, str], float] | None keep: Positions to reuse by
            (source lane, target lane); this agent's own by default. A crossover
            child passes both parents'.
        :param int attempts: Maximum number of attempts to place a valid proposal.
        :returns: An agent on the proposed route with its profile and dependent
            route genes adjusted.
        :rtype: NPCVehicleAgent
        :raises InvalidRoute: If no valid lane-change positions were found.
        """
        keep = dict(zip(self.route.lane_change_pairs, self.lane_change_s) if keep is None else keep)
        error = InvalidRoute("no valid lane-change positions found")
        for _ in range(attempts):
            try:
                positions = space.lane_change_positions(route, rng, keep=keep)
                candidate = replace(self, route=route, lane_change_s=positions,
                                    speed_profile=self._rebased(space, route, positions))
                candidate.validate(space)
            except InvalidRoute as problem:
                error = problem
                # Positions can fit the route yet yield a path too tight for
                # this vehicle. Retry the geometry without inherited pins.
                keep = {}
                continue
            return candidate
        raise error

    @classmethod
    def random(cls, space: RouteSpace, rng: Random, *, vehicle_type: VehicleType,
               max_start_t: float = 10.0, min_length: float = 150.0, max_length: float = 500.0,
               lane_change_probability: float = 0.2, stop_probability: float = 0.0,
               max_dwell: float = 5.0, min_speed: float = 4.5, max_attempts: int = 200,
               min_arrival_t: float = 0.0,
               ) -> "NPCVehicleAgent":
        """
        Draw a random valid NPC of the given type.

        :param RouteSpace space: Built with ``NPCVehicleAgent.rules(vehicle_type)``.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param VehicleType vehicle_type: Vehicle category selecting dimensions and motion limits.
        :param float max_start_t: Maximum sampled start time, in scenario seconds.
        :param float min_length: Minimum length of the final replay path in metres.
            The default keeps an NPC driving for most of a 30 s scenario.
        :param float max_length: Maximum length of the final replay path in metres.
        :param float lane_change_probability: Probability of selecting a lane
            change during random route generation.
        :param float stop_probability: Chance that an inner speed point is a stop.
            Speed points are drawn at random shares of the route's driven
            distance and anchored to the lane there; stops move off junction
            lanes and out of lane changes where their neighbours leave room.
        :param float max_dwell: Longest stop in seconds.
        :param float min_speed: Lowest nonzero target speed in m/s. The default
            keeps NPCs from crawling for most of a 30 s scenario.
        :param int max_attempts: Candidate limit, including rejected path geometry.
        :param float min_arrival_t: Earliest allowed route-end time in scenario seconds;
            set to the scenario duration to prevent early terminal parking. Zero disables it.
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: NPCVehicleAgent
        :raises ValueError: If input fields or bounds violate the validation conditions
            of this operation.
        :raises RuntimeError: If no valid agent is found within ``max_attempts``.
        """
        require_nonnegative(min_arrival_t=min_arrival_t)
        spec = VEHICLE_SPECS[vehicle_type]
        last_problem = "no candidates drawn"
        for _ in range(max_attempts):
            try:
                route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                           lane_change_probability=lane_change_probability)
            except RuntimeError as problem:
                # Sparse spaces (large vehicles on maps of tight junctions)
                # can need more route draws than one call allows.
                last_problem = str(problem)
                continue
            try:
                positions = space.lane_change_positions(route, rng)
            except InvalidRoute as problem:
                last_problem = str(problem)
                continue
            draws = random_draws(rng, spec.max_speed, stop_probability, max_dwell, min_speed)
            frame = RouteFrame(space, route, positions)
            profile = drawn_speeds(space, anchored(frame, draws), rng, vehicle_type.name, spec.max_speed, min_speed)
            agent = cls(route, positions, profile, rng.uniform(0, max_start_t), vehicle_type, rng.random())
            try:
                # Explicit lane changes and joins change the geometric length.
                # Fit the final leg to the original length target when possible,
                # then validate the complete path. Never shift lane-change genes.
                path = replay_path(space, route, positions, agent._lane_change_distances(space)).path
                target = min(max(space.length(route), min_length), max_length)
                # Smoothing and lane-change lengths depend slightly on the end:
                # refit a few times until the replayed length is in bounds.
                for _ in range(_LENGTH_FITS):
                    if min_length <= path.length <= max_length:
                        break
                    route = refit_end(space, route, path.length, target)
                    agent = replace(agent, route=route, speed_profile=agent._rebased(space, route, positions))
                    path = replay_path(space, route, positions, agent._lane_change_distances(space)).path
                agent = agent._placed_stops(space)
                path, _, _ = agent._validated_path(space)
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
        raise RuntimeError(f"no valid NPC found in {max_attempts} attempts; last candidate: {last_problem}")

    def _lane_change_distances(self, space: RouteSpace, frame: RouteFrame | None = None,
                               ) -> Tuple[float, ...]:
        """
        Desired length of each lane change: ``LANE_CHANGE_TIME`` at the speed the
        profile targets where the change starts, below the lanes' limits, within
        the rules' minimum and ``MAX_LANE_CHANGE_DISTANCE``. Positions use the
        route's driven length with minimum-length changes (``RouteFrame``), so
        the result does not depend on itself.

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :param RouteFrame | None frame: Optional precomputed route frame used to
            map lane positions to driven distance.
        :returns: Desired geometric lengths of the route's lane changes, in route order.
        :rtype: Tuple[float, ...]
        """
        if not self.route.lane_changes:
            return ()
        frame = frame or RouteFrame(space, self.route, self.lane_change_s)
        marks = [frame.distance(point.lane_id, point.s) for point in self.speed_profile]
        minimum = space.rules.min_lane_change_distance
        distances = []
        for index in self.route.lane_change_indices:
            start = frame.offsets[index] + frame.ends[index] - frame.starts[index]
            speed = min(self._target_speed(start, marks), self.spec.max_speed)
            limit = speed_limit(space, (self.route.lanes[index], self.route.lanes[index + 1]))
            if limit is not None:
                speed = min(speed, limit)
            distances.append(min(max(LANE_CHANGE_TIME * speed, minimum), max(MAX_LANE_CHANGE_DISTANCE, minimum)))
        return tuple(distances)

    def _target_speed(self, position: float, marks: Sequence[float]) -> float:
        """
        The profile's desired speed at a position, with the points at
        ``marks`` in the same units: a stop's next speed, as in ``replay.speed_grid``.

        :param float position: Position along the replay path, in metres.
        :param Sequence[float] marks: Driven distances of the speed points, in metres.
        :returns: The profile's desired speed at a position, with the points at ``marks`` in the same units: a stop's next speed, as in ``replay.speed_grid``.
        :rtype: float
        """
        index = max(bisect_right(list(marks), position) - 1, 0)
        for point in self.speed_profile[index:]:
            if point.speed > 0:
                return point.speed
        return self.speed_profile[-1].speed

    def _rebased(self, space: RouteSpace, route: LaneRoute, lane_change_s: Tuple[float, ...],
                 ) -> Tuple[SpeedPoint, ...]:
        """
        This agent's speed points on another route of the same space (``rebased``).

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :param LaneRoute route: Route to inspect, validate, or convert.
        :param Tuple[float, ...] lane_change_s: Lane-change start positions on
            source lanes, in route order.
        :returns: The speed profile rebased onto ``route``.
        :rtype: Tuple[SpeedPoint, ...]
        :raises InvalidRoute: If ``route`` or ``lane_change_s`` is invalid.
        """
        return rebased(space, self.route, self.lane_change_s, self.speed_profile, route, lane_change_s)

    def _placed_stops(self, space: RouteSpace) -> "NPCVehicleAgent":
        """
        This agent with each stop moved, if it lies on a junction lane or within
        a lane change's expected length, to the nearest driven position off
        junction lanes and outside lane changes between its neighbours. Stops
        without such a position stay where they are.

        :param RouteSpace space: Route space supplying lane geometry, topology,
            and vehicle-specific constraints.
        :returns: A copy of this agent with ordinary stops moved to valid driven positions.
        :rtype: NPCVehicleAgent
        """
        if not any(point.speed == 0 for point in self.speed_profile):
            return self
        frame = RouteFrame(space, self.route, self.lane_change_s)
        excluded = []
        for index, distance in zip(self.route.lane_change_indices, self._lane_change_distances(space, frame)):
            start = frame.offsets[index] + frame.ends[index] - frame.starts[index]
            excluded.append((start - STOP_CLEARANCE, start + distance + STOP_CLEARANCE))
        try:
            return replace(self, speed_profile=placed_stops(frame, self.speed_profile, excluded,
                                                            space.is_junction_lane))
        except ValueError:
            return self

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation, format ``SPEED_PROFILE_VERSION``.

        :returns: A new JSON-compatible mapping of the object fields.
        :rtype: Dict
        """
        return {
            "version": SPEED_PROFILE_VERSION,
            "route": self.route.to_dict(),
            "lane_change_s": list(self.lane_change_s),
            "speed_profile": [{"lane_id": point.lane_id, "s": point.s, "speed": point.speed, "dwell": point.dwell}
                              for point in self.speed_profile],
            "start_t": self.start_t,
            "vehicle_type": self.vehicle_type.name,
            "drive_style": self.drive_style,
        }

    @classmethod
    def from_dict(cls, data: Dict, space: RouteSpace | None = None) -> "NPCVehicleAgent":
        """
        Build an agent from ``to_dict`` output.

        Version 1 dicts (no ``version``; speed points by ``fraction`` of the
        replay path) need the agent type's ``space``: each fraction is placed on
        the replay path the old format implied, and that map point is anchored
        to the nearest lane of the path section there, at its projected ``s``
        clamped to the lane's driven stretch. A point inside the minimum stretch
        of a lane change snaps to the nearer end of that stretch.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :param RouteSpace | None space: Required for version 1 dicts only.
        :returns: The reconstructed object.
        :rtype: NPCVehicleAgent
        :raises ValueError: For a version 1 dict without ``space``, or an unknown version.
        :raises KeyError: If a required serialized field is absent or ``vehicle_type``
            names an unknown vehicle type.
        """
        version = data.get("version", 1)
        route = LaneRoute.from_dict(data["route"])
        lane_change_s = tuple(data["lane_change_s"])
        vehicle_type = VehicleType[data["vehicle_type"]]
        if version == 1:
            if space is None:
                raise ValueError("a version 1 NPC dict (speed points by fraction) needs its RouteSpace to load")
            draws = [(point["fraction"], point["speed"], point.get("dwell", 0.0)) for point in data["speed_profile"]]
            return cls(route, lane_change_s, _from_fractions(space, route, lane_change_s, draws, vehicle_type),
                       data["start_t"], vehicle_type, data.get("drive_style", 0.5))
        if version != SPEED_PROFILE_VERSION:
            raise ValueError(f"unknown NPC dict version {version!r}")
        return cls(route, lane_change_s, tuple(SpeedPoint(**point) for point in data["speed_profile"]),
                   data["start_t"], vehicle_type, data.get("drive_style", 0.5))


def _from_fractions(space: RouteSpace, route: LaneRoute, lane_change_s: Tuple[float, ...],
                    draws: Sequence[Tuple[float, float, float]], vehicle_type: VehicleType,
                    ) -> Tuple[SpeedPoint, ...]:
    """
    Version 1 speed points (fraction of the replay path, speed, dwell) as map
    anchors: the replay path is rebuilt as version 1 did (lane-change lengths
    from the speed the fractions target), each fraction's map point is found
    on it and anchored to the nearest lane of its section (``from_dict``).

    :param RouteSpace space: Route space supplying lane geometry, topology,
        and vehicle-specific constraints.
    :param LaneRoute route: Route to inspect, validate, or convert.
    :param Tuple[float, ...] lane_change_s: Lane-change start positions on source
        lanes, in route order.
    :param Sequence[Tuple[float, float, float]] draws: Fraction-of-distance,
        speed, and dwell triples to anchor on the route.
    :param VehicleType vehicle_type: Vehicle category selecting dimensions and motion limits.
    :returns: Version-1 speed points converted to lane-anchored speed points.
    :rtype: Tuple[SpeedPoint, ...]
    """
    frame = RouteFrame(space, route, lane_change_s)
    spec = VEHICLE_SPECS[vehicle_type]
    fractions = [fraction for fraction, _, _ in draws]

    def target(fraction: float) -> float:
        """
        Evaluate the legacy speed profile at a fraction of the replay path.

        :param float fraction: Fraction of the total travelled distance.
        :returns: The desired moving speed at the requested replay-path fraction.
        :rtype: float
        """
        index = max(bisect_right(fractions, fraction) - 1, 0)
        return next((speed for _, speed, _ in draws[index:] if speed > 0), draws[-1][1])

    minimum = space.rules.min_lane_change_distance
    distances = []
    for index in route.lane_change_indices:
        start = frame.offsets[index] + frame.ends[index] - frame.starts[index]
        speed = min(target(start / frame.total if frame.total > 0 else 0.0), spec.max_speed)
        limit = speed_limit(space, (route.lanes[index], route.lanes[index + 1]))
        if limit is not None:
            speed = min(speed, limit)
        distances.append(min(max(LANE_CHANGE_TIME * speed, minimum), max(MAX_LANE_CHANGE_DISTANCE, minimum)))
    layout = replay_layout(space, route, lane_change_s, tuple(distances))
    placed = []
    for rank, (fraction, speed, dwell) in enumerate(draws):
        position = fraction * layout.path.length
        point = layout.path.interpolate(position)
        best = None
        for lane_id in layout.lanes_at(position):
            line = space.service.get_lane_central_curve_by_id(lane_id)
            low, high = frame.window(lane_id)
            s = min(max(line.project(point), low), high)
            gap = line.interpolate(s).distance(point)
            if best is None or gap < best[0]:
                best = (gap, lane_id, s)
        _, lane_id, s = best
        placed.append((frame.distance(lane_id, s), rank, SpeedPoint(lane_id, s, speed, dwell)))
    moving = next((speed for _, speed, _ in draws if speed > 0), draws[-1][1])
    return ordered_points(placed, SpeedPoint(route.lanes[0], route.start_s, moving))
