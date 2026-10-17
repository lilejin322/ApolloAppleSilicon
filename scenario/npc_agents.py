"""
NPC vehicle chromosomes and the trajectories they replay.

An ``NPCVehicleAgent`` drives a lane route that Apollo never sees: where it
changes lanes, how fast it goes, and its vehicle type are genes. The NPC does
not react to the ego. It waits at its start pose until ``start_t``, follows
its speed profile within the vehicle's acceleration limits, and stops at the
end of its route, where it stays. An NPC manager publishes ``state_at(t)``.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from enum import Enum
from math import ceil, isfinite, sqrt
from random import Random
from types import MappingProxyType
from typing import Mapping, NamedTuple, Sequence

from shapely.geometry import LineString

from .routing import InvalidRoute, LaneRoute, RouteRules, RouteSpace
from ..map.geometry import CenterlineProfile, normalize_angle
from ..utils import require_nonnegative

MAX_SPEED_POINTS = 4
_GRID_STEP = 0.5
"""Metres between speed-profile grid points along the path."""
_PATH_TOLERANCE = 0.05
"""Metres the replay path may deviate from the route's lane geometry. Dense
maps place vertices centimetres apart with millimetres of lateral noise; kept,
each would read as a turn tighter than any vehicle can drive."""


class VehicleType(Enum):
    CAR = "CAR"
    VAN = "VAN"
    TRUCK = "TRUCK"
    BUS = "BUS"


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
    """

    length: float
    width: float
    height: float
    sub_type: str
    min_turn_radius: float
    max_speed: float
    max_acceleration: float
    max_deceleration: float
    max_lateral_acceleration: float = 2.5


VEHICLE_SPECS = MappingProxyType({
    VehicleType.CAR: VehicleSpec(4.7, 1.9, 1.5, "ST_CAR", 5.5, 20.0, 3.0, 6.0),
    VehicleType.VAN: VehicleSpec(5.5, 2.0, 2.2, "ST_VAN", 6.5, 18.0, 2.5, 5.5),
    VehicleType.TRUCK: VehicleSpec(9.0, 2.5, 3.5, "ST_TRUCK", 10.0, 15.0, 1.5, 4.5),
    VehicleType.BUS: VehicleSpec(12.0, 2.55, 3.2, "ST_BUS", 10.5, 15.0, 1.5, 4.5),
})
"""Default dimensions and limits per vehicle type."""


@dataclass(frozen=True)
class SpeedPoint:
    """
    A target speed from a position along the route onwards.

    A zero speed is a stop: the vehicle halts exactly there, waits ``dwell``
    seconds, then heads for the next point's speed.

    :param float fraction: Position as a fraction of the driven path length, in [0, 1).
    :param float speed: Desired speed in m/s; replay keeps below lane limits and curve caps.
    :param float dwell: Seconds to wait at a stop; only for zero speeds.
    """

    fraction: float
    speed: float
    dwell: float = 0.0


class VehicleState(NamedTuple):
    """Pose and speed of an NPC at one time, in map coordinates and m/s."""

    x: float
    y: float
    heading: float
    speed: float


def _replay_path(space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...],
                 ) -> tuple[LineString, list[tuple[float, float, float]]]:
    """
    The route's path, each section simplified within ``_PATH_TOLERANCE`` to
    drop vertex noise, and its speed-limit spans: (start, end, limit) in path
    metres for every section whose lanes have a limit.
    """
    points: list[tuple[float, float]] = []
    spans = []
    s = 0.0
    for section in space.path_sections(route, lane_change_s=lane_change_s):
        line = section.line.simplify(_PATH_TOLERANCE, preserve_topology=False)
        coords = list(line.coords)
        points.extend(coords[1:] if points else coords)
        limit = _speed_limit(space, section.lanes)
        if limit is not None:
            spans.append((s, s + line.length, limit))
        s += line.length
    path = LineString(points)
    # Summed section lengths can differ from the joined length in the last bits:
    # clamp overshoot, and end a span on the last section exactly at the path end.
    spans = [(min(low, path.length), min(high, path.length), limit) for low, high, limit in spans]
    if spans and spans[-1][1] == min(s, path.length):
        spans[-1] = (spans[-1][0], path.length, spans[-1][2])
    return path, spans


def _speed_limit(space: RouteSpace, lanes: tuple[str, ...]) -> float | None:
    """The lowest positive ``speed_limit`` of the lanes, or None if none has one."""
    limits = [lane.speed_limit for lane in map(space.service.get_lane_by_id, lanes)
              if lane.HasField("speed_limit") and lane.speed_limit > 0]
    return min(limits) if limits else None


def _curvature_spans(path: LineString) -> list[tuple[float, float, float]]:
    """Absolute d(heading)/ds on every interval of the replay heading model."""
    profile = CenterlineProfile(path)
    spans = []
    for index in range(1, len(profile.midpoint_s)):
        low, high = profile.midpoint_s[index - 1:index + 1]
        if high <= low:
            raise InvalidRoute("path has a degenerate heading interval")
        turn = abs(normalize_angle(profile.headings[index] - profile.headings[index - 1]))
        if turn:
            spans.append((low, high, turn / (high - low)))
    return spans


class NPCTrajectory:
    """
    The time-parameterized motion of one NPC along its path.

    Between grid points the vehicle accelerates uniformly, so speeds and
    times are exact for the computed profile.
    """

    def __init__(self, path: LineString, start_t: float, s: list[float], v: list[float],
                 dwell: list[float]) -> None:
        self.path = path
        self.start_t = start_t
        self._profile = CenterlineProfile(path)
        self._s, self._v = s, v
        self._arrive, self._depart = [], []
        clock = 0.0
        for index in range(len(s)):
            if index:
                clock += 2 * (s[index] - s[index - 1]) / (v[index - 1] + v[index])
            self._arrive.append(clock)
            clock += dwell[index]
            self._depart.append(clock)

    @property
    def arrival_t(self) -> float:
        """
        Scenario time at which the vehicle stops at the end of its route.

        :rtype: float
        """
        return self.start_t + self._arrive[-1]

    def state_at(self, t: float) -> VehicleState:
        """
        Return the vehicle's pose and speed at scenario time ``t``.

        :param float t: Seconds since the scenario started.
        :rtype: VehicleState
        """
        s, v = self._s, self._v
        elapsed = t - self.start_t
        index = bisect_right(self._depart, elapsed) - 1
        if index < 0:
            distance, speed = 0.0, 0.0                       # Waiting at the start.
        elif index >= len(s) - 1:
            distance, speed = s[-1], 0.0                     # Stopped at the end.
        elif elapsed >= self._arrive[index + 1]:
            distance, speed = s[index + 1], 0.0              # Dwelling at a stop.
        else:
            step = s[index + 1] - s[index]
            acceleration = (v[index + 1] ** 2 - v[index] ** 2) / (2 * step)
            dt = elapsed - self._depart[index]
            distance = min(s[index] + v[index] * dt + acceleration * dt * dt / 2, s[index + 1])
            speed = max(v[index] + acceleration * dt, 0.0)
        point = self.path.interpolate(distance)
        return VehicleState(point.x, point.y, self._profile.heading_at(distance), speed)


@dataclass(frozen=True)
class NPCVehicleAgent:
    """
    Genetic representation of one NPC vehicle.

    :param LaneRoute route: City lanes to drive, with start and end positions.
    :param tuple[float, ...] lane_change_s: Where each lane change starts on its
        source lane, one per lateral transition, in route order.
    :param tuple[SpeedPoint, ...] speed_profile: One to four target speeds by
        path fraction, strictly increasing; the last speed must not be zero.
    :param float start_t: Seconds after scenario start to move off.
    :param VehicleType vehicle_type: Dimensions and limits from ``VEHICLE_SPECS``.
    """

    route: LaneRoute
    lane_change_s: tuple[float, ...]
    speed_profile: tuple[SpeedPoint, ...]
    start_t: float
    vehicle_type: VehicleType

    def __post_init__(self) -> None:
        if len(self.lane_change_s) != self.route.lane_changes:
            raise ValueError(f"lane_change_s needs one position per lane change "
                             f"({self.route.lane_changes}), got {len(self.lane_change_s)}")
        if not 1 <= len(self.speed_profile) <= MAX_SPEED_POINTS:
            raise ValueError(f"speed_profile needs 1 to {MAX_SPEED_POINTS} points")
        spec = self.spec
        previous = -1.0
        for index, point in enumerate(self.speed_profile):
            if not (isfinite(point.fraction) and 0 <= point.fraction < 1):
                raise ValueError("speed point fraction must be in [0, 1)")
            if point.fraction <= previous:
                raise ValueError("speed point fractions must be strictly increasing")
            previous = point.fraction
            if not (isfinite(point.speed) and 0 <= point.speed <= spec.max_speed):
                raise ValueError(f"speed must be within [0, max_speed={spec.max_speed}] "
                                 f"for a {self.vehicle_type.name}")
            require_nonnegative(dwell=point.dwell)
            if point.dwell > 0 and point.speed != 0:
                raise ValueError("dwell is only allowed at zero-speed points")
            if point.speed == 0 and index == len(self.speed_profile) - 1:
                raise ValueError("the last speed point cannot be a stop; the vehicle stops at the end")
        require_nonnegative(start_t=self.start_t)

    @property
    def spec(self) -> VehicleSpec:
        """
        The vehicle type's dimensions and limits.

        :rtype: VehicleSpec
        """
        return VEHICLE_SPECS[self.vehicle_type]

    @staticmethod
    def rules(vehicle_type: VehicleType) -> RouteRules:
        """
        Route rules for an NPC vehicle: city lanes, no U-turn tighter than the
        type can drive, no ambiguity checks (Apollo never matches NPC poses).

        :param VehicleType vehicle_type: The NPC's type.
        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"CITY_DRIVING"}),
                          min_turn_radius=VEHICLE_SPECS[vehicle_type].min_turn_radius,
                          unambiguous=False)

    def validate(self, space: RouteSpace) -> None:
        """
        Check the route, lane-change positions, and final path's turning radius.

        :param RouteSpace space: Built with ``NPCVehicleAgent.rules`` for this type.
        :raises ValueError: If the space allows U-turns too tight for this type.
        :raises InvalidRoute: With the first problem found.
        """
        self._validated_path(space)

    def _validated_path(self, space: RouteSpace) -> tuple[
            LineString, list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        """The replay path, its curvature spans and its speed-limit spans, once the path is known to be drivable."""
        if space.rules.min_turn_radius < self.spec.min_turn_radius:
            raise ValueError(f"the space allows U-turns down to {space.rules.min_turn_radius} m; a "
                             f"{self.vehicle_type.name} needs NPCVehicleAgent.rules({self.vehicle_type})")
        space.validate(self.route, lane_change_s=self.lane_change_s)
        path, limits = _replay_path(space, self.route, self.lane_change_s)
        spans = _curvature_spans(path)
        for low, high, curvature in spans:
            if curvature > 1 / self.spec.min_turn_radius + 1e-9:
                raise InvalidRoute(f"path turning radius {1 / curvature:.3f} m at "
                                   f"s=[{low:.3f}, {high:.3f}] is below "
                                   f"{self.vehicle_type.name}'s {self.spec.min_turn_radius} m")
        return path, spans, limits

    def trajectory(self, space: RouteSpace) -> NPCTrajectory:
        """
        Compute the motion this agent replays.

        :param RouteSpace space: As in ``validate``.
        :raises InvalidRoute: If the route or lane-change positions are invalid.
        :rtype: NPCTrajectory
        """
        path, curvature, limits = self._validated_path(space)
        s, v, dwell = self._speeds(path.length, curvature_spans=curvature, speed_limits=limits)
        return NPCTrajectory(path, self.start_t, s, v, dwell)

    def with_route(self, route: LaneRoute, space: RouteSpace, rng: Random, *,
                   keep: Mapping[tuple[str, str], float] | None = None,
                   attempts: int = 10) -> "NPCVehicleAgent":
        """
        Return this agent on another route, e.g. after crossover or mutation.

        Lane changes between the same lanes keep positions that leave a feasible
        continuation; other positions are sampled with the remaining route checked.

        :param keep: Positions to reuse by (source lane, target lane); this
            agent's own by default. A crossover child passes both parents'.
        :raises InvalidRoute: If no valid lane-change positions were found.
        :rtype: NPCVehicleAgent
        """
        keep = dict(zip(self.route.lane_change_pairs, self.lane_change_s) if keep is None else keep)
        error = InvalidRoute("no valid lane-change positions found")
        for _ in range(attempts):
            try:
                positions = space.lane_change_positions(route, rng, keep=keep)
                candidate = replace(self, route=route, lane_change_s=positions)
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
               max_start_t: float = 10.0, min_length: float = 50.0, max_length: float = 500.0,
               lane_change_probability: float = 0.2, stop_probability: float = 0.1,
               max_dwell: float = 5.0, min_speed: float = 2.0, max_attempts: int = 200,
               ) -> "NPCVehicleAgent":
        """
        Draw a random valid NPC of the given type.

        :param RouteSpace space: Built with ``NPCVehicleAgent.rules(vehicle_type)``.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param float stop_probability: Chance that an inner speed point is a stop.
        :param float max_dwell: Longest stop in seconds.
        :param float min_speed: Lowest nonzero target speed in m/s.
        :param float min_length: Minimum length of the final replay path in metres.
        :param float max_length: Maximum length of the final replay path in metres.
        :param int max_attempts: Candidate limit, including rejected path geometry.
        :raises RuntimeError: If no valid agent was found.
        :rtype: NPCVehicleAgent
        """
        spec = VEHICLE_SPECS[vehicle_type]
        last_problem = "no candidates drawn"
        for _ in range(max_attempts):
            route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                       lane_change_probability=lane_change_probability)
            try:
                positions = space.lane_change_positions(route, rng)
            except InvalidRoute as problem:
                last_problem = str(problem)
                continue
            agent = cls(route, positions,
                        _random_profile(rng, spec, stop_probability, max_dwell, min_speed),
                        rng.uniform(0, max_start_t), vehicle_type)
            try:
                # Explicit lane changes and joins change the geometric length.
                # Fit the final leg to the original length target when possible,
                # then validate the complete path. Never shift lane-change genes.
                path, _ = _replay_path(space, route, positions)
                if not min_length <= path.length <= max_length:
                    target = min(max(space.length(route), min_length), max_length)
                    route = replace(route, end_s=route.end_s + target - path.length)
                    agent = replace(agent, route=route)
                path, _, _ = agent._validated_path(space)
            except InvalidRoute as problem:
                last_problem = str(problem)
                continue
            if min_length <= path.length <= max_length:
                return agent
            last_problem = f"final path length {path.length} is outside [{min_length}, {max_length}]"
        raise RuntimeError(f"no valid NPC found in {max_attempts} attempts; last candidate: {last_problem}")

    def to_dict(self) -> dict:
        """
        Return a JSON-compatible representation.

        :rtype: dict
        """
        return {
            "route": self.route.to_dict(),
            "lane_change_s": list(self.lane_change_s),
            "speed_profile": [{"fraction": point.fraction, "speed": point.speed, "dwell": point.dwell}
                              for point in self.speed_profile],
            "start_t": self.start_t,
            "vehicle_type": self.vehicle_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NPCVehicleAgent":
        """
        Build an agent from ``to_dict`` output.

        :rtype: NPCVehicleAgent
        """
        return cls(
            LaneRoute.from_dict(data["route"]),
            tuple(data["lane_change_s"]),
            tuple(SpeedPoint(**point) for point in data["speed_profile"]),
            data["start_t"],
            VehicleType[data["vehicle_type"]],
        )

    def _speeds(self, length: float, *, curvature_spans: Sequence[tuple[float, float, float]] = (),
                speed_limits: Sequence[tuple[float, float, float]] = (),
                ) -> tuple[list[float], list[float], list[float]]:
        """Grid positions, feasible speeds, and dwell times along the path."""
        spec = self.spec
        positions = [point.fraction * length for point in self.speed_profile]
        stops = {position: point.dwell for position, point in zip(positions, self.speed_profile)
                 if point.speed == 0}
        count = max(1, ceil(length / _GRID_STEP))
        # The end is added as `length` itself: length * count / count can differ
        # in the last bit and would not be recognized as the end below.
        grid = sorted({length * index / count for index in range(count)} | {length}
                      | set(positions) | {s for low, high, _ in curvature_spans for s in (low, high)}
                      | {s for low, high, _ in speed_limits for s in (low, high)})
        # Two adjacent zero-speed constraints would leave no room to move.
        pinned = set(stops) | {0.0, length}
        for left, right in zip(list(grid), grid[1:]):
            if left in pinned and right in pinned:
                grid.append((left + right) / 2)
        grid.sort()

        def target(s: float) -> float:
            index = max(bisect_right(positions, s) - 1, 0)
            # A stop holds only at its own position; afterwards the next speed applies.
            for point in self.speed_profile[index:]:
                if point.speed > 0:
                    return point.speed
            return self.speed_profile[-1].speed

        speeds = [0.0 if s in stops or s in (0.0, length) else target(s) for s in grid]
        # Include every curvature boundary, even spans shorter than the normal
        # grid step. Constrain both ends of each interval so constant-acceleration
        # interpolation cannot exceed the lateral limit between samples.
        for low, high, curvature in curvature_spans:
            limit = sqrt(spec.max_lateral_acceleration / curvature)
            for index in range(bisect_left(grid, low), bisect_right(grid, high)):
                speeds[index] = min(speeds[index], limit)
        # Lane speed limits, with both ends of every span on the grid, as for curvature.
        for low, high, limit in speed_limits:
            for index in range(bisect_left(grid, low), bisect_right(grid, high)):
                speeds[index] = min(speeds[index], limit)
        for index in range(1, len(grid)):
            reachable = sqrt(speeds[index - 1] ** 2 + 2 * spec.max_acceleration * (grid[index] - grid[index - 1]))
            speeds[index] = min(speeds[index], reachable)
        for index in range(len(grid) - 2, -1, -1):
            stoppable = sqrt(speeds[index + 1] ** 2 + 2 * spec.max_deceleration * (grid[index + 1] - grid[index]))
            speeds[index] = min(speeds[index], stoppable)
        return grid, speeds, [stops.get(s, 0.0) for s in grid]


def _random_profile(rng: Random, spec: VehicleSpec, stop_probability: float, max_dwell: float,
                    min_speed: float) -> tuple[SpeedPoint, ...]:
    count = rng.randint(1, MAX_SPEED_POINTS)
    fractions = [0.0] + sorted(rng.uniform(0.05, 0.95) for _ in range(count - 1))
    points = []
    for index, fraction in enumerate(fractions):
        if 0 < index < count - 1 and rng.random() < stop_probability:
            points.append(SpeedPoint(fraction, 0.0, rng.uniform(0, max_dwell)))
        else:
            points.append(SpeedPoint(fraction, rng.uniform(min(min_speed, spec.max_speed), spec.max_speed)))
    return tuple(points)
