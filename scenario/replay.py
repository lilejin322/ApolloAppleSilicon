"""
Replayed motion shared by NPC vehicles and bicycles.

A replayed agent follows a path Apollo never sees. It waits at the start
until ``start_t``, follows its speed profile within its motion limits and
the lane speed limits, and stops at the end of its path, where it stays.
Speed points are map positions (lane, s) on the route. Pivots, the corners
of virtual lanes, are forced stops: the agent halts there, turns in place,
waits, and goes on.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from math import ceil, isfinite, pi, sqrt
from random import Random
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Protocol, Sequence

import numpy as np
from shapely.geometry import LineString, Point

from .routing import InvalidRoute, LaneRoute, RouteSpace, Transition
from .realism import JERK_TARGET, draw_speed, reference_speed
from ..map.geometry import CenterlineProfile, normalize_angle, smooth_polyline
from ..utils import require_nonnegative

MAX_SPEED_POINTS = 4
_GRID_STEP = 0.5
"""Metres between speed-profile grid points along the path."""
_SPEED_SMOOTHING = 4.0
"""Metres on either side over which speed caps and the speed profile are averaged."""
CAP_HOLD_TIME = 1.0
"""Seconds of travel at the top speed over which speed caps are smoothed (minimum then mean)."""
_FINE_STEP = 0.1
"""Metres between samples when averaging a speed profile."""
_POINT_GAP = 1e-6
"""Metres of driven distance below which two speed points count as one position."""
STOP_CLEARANCE = 2.0
"""Metres a placed stop keeps from its neighbours and from stretches it must avoid."""
PIVOT_YAW_RATE = pi / 4
"""Maximum radians per second when a stopped bicycle turns at a pivot."""
_PATH_TOLERANCE = 0.05
"""Metres the replay path may deviate from the route's lane geometry. Dense
maps place vertices centimetres apart with millimetres of lateral noise; kept,
each would read as a turn tighter than any vehicle can drive."""

Span = tuple[float, float, float]
"""(start, end, value) in path metres."""


class MotionLimits(Protocol):
    """Limits a replayed agent's motion stays within, in metres and seconds."""

    max_speed: float
    max_acceleration: float
    max_deceleration: float
    max_lateral_acceleration: float
    comfortable_deceleration: float
    min_turn_radius: float


@dataclass(frozen=True)
class SpeedPoint:
    """
    A target speed from a map position on the route onwards.

    A zero speed is a stop: the agent halts exactly there, waits ``dwell``
    seconds, then heads for the next point's speed. The position is a lane of
    the agent's route and an ``s`` within the stretch of that lane the route
    drives, so a stop stays at the same map location when other parts of the
    route change.

    :param str lane_id: A lane of the agent's route.
    :param float s: Metres along that lane's centerline.
    :param float speed: Desired speed in m/s; replay keeps below lane limits and curve caps.
    :param float dwell: Seconds to wait at a stop; only for zero speeds.
    """

    lane_id: str
    s: float
    speed: float
    dwell: float = 0.0


class AgentState(NamedTuple):
    """Pose and speed of a replayed agent at one time, in map coordinates and m/s."""

    x: float
    y: float
    heading: float
    speed: float


def check_profile(route: LaneRoute, profile: Sequence[SpeedPoint], max_speed: float, name: str) -> None:
    """
    Check a speed profile's shape and speeds against its route.

    :param LaneRoute route: The agent's route.
    :param profile: One to ``MAX_SPEED_POINTS`` points on lanes of the route,
        strictly increasing in route order (lane, then s); the last one must not be a stop.
    :param float max_speed: Highest allowed speed in m/s.
    :param str name: The agent type, for error messages.
    :raises ValueError: With the first problem found.
    """
    if not 1 <= len(profile) <= MAX_SPEED_POINTS:
        raise ValueError(f"speed_profile needs 1 to {MAX_SPEED_POINTS} points")
    lanes = {lane_id: index for index, lane_id in enumerate(route.lanes)}
    previous = None
    for index, point in enumerate(profile):
        if point.lane_id not in lanes:
            raise ValueError(f"speed point lane {point.lane_id!r} is not on the route")
        if not (isfinite(point.s) and point.s >= 0):
            raise ValueError("speed point s must be finite and nonnegative")
        key = (lanes[point.lane_id], point.s)
        if previous is not None and key <= previous:
            raise ValueError("speed points must be strictly increasing along the route")
        previous = key
        if not (isfinite(point.speed) and 0 <= point.speed <= max_speed):
            raise ValueError(f"speed must be within [0, max_speed={max_speed}] for a {name}")
        require_nonnegative(dwell=point.dwell)
        if point.dwell > 0 and point.speed != 0:
            raise ValueError("dwell is only allowed at zero-speed points")
        if point.speed == 0 and index == len(profile) - 1:
            raise ValueError("the last speed point cannot be a stop; the agent stops at the end")


class RouteFrame:
    """
    Driven distance along a route's legs, each lane change counting as the
    rules' minimum: where each lane's driven stretch lies, and conversions
    between (lane, s) and that distance. Independent of the speed profile.
    """

    def __init__(self, space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...]) -> None:
        legs = space.legs(route, lane_change_s=lane_change_s)
        minimum = space.rules.min_lane_change_distance
        self.route = route
        self.index = {leg.lane_id: index for index, leg in enumerate(legs)}
        self.starts = [leg.start_s for leg in legs]
        self.ends = [max(leg.end_s, leg.start_s) for leg in legs]
        self.offsets, self.gaps = [], []
        distance = 0.0
        for index, leg in enumerate(legs):
            self.offsets.append(distance)
            distance += self.ends[index] - self.starts[index]
            gap = minimum if index < len(legs) - 1 and route.transitions[index] is not Transition.FORWARD else 0.0
            self.gaps.append(gap)
            distance += gap
        self.total = distance

    def window(self, lane_id: str) -> tuple[float, float]:
        """The driven stretch [start, end] of a route lane."""
        index = self.index[lane_id]
        return self.starts[index], self.ends[index]

    def distance(self, lane_id: str, s: float) -> float:
        """Driven distance to ``s`` on a route lane, clamped to its stretch."""
        index = self.index[lane_id]
        return self.offsets[index] + min(max(s, self.starts[index]), self.ends[index]) - self.starts[index]

    def at(self, distance: float) -> tuple[str, float]:
        """The (lane, s) at a driven distance; inside a lane change's minimum
        stretch, the nearer of its source end and target start."""
        distance = min(max(distance, 0.0), self.total)
        for index, lane_id in enumerate(self.route.lanes):
            low = self.offsets[index]
            high = low + self.ends[index] - self.starts[index]
            last = index == len(self.route.lanes) - 1
            if distance <= high or last:
                return lane_id, self.starts[index] + min(max(distance - low, 0.0), high - low)
            if distance < high + self.gaps[index]:
                if distance - high <= high + self.gaps[index] - distance:
                    return lane_id, self.ends[index]
                return self.route.lanes[index + 1], self.starts[index + 1]
        raise AssertionError("unreachable")

    def check_windows(self, profile: Sequence[SpeedPoint]) -> None:
        """
        Check that every speed point lies on its lane's driven stretch.

        :raises InvalidRoute: For the first point outside.
        """
        for point in profile:
            low, high = self.window(point.lane_id)
            if not low - 1e-6 <= point.s <= high + 1e-6:
                raise InvalidRoute(f"speed point s={point.s:.3f} is outside the driven stretch "
                                   f"[{low:.3f}, {high:.3f}] of lane {point.lane_id!r}")


def ordered_points(placed: Sequence[tuple[float, int, SpeedPoint]], fallback: SpeedPoint,
                   ) -> tuple[SpeedPoint, ...]:
    """
    Speed points sorted by driven distance: of points at the same distance the
    one with the lowest rank stays, trailing stops are dropped (the agent
    stops at the route end anyway), and ``fallback`` stands in for an empty result.
    """
    points: list[tuple[float, SpeedPoint]] = []
    for distance, _, point in sorted(placed, key=lambda item: (item[0], item[1])):
        if points and distance <= points[-1][0] + _POINT_GAP:
            continue
        points.append((distance, point))
    while points and points[-1][1].speed == 0:
        points.pop()
    return tuple(point for _, point in points[:MAX_SPEED_POINTS]) or (fallback,)


def rebased(space: RouteSpace, old_route: LaneRoute, old_lane_change_s: tuple[float, ...],
            profile: Sequence[SpeedPoint], route: LaneRoute, lane_change_s: tuple[float, ...],
            ) -> tuple[SpeedPoint, ...]:
    """
    Speed points of one route moved onto another route of the same space. A
    point whose lane is on the new route, within the lane's driven stretch
    there, keeps its lane and ``s``. Any other point moves to the same share of
    the driven distance it had on the old route (the start of the new route if
    the old route is invalid). Points are then put in driving order
    (``ordered_points``). Clamping points outside the stretch onto its ends
    merged them when a route was truncated, and point counts fell over
    generations (-0.05 points per extend/truncate mutation).

    :raises InvalidRoute: If ``route`` or ``lane_change_s`` is invalid.
    """
    new = RouteFrame(space, route, lane_change_s)
    try:
        old = RouteFrame(space, old_route, old_lane_change_s)
    except InvalidRoute:
        old = None
    placed = []
    for point in profile:
        low, high = new.window(point.lane_id) if point.lane_id in new.index else (0.0, -1.0)
        if low <= point.s <= high:
            placed.append((new.distance(point.lane_id, point.s), 0, point))
        else:
            share = old.distance(point.lane_id, point.s) / old.total if old is not None and old.total > 0 else 0.0
            lane_id, s = new.at(share * new.total)
            placed.append((new.distance(lane_id, s), 1, replace(point, lane_id=lane_id, s=s)))
    moving = next((point.speed for point in profile if point.speed > 0), profile[-1].speed)
    return ordered_points(placed, SpeedPoint(route.lanes[0], route.start_s, moving))


def refit_end(space: RouteSpace, route: LaneRoute, length: float, target: float) -> LaneRoute:
    """
    The route with its end moved so that a replay path of ``length`` metres
    would become ``target`` metres long; where the space keeps endpoints away
    from junctions and the moved end is not allowed, the nearest allowed end
    on the last lane instead, if there is one.
    """
    end_s = route.end_s + target - length
    if space.restricts_endpoints and not space.endpoint_allowed(route.lanes[-1], end_s):
        nearest = space.nearest_endpoint(route.lanes[-1], end_s)
        end_s = end_s if nearest is None else nearest
    return replace(route, end_s=end_s)


def placed_stops(frame: RouteFrame, profile: Sequence[SpeedPoint], avoid: Sequence[tuple[float, float]],
                 blocked: Callable[[str], bool]) -> tuple[SpeedPoint, ...]:
    """
    The profile with each stop that lies on a ``blocked`` lane or within an
    ``avoid`` stretch (driven distances) moved to the nearest driven position
    off blocked lanes and outside those stretches, between its neighbours with
    ``STOP_CLEARANCE``. Stops without such a position stay where they are.
    """
    allowed = []
    for index, lane_id in enumerate(frame.route.lanes):
        if blocked(lane_id):
            continue
        low = frame.offsets[index]
        pieces = [(low, low + frame.ends[index] - frame.starts[index])]
        for a, b in avoid:
            pieces = [piece for x, y in pieces
                      for piece in ((x, min(y, a)), (max(x, b), y)) if piece[1] > piece[0]]
        allowed.extend(pieces)
    marks = [frame.distance(point.lane_id, point.s) for point in profile]
    profile = list(profile)
    for index, point in enumerate(profile):
        if point.speed > 0:
            continue
        here = marks[index]
        before = marks[index - 1] + STOP_CLEARANCE if index else 0.0
        after = marks[index + 1] - STOP_CLEARANCE if index + 1 < len(marks) else frame.total
        if any(a <= here <= b for a, b in allowed):
            continue
        options = [min(max(here, max(a, before)), min(b, after)) for a, b in allowed
                   if min(b, after) >= max(a, before)]
        if not options:
            continue
        target = min(options, key=lambda value: abs(value - here))
        lane_id, s = frame.at(target)
        profile[index] = replace(point, lane_id=lane_id, s=s)
        marks[index] = frame.distance(lane_id, s)
    return tuple(profile)


class ReplayLayout(NamedTuple):
    """A replay path, its speed-limit spans, its sections: (lanes, simplified
    line, start in joined metres), with ``moved`` mapping joined metres onto
    the smoothed path, and its pivots in path metres."""

    path: LineString
    limits: list[Span]
    pieces: list[tuple[tuple[str, ...], LineString, float]]
    moved: Callable[[float], float]
    pivots: tuple[float, ...] = ()

    def position(self, space: RouteSpace, lane_id: str, s: float) -> float | None:
        """Path metres of the map point at ``s`` on a lane: its projection onto
        the nearest section that lies on the lane; None if none does."""
        point = space.centerline(lane_id).interpolate(s)
        best = None
        for lanes, line, offset in self.pieces:
            if lane_id not in lanes:
                continue
            gap = line.distance(point)
            if best is None or gap < best[0] - 1e-9:
                best = (gap, offset + line.project(point))
        return None if best is None else min(max(self.moved(best[1]), 0.0), self.path.length)

    def lanes_at(self, position: float) -> tuple[str, ...]:
        """The lanes of the section at path metres ``position``."""
        for lanes, line, offset in self.pieces:
            if position <= self.moved(offset + line.length) + 1e-9:
                return lanes
        return self.pieces[-1][0]

    def positions(self, space: RouteSpace, profile: Sequence[SpeedPoint]) -> list[float]:
        """
        Path metres of each speed point.

        :raises InvalidRoute: If a point is off the path or the points are not in driving order.
        """
        positions = [self.position(space, point.lane_id, point.s) for point in profile]
        if any(b is None or a is None or b <= a for a, b in zip(positions, positions[1:])) or None in positions:
            raise InvalidRoute("speed points do not lie in driving order along the replay path")
        return positions


def replay_layout(space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...],
                  lane_change_distances: tuple[float, ...] | None = None) -> ReplayLayout:
    """
    The route's path, each section simplified within ``_PATH_TOLERANCE`` to
    drop vertex noise, then smoothed (``smooth_polyline``), with its
    speed-limit spans: (start, end, limit) in path metres for every section
    whose lanes have a limit. Lane changes have the lengths
    ``lane_change_distances`` as far as the lanes allow
    (``RouteSpace.path_sections``); the rules' minimum by default.

    Virtual lanes have no speed limit. Their pivots split the path into
    stretches smoothed on their own, so each corner stays a vertex of the path.

    :raises InvalidRoute: If the route is illegal.
    """
    points: list[tuple[float, float]] = []
    spans, pieces, corners = [], [], []
    s = 0.0
    for section in space.path_sections(route, lane_change_s=lane_change_s,
                                       lane_change_distances=lane_change_distances):
        line = section.line.simplify(_PATH_TOLERANCE, preserve_topology=False)
        coords = list(line.coords)
        offset = len(points) - 1 if points else 0
        for lane_id in section.lanes:
            lane = space.virtual_lane(lane_id)
            for pivot in lane.pivots if lane is not None else ():
                corner = lane.line.interpolate(pivot)
                index = min(range(len(coords)), key=lambda i: corner.distance(Point(coords[i])))
                if corner.distance(Point(coords[index])) > 1e-6:
                    raise InvalidRoute(f"the corner of {lane_id!r} at s={pivot} was simplified away")
                corners.append(offset + index)
        points.extend(coords[1:] if points else coords)
        limit = speed_limit(space, section.lanes)
        if limit is not None:
            spans.append((s, s + line.length, limit))
        pieces.append((section.lanes, line, s))
        s += line.length
    # Position and heading both come from one smoothed line, so the reported
    # heading follows the direction of motion even at coarse map vertices.
    bounds = [0, *corners, len(points) - 1]
    vertices, resampled, smoothed = [], [], []
    for low, high in zip(bounds, bounds[1:]):
        piece, resampled_at = smooth_polyline(points[low:high + 1])
        vertices.extend(piece if not vertices else piece[1:])
        resampled.append(resampled_at)
        smoothed.append(np.r_[0.0, np.cumsum(np.hypot(*np.diff(piece, axis=0).T))])
    joined_starts = np.r_[0.0, np.cumsum([at[-1] for at in resampled])]
    smoothed_starts = np.r_[0.0, np.cumsum([along[-1] for along in smoothed])]
    path = LineString(vertices)
    joined = joined_starts[-1]
    total = s

    def moved(value: float) -> float:
        at = value * joined / total if total > 0 else 0.0
        piece = min(max(int(np.searchsorted(joined_starts, at, side="right")) - 1, 0), len(resampled) - 1)
        return float(smoothed_starts[piece] + np.interp(at - joined_starts[piece], resampled[piece], smoothed[piece]))

    spans = [(moved(low), moved(high), limit) for low, high, limit in spans]
    # Summed section lengths can differ from the joined length in the last bits:
    # clamp overshoot, and end a span on the last section exactly at the path end.
    last_end = spans[-1][1] if spans else None
    spans = [(min(low, path.length), min(high, path.length), limit) for low, high, limit in spans]
    if spans and last_end is not None and last_end >= smoothed_starts[-1] - 1e-6:
        spans[-1] = (spans[-1][0], path.length, spans[-1][2])
    cumulative = CenterlineProfile(path).cumulative_s if corners else []
    pivots = tuple(cumulative[index] for index in np.cumsum([len(along) - 1 for along in smoothed])[:-1])
    return ReplayLayout(path, spans, pieces, moved, pivots)


class ReplayPath(NamedTuple):
    """The path an agent replays, the speed limits along it, and its pivots."""

    path: LineString
    speed_limits: list[Span]
    pivots: tuple[float, ...] = ()


def replay_path(space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...],
                lane_change_distances: tuple[float, ...] | None = None) -> ReplayPath:
    """
    The route's replay path, its speed-limit spans and its pivots, as in ``replay_layout``.

    :raises InvalidRoute: If the route is illegal.
    :rtype: ReplayPath
    """
    layout = replay_layout(space, route, lane_change_s, lane_change_distances)
    return ReplayPath(layout.path, layout.limits, layout.pivots)


def speed_limit(space: RouteSpace, lanes: tuple[str, ...]) -> float | None:
    """
    The lowest positive ``speed_limit`` of the lanes, or None if none has one.
    Virtual lanes have no limit.

    :rtype: float | None
    """
    limits = [lane.speed_limit for lane in (space.service.get_lane_by_id(lane_id) for lane_id in lanes
                                            if space.virtual_lane(lane_id) is None)
              if lane.HasField("speed_limit") and lane.speed_limit > 0]
    return min(limits) if limits else None


def _pieces(path: LineString, pivots: Sequence[float]) -> list[tuple[float, LineString]]:
    """The path split at the vertices nearest its pivots: (start s, piece) in order."""
    if not pivots:
        return [(0.0, path)]
    coords = list(path.coords)
    cumulative = CenterlineProfile(path).cumulative_s
    cuts = []
    for pivot in pivots:
        index = bisect_left(cumulative, pivot)
        cuts.append(index if index < len(cumulative) and (
            index == 0 or cumulative[index] - pivot <= pivot - cumulative[index - 1]) else index - 1)
    bounds = [0, *cuts, len(coords) - 1]
    return [(cumulative[low], LineString(coords[low:high + 1])) for low, high in zip(bounds, bounds[1:])]


def curvature_spans(path: LineString, pivots: Sequence[float] = ()) -> list[Span]:
    """
    Absolute d(heading)/ds on every interval of the replay heading model.

    The path is split at its pivots and each piece measured on its own, so
    the corner at a pivot, turned at rest, is never measured.

    :raises InvalidRoute: If the path has a degenerate heading interval.
    :rtype: list[tuple[float, float, float]]
    """
    spans = []
    for start, piece in _pieces(path, pivots):
        profile = CenterlineProfile(piece)
        for index in range(1, len(profile.midpoint_s)):
            low, high = profile.midpoint_s[index - 1:index + 1]
            if high <= low:
                raise InvalidRoute("path has a degenerate heading interval")
            turn = abs(normalize_angle(profile.headings[index] - profile.headings[index - 1]))
            if turn:
                spans.append((start + low, start + high, turn / (high - low)))
    return spans


def check_turning_radius(spans: Sequence[Span], min_turn_radius: float, name: str) -> None:
    """
    Check that no curvature span is tighter than ``min_turn_radius``.

    :param str name: The agent type, for error messages.
    :raises InvalidRoute: For the first span that is too tight.
    """
    for low, high, curvature in spans:
        if curvature > 1 / min_turn_radius + 1e-9:
            raise InvalidRoute(f"path turning radius {1 / curvature:.3f} m at "
                               f"s=[{low:.3f}, {high:.3f}] is below {name}'s {min_turn_radius} m")


class ReplayTrajectory:
    """
    The time-parameterized motion of one replayed agent along its path.

    Between grid points the agent accelerates uniformly, so speeds and
    times are exact for the computed profile. Headings follow the path piece
    by piece between pivots; at a pivot the agent turns at a bounded angular
    speed, then waits for its dwell before departing.
    """

    def __init__(self, path: LineString, start_t: float, s: list[float], v: list[float],
                 dwell: list[float], pivots: Sequence[float] = ()) -> None:
        self.path = path
        self.start_t = start_t
        pieces = _pieces(path, pivots)
        self._starts = [start for start, _ in pieces]
        self._profiles = [CenterlineProfile(piece) for _, piece in pieces]
        self._s, self._v = s, v
        self._arrive, self._depart = [], []
        self._turns = {}
        for piece in range(1, len(pieces)):
            incoming = self._profiles[piece - 1].heading_at(pieces[piece - 1][1].length)
            outgoing = self._profiles[piece].heading_at(0)
            angle = normalize_angle(outgoing - incoming)
            # Smoothstep's peak slope is 1.5, so this bounds angular velocity.
            self._turns[self._starts[piece]] = (incoming, angle, 1.5 * abs(angle) / PIVOT_YAW_RATE)
        clock = 0.0
        for index in range(len(s)):
            if index:
                clock += 2 * (s[index] - s[index - 1]) / (v[index - 1] + v[index])
            self._arrive.append(clock)
            clock += dwell[index] + self._turns.get(s[index], (0, 0, 0))[2]
            self._depart.append(clock)

    @property
    def arrival_t(self) -> float:
        """
        Scenario time at which the agent stops at the end of its path.

        :rtype: float
        """
        return self.start_t + self._arrive[-1]

    def moving_time(self, duration: float, *, min_speed: float = 0.5) -> float:
        """Seconds moving at least ``min_speed`` within scenario time [0, duration].

        Integrates the linear speed of each travel interval exactly. Initial
        waiting, dwell, pivot rotation and terminal parking never count.
        """
        require_nonnegative(duration=duration, min_speed=min_speed)
        total = 0.0
        for index in range(len(self._s) - 1):
            start = self.start_t + self._depart[index]
            end = self.start_t + self._arrive[index + 1]
            low, high = max(0.0, start), min(duration, end)
            if high <= low:
                continue
            before, after = self._v[index:index + 2]
            if max(before, after) <= 0:
                continue
            if before == after:
                if before >= min_speed:
                    total += high - low
                continue
            crossing = start + (min_speed - before) * (end - start) / (after - before)
            if after > before:
                low = max(low, crossing)
            else:
                high = min(high, crossing)
            total += max(0.0, high - low)
        return total


    def state_at(self, t: float) -> AgentState:
        """
        Return the agent's pose and speed at scenario time ``t``.

        :param float t: Seconds since the scenario started.
        :rtype: AgentState
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
        piece = bisect_right(self._starts, distance) - 1
        heading = self._profiles[piece].heading_at(distance - self._starts[piece])
        turn = self._turns.get(distance)
        if turn is not None:
            incoming, angle, duration = turn
            stop_index = bisect_left(s, distance)
            u = min(max((elapsed - self._arrive[stop_index]) / duration, 0.0), 1.0) if duration else 1.0
            heading = normalize_angle(incoming + angle * u * u * (3 - 2 * u))
        return AgentState(point.x, point.y, heading, speed)


def check_activity_window(duration: float, min_moving_fraction: float) -> None:
    """Validate a finite scenario horizon and the required fraction spent moving."""
    if not isfinite(duration) or duration <= 0:
        raise ValueError("scenario_duration must be finite and positive")
    if not isfinite(min_moving_fraction) or not 0 <= min_moving_fraction <= 1:
        raise ValueError("min_moving_fraction must be within [0, 1]")


def speed_grid(length: float, profile: Sequence[SpeedPoint], positions: Sequence[float], limits: MotionLimits, *,
               curvature_spans: Sequence[Span] = (), speed_limits: Sequence[Span] = (),
               forced_stops: Mapping[float, float] = MappingProxyType({}),
               ) -> tuple[list[float], list[float], list[float]]:
    """
    Return grid positions, feasible speeds, and dwell times along a path.

    Speeds follow the profile's targets, capped by the lateral acceleration
    on curves and by lane speed limits, smoothed, and made reachable within
    the acceleration limit and the comfortable deceleration. The agent is at
    rest at the start, the end, every stop of the profile, and every forced stop.

    :param float length: Path length in metres.
    :param profile: The agent's speed points.
    :param positions: Path metres of each speed point, increasing.
    :param MotionLimits limits: The agent's limits.
    :param curvature_spans: As returned by ``curvature_spans``.
    :param speed_limits: As in ``ReplayLayout.limits``.
    :param forced_stops: Seconds to wait by path metres, such as at pivots; a
        profile stop at the same position adds its dwell.
    :rtype: tuple[list[float], list[float], list[float]]
    """
    positions = [float(position) for position in positions]
    stops = {position: point.dwell for position, point in zip(positions, profile) if point.speed == 0}
    for position, dwell in forced_stops.items():
        stops[position] = stops.get(position, 0.0) + dwell
    count = max(1, ceil(length / _GRID_STEP))
    # The end is added as `length` itself: length * count / count can differ
    # in the last bit and would not be recognized as the end below.
    grid = sorted({length * index / count for index in range(count)} | {length} | set(stops)
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
        for point in profile[index:]:
            if point.speed > 0:
                return point.speed
        return profile[-1].speed

    targets = [target(s) for s in grid]
    # Curvature caps include every curvature boundary, even spans shorter
    # than the normal grid step, and constrain both ends of each interval,
    # so constant-acceleration interpolation (v^2 linear in s) cannot
    # exceed the lateral limit between samples.
    for low, high, curvature in curvature_spans:
        limit = sqrt(limits.max_lateral_acceleration / curvature)
        for index in range(bisect_left(grid, low), bisect_right(grid, high)):
            targets[index] = min(targets[index], limit)
    # Lane speed limits, with both ends of every span on the grid, as for curvature.
    for low, high, limit in speed_limits:
        for index in range(bisect_left(grid, low), bisect_right(grid, high)):
            targets[index] = min(targets[index], limit)
    zero = [s in stops or s in (0.0, length) for s in grid]
    braking = min(limits.comfortable_deceleration, limits.max_deceleration)
    caps = np.square(np.asarray(targets))
    # A smooth lower envelope of the caps: speeds no longer chatter
    # between many nearby curvature limits, and every value stays at or
    # below the raw cap at its own position.
    # The window covers about ``CAP_HOLD_TIME`` of travel at the top speed:
    # through a long curve whose centerline curvature wobbles, the speed holds
    # at the lowest nearby cap instead of oscillating (jerk spikes).
    peak = float(np.sqrt(np.max(caps))) if len(caps) else 0.0
    smooth_caps = _envelope(np.asarray(grid), caps, max(_SPEED_SMOOTHING, CAP_HOLD_TIME * peak))
    squared = _passes(grid, np.where(zero, 0.0, smooth_caps), limits.max_acceleration, braking)
    # Round the corners of the accelerate / cruise / brake profile so the
    # acceleration ramps over a few metres instead of jumping.
    # The rounding reach keeps jerk near ``JERK_TARGET`` at the profile's top
    # speed: an acceleration change spread over 2 * reach metres at speed v
    # takes 2 * reach / v seconds.
    top = float(np.sqrt(np.max(smooth_caps))) if len(smooth_caps) else 0.0
    reach = max(_SPEED_SMOOTHING, limits.max_acceleration * top / (2 * JERK_TARGET))
    rounded = _rounded(np.asarray(grid), squared, zero, reach)
    rounded = _passes(grid, np.where(zero, 0.0, np.minimum(rounded, smooth_caps)),
                      limits.max_acceleration, braking)
    speeds = [0.0 if pinned else sqrt(max(value, 0.0)) for pinned, value in zip(zero, rounded)]
    return grid, speeds, [stops.get(s, 0.0) for s in grid]


def _passes(grid: Sequence[float], squared: np.ndarray, acceleration: float,
            deceleration: float) -> np.ndarray:
    """Largest squared speeds at or below ``squared`` reachable under both limits."""
    result = [float(value) for value in squared]
    for index in range(1, len(grid)):
        result[index] = min(result[index], result[index - 1] + 2 * acceleration * (grid[index] - grid[index - 1]))
    for index in range(len(grid) - 2, -1, -1):
        result[index] = min(result[index], result[index + 1] + 2 * deceleration * (grid[index + 1] - grid[index]))
    return np.asarray(result)


def _window_mean(s: np.ndarray, values: np.ndarray, reach: float, low: float, high: float,
                 extend) -> np.ndarray:
    """Mean of the piecewise-linear ``values`` over [s - reach, s + reach] for
    every ``s``; ``extend(points)`` gives the function outside [low, high]."""
    step = min(_FINE_STEP, reach / 4)
    fine = np.arange(low - reach - step, high + reach + 2 * step, step)
    sampled = extend(fine)
    integral = np.r_[0.0, np.cumsum((sampled[1:] + sampled[:-1]) / 2 * step)]
    upper = np.interp(s + reach, fine, integral)
    lower = np.interp(s - reach, fine, integral)
    return (upper - lower) / (2 * reach)


def _envelope(s: np.ndarray, caps: np.ndarray, reach: float) -> np.ndarray:
    """A smooth function at or below ``caps`` at every grid point: the moving
    minimum over a slightly wider window, then the moving mean."""
    if len(s) < 3:
        return caps
    gap = float(np.max(np.diff(s)))
    wide = reach + gap
    lows = np.searchsorted(s, s - wide, side="left")
    highs = np.searchsorted(s, s + wide, side="right")
    eroded = np.array([caps[a:b].min() for a, b in zip(lows, highs)])
    mean = _window_mean(s, eroded, reach, s[0], s[-1], lambda points: np.interp(points, s, eroded))
    return np.minimum(mean, caps)


def _rounded(s: np.ndarray, squared: np.ndarray, zero: Sequence[bool], reach: float) -> np.ndarray:
    """
    Moving mean of squared speeds within each stretch between zero-speed
    points, mirrored oddly at them so a stretch keeps its zero ends and the
    constant acceleration it leaves and reaches them with. A stretch whose
    mean is not positive keeps its values.
    """
    result = squared.copy()
    pins = [index for index, pinned in enumerate(zero) if pinned]
    for first, last in zip(pins, pins[1:]):
        if last - first < 2:
            continue
        piece_s, piece = s[first:last + 1], squared[first:last + 1]
        low, high = piece_s[0], piece_s[-1]

        def extend(points, piece_s=piece_s, piece=piece, low=low, high=high):
            inside = np.interp(points, piece_s, piece)
            before = -np.interp(2 * low - points, piece_s, piece)
            after = -np.interp(2 * high - points, piece_s, piece)
            return np.where(points < low, before, np.where(points > high, after, inside))

        mean = _window_mean(piece_s, piece, min(reach, (high - low) / 2), low, high, extend)
        inner = mean[1:-1]
        if np.all(inner > 0):
            result[first + 1:last] = inner
    return result


def point_reference(space: RouteSpace, lane_id: str, kind: str, max_speed: float) -> float:
    """What a target speed on ``lane_id`` is relative to (``realism.reference_speed``)."""
    return reference_speed(kind, speed_limit(space, (lane_id,)), max_speed)


def drawn_speeds(space: RouteSpace, profile: Sequence[SpeedPoint], rng: Random, kind: str, max_speed: float,
                 min_speed: float) -> tuple[SpeedPoint, ...]:
    """The profile with every moving target redrawn from the cruise prior of its lane (``realism``)."""
    low = min(min_speed, max_speed)
    return tuple(point if point.speed == 0 else replace(
        point, speed=draw_speed(rng, kind, point_reference(space, point.lane_id, kind, max_speed), low, max_speed))
        for point in profile)


def random_draws(rng: Random, max_speed: float, stop_probability: float, max_dwell: float,
                 min_speed: float) -> list[tuple[float, float, float]]:
    """Random (share of driven distance, speed, dwell) draws for one to ``MAX_SPEED_POINTS`` points."""
    count = rng.randint(1, MAX_SPEED_POINTS)
    fractions = [0.0] + sorted(rng.uniform(0.05, 0.95) for _ in range(count - 1))
    points = []
    for index, fraction in enumerate(fractions):
        if 0 < index < count - 1 and rng.random() < stop_probability:
            points.append((fraction, 0.0, rng.uniform(0, max_dwell)))
        else:
            points.append((fraction, rng.uniform(min(min_speed, max_speed), max_speed), 0.0))
    return points


def anchored(frame: RouteFrame, draws: Sequence[tuple[float, float, float]]) -> tuple[SpeedPoint, ...]:
    """Speed points at shares of a route's driven distance, anchored to the lane there."""
    placed = []
    for rank, (fraction, speed, dwell) in enumerate(draws):
        lane_id, s = frame.at(fraction * frame.total)
        placed.append((frame.distance(lane_id, s), rank, SpeedPoint(lane_id, s, speed, dwell)))
    moving = next((speed for _, speed, _ in draws if speed > 0), draws[-1][1])
    return ordered_points(placed, SpeedPoint(frame.route.lanes[0], frame.route.start_s, moving))
