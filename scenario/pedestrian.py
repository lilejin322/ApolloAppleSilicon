"""
Pedestrian chromosomes and the trajectories they replay.

A ``PedestrianAgent`` walks a ``WalkRoute`` of the map's walk network
(``walkways``): along sidewalks, and off them on up to two excursions, a
crossing in a crosswalk, beside one, mid-block or diagonally through a
junction, or a stretch walked in the road along the curb. Its walk, its
speed profile, its start time, its walker type and its walk style are genes.
Like NPC vehicles and bicycles it ignores signals and the ego: it waits at its
start pose until ``start_t``, follows its speed profile within the walker's
limits, stops and turns in place at each pivot, and stays at the end of its
walk. Speed-profile stops make it wait at or just off the curb, on a median,
or mid-road. A pedestrian manager will publish ``state_at(t)``.
"""

import numpy as np
from math import pi
from enum import Enum
from random import Random
from shapely.geometry import LineString
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Sequence, Dict, List, Tuple, Self

from ..map.geometry import normalize_angle, smooth_polyline
from ..utils import require_nonnegative
from .realism import EXCURSION_COUNT_MIX, EXCURSION_MIX, WALKER_MIX, comfort_limits, draw_speed, draw_type
from .replay import (
    MAX_SPEED_POINTS, ReplayTrajectory, SpeedPoint, Span, check_profile, check_turning_radius, curvature_spans,
    ordered_points, speed_grid,
)
from .routing import InvalidRoute
from .walkways import MAX_EXCURSIONS, ExcursionKind, WalkFrame, WalkNetwork, WalkRoute

WALK_SMOOTHING = 0.8
"""Metres: standard deviation of the Gaussian that rounds a walk's corners. A
walker turns from the sidewalk into a crossing within a step or two; the
vehicle default (2 m) would cut corners across the curb. At 0.5 m a 90° corner
had a radius of about 0.5 m and the lateral limit slowed adults from 1.4 to
1.0 m/s round it; at 0.8 m (radius about 1.3 m) walkers keep their pace and
runners still slow down."""
WALK_STEP = 0.25
"""Largest metres between the vertices of a smoothed walk."""
WALK_SPEED_SMOOTHING = 0.75
"""Metres over which a walker's speed changes are rounded (``replay.speed_grid``
``smoothing``). People reach their walking pace within two or three steps
(about 1-2 s); the vehicle default of 4 m spread a walk-off from rest over 4-6 s."""
WALKER_YAW_RATE = pi
"""Radians per second a walker turns at when it turns in place (a pivot):
about one and a half seconds to turn round."""
GUTTER_REACH = 1.0
"""Metres past the curb within which a stop makes the walker turn to face the
road at the entry first: it waits at the curb, or just off it, facing across."""
_TURN_AT_ENTRY = pi / 9
"""Smallest turn at an entry that a waiting walker makes in place."""

CURB_WAIT_PROBABILITY = 0.5
"""Chance that a random walker waits before an excursion."""
CURB_STAND_BACK = 0.3
"""Metres before the curb where a walker waits on the sidewalk."""
GUTTER_SHARE = 0.3
"""Share of curb waits made standing in the road, ``GUTTER_RANGE`` past the curb."""
GUTTER_RANGE = (0.2, 0.8)
MAX_CURB_WAIT = 20.0
"""Longest wait at the curb in seconds."""
ISLAND_WAIT_PROBABILITY = 0.3
"""Chance that a random walker crossing a median or island waits on it (a two-stage crossing)."""
ISLAND_WAIT = (1.0, 15.0)
FASTER_PROBABILITY = 0.3
"""Chance that a random walker speeds up while crossing, by ``FASTER_FACTOR``:
crossing speeds exceed sidewalk walking speeds."""
FASTER_FACTOR = (1.0, 1.3)
HESITATION_PROBABILITY = 0.05
"""Chance that a random walker stops briefly mid-road (``HESITATION``)."""
HESITATION = (0.5, 3.0)
RESUME_GAP = 0.5
"""Metres after a stop where the walker's next target speed applies."""
STOP_SPACING = 1.0
"""Metres between two drawn stops."""


class WalkerType(Enum):
    """
    Walker categories selecting body dimensions, motion limits, and speed priors.
    """
    ADULT = "ADULT"
    ELDERLY = "ELDERLY"
    CHILD = "CHILD"
    RUNNER = "RUNNER"


def random_walker_type(rng: Random, weights: Mapping[str, float] = WALKER_MIX) -> WalkerType:
    """
    A walker type drawn from a composition (type name -> weight); by default
    ``realism.WALKER_MIX``.

    :param Random rng: Random number generator used for sampling and selection.
    :param Mapping[str, float] weights: Relative nonnegative sampling weights keyed by type name.
    :returns: The sampled walker category.
    :rtype: WalkerType
    """
    return WalkerType[draw_type(rng, weights)]


@dataclass(frozen=True)
class WalkerSpec:
    """
    Fixed properties of a walker type; Apollo's obstacle type is PEDESTRIAN.

    :param float length: Body depth in metres, along the heading.
    :param float width: Shoulder breadth in metres.
    :param float height: Metres.
    :param str sub_type: Apollo ``PerceptionObstacle.SubType`` name.
    :param float min_turn_radius: Metres, on the path apart from pivots.
    :param float max_speed: m/s.
    :param float max_acceleration: m/s^2.
    :param float max_deceleration: m/s^2, positive.
    :param float max_lateral_acceleration: m/s^2; walkers slow down round tight corners.
    :param float comfortable_deceleration: m/s^2, positive.
    """

    length: float
    """Body depth in metres, along the heading."""
    width: float
    """Shoulder breadth in metres."""
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
    """m/s^2; walkers slow down round tight corners."""
    comfortable_deceleration: float
    """m/s^2, positive."""


WALKER_SPECS = MappingProxyType({
    WalkerType.ADULT: WalkerSpec(0.35, 0.55, 1.70, "ST_PEDESTRIAN", 0.25, 3.5, 1.5, 2.0, 2.0, 1.0),
    WalkerType.ELDERLY: WalkerSpec(0.35, 0.55, 1.65, "ST_PEDESTRIAN", 0.25, 2.0, 1.0, 1.5, 1.2, 0.8),
    WalkerType.CHILD: WalkerSpec(0.25, 0.40, 1.25, "ST_PEDESTRIAN", 0.25, 4.0, 2.0, 2.5, 2.5, 1.2),
    WalkerType.RUNNER: WalkerSpec(0.40, 0.55, 1.75, "ST_PEDESTRIAN", 0.25, 5.0, 2.5, 3.0, 3.0, 1.5),
})
"""Dimensions and replay limits per walker type. Bodies follow FHWA's standing body
ellipse of 0.5 x 0.6 m (FHWA-RD-98-107) and 95th-percentile breadth and depth of
0.58 and 0.33 m; the limits are replay-model limits, not calibrated gait dynamics.
Every type has the same turning radius, so a type change never fails on geometry
and the population's type mix does not drift."""


class WalkLayout(NamedTuple):
    """
    A walk's smoothed path, its pivots in path metres, and ``moved`` mapping
    walked (raw) metres onto the path.

    :param LineString path: Replay path in map metres.
    :param Tuple[float, ...] pivots: Path distances where the agent stops and turns in place, in metres.
    :param Callable[[float], float] moved: Maps raw walked distance to distance
        along the smoothed replay path.
    """

    path: LineString
    """Replay path in map metres."""
    pivots: Tuple[float, ...]
    """Path distances where the agent stops and turns in place, in metres."""
    moved: Callable[[float], float]
    """Map raw walked metres to metres along the smoothed replay path."""


def walk_layout(network: WalkNetwork, frame: WalkFrame, profile: Sequence[SpeedPoint]) -> WalkLayout:
    """
    The path a walker replays: its walk's raw elements joined, split at the
    pivots and each piece smoothed (``WALK_SMOOTHING``).

    Pivots are every turning point and, where the walker turns by more than
    ``_TURN_AT_ENTRY`` onto an excursion, the spot where it waits at that
    entry (within ``GUTTER_REACH`` past the curb): it rounds the corner, stops
    there, turns in place to face the road, then waits. A vertex is inserted
    at each pivot.

    :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
    :param WalkFrame frame: Walk frame used to map element positions to travelled distance.
    :param Sequence[SpeedPoint] profile: Ordered target-speed points; positions are
        walk-element metres.
    :returns: The smoothed replay-path layout, including pivots and the mapping
        from raw walked distance to smoothed path distance.
    :rtype: WalkLayout
    :raises InvalidRoute: If the walk has no length.
    """
    points: List[Tuple[float, float]] = []
    raw: List[float] = []
    for element in frame.elements:
        for xy in element.coords:
            step = float(np.hypot(xy[0] - points[-1][0], xy[1] - points[-1][1])) if points else 0.0
            if points and step < 1e-9:
                continue
            raw.append(raw[-1] + step if raw else 0.0)
            points.append(xy)
    if len(points) < 2:
        raise InvalidRoute("the walk has no length")
    raw_array = np.asarray(raw)
    turns = [frame.offsets[index] for index, element in enumerate(frame.elements) if element.name.endswith("b")]
    waits = []
    for index, element in enumerate(frame.elements):
        if element.name.startswith("x") and not element.name.endswith("b"):
            wait = _entry_wait(network, frame, index, profile)
            if wait is not None and _turn(points, raw_array, frame.offsets[index]) > _TURN_AT_ENTRY:
                waits.append(wait)
    wanted = sorted({distance for distance in turns + waits if 1e-6 < distance < raw[-1] - 1e-6})
    points, raw, pivot_indices = _with_vertices(points, raw, wanted)
    bounds = [0, *pivot_indices, len(points) - 1]
    vertices, pieces = [], []
    for low, high in zip(bounds, bounds[1:]):
        piece, resampled_at = smooth_polyline(points[low:high + 1], sigma=WALK_SMOOTHING, step=WALK_STEP)
        along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(piece, axis=0).T))]
        pieces.append((raw[low], raw[low] + resampled_at, along))
        vertices.extend(piece.tolist() if not vertices else piece[1:].tolist())
    path = LineString(vertices)
    starts = np.r_[0.0, np.cumsum([along[-1] for _, _, along in pieces])]
    raw_starts = [start for start, _, _ in pieces]

    def moved(value: float) -> float:
        """
        Map a raw walked distance onto the simplified replay path.

        :param float value: Distance along the raw walk, in metres.
        :returns: Corresponding distance along the smoothed replay path, in metres.
        :rtype: float
        """
        piece = min(max(int(np.searchsorted(raw_starts, value, side="right")) - 1, 0), len(pieces) - 1)
        _, raw_at, along = pieces[piece]
        return float(min(starts[piece] + np.interp(value, raw_at, along), path.length))

    pivots = tuple(float(value) for value in starts[1:-1])
    return WalkLayout(path, pivots, moved)


def _turn(points: Sequence[Tuple[float, float]], raw: np.ndarray, distance: float) -> float:
    """
    The heading change of a polyline at walked distance ``distance``, between a metre before and after.

    :param Sequence[Tuple[float, float]] points: Ordered polyline vertices in map metres.
    :param np.ndarray raw: Cumulative distances along the unsmoothed input polyline, in metres.
    :param float distance: Travelled distance, in metres.
    :returns: The heading change of a polyline at walked distance ``distance``, between a metre before and after.
    :rtype: float
    """
    xs, ys = np.asarray([xy[0] for xy in points]), np.asarray([xy[1] for xy in points])

    def at(value: float) -> Tuple[float, float]:
        """
        Interpolate a point at a distance along the raw polyline.

        :param float value: Distance along the raw polyline, in metres.
        :returns: The interpolated map coordinate at that distance.
        :rtype: Tuple[float, float]
        """
        return float(np.interp(value, raw, xs)), float(np.interp(value, raw, ys))

    before, centre, after = at(distance - 1.0), at(distance), at(distance + 1.0)
    incoming = np.arctan2(centre[1] - before[1], centre[0] - before[0])
    outgoing = np.arctan2(after[1] - centre[1], after[0] - centre[0])
    return abs(normalize_angle(float(outgoing - incoming)))


def _with_vertices(points: List, raw: List, distances: Sequence[float]) -> Tuple[List, List, List[int]]:
    """
    The polyline with a vertex at each walked distance, and those vertices' indices.

    :param List points: Ordered polyline vertices in map metres.
    :param List raw: Cumulative distances along the unsmoothed input polyline, in metres.
    :param Sequence[float] distances: Positions along the raw polyline where vertices are required, in metres.
    :returns: The polyline with a vertex at each walked distance, and those vertices' indices.
    :rtype: Tuple[List, List, List[int]]
    """
    points, raw, indices = list(points), list(raw), []
    for distance in distances:
        index = int(np.searchsorted(raw, distance))
        if index < len(raw) and abs(raw[index] - distance) < 1e-6:
            indices.append(index)
            continue
        if abs(raw[index - 1] - distance) < 1e-6:
            indices.append(index - 1)
            continue
        share = (distance - raw[index - 1]) / (raw[index] - raw[index - 1])
        (x0, y0), (x1, y1) = points[index - 1], points[index]
        points.insert(index, (x0 + share * (x1 - x0), y0 + share * (y1 - y0)))
        raw.insert(index, distance)
        indices.append(index)
    return points, raw, indices


def _entry_wait(network: WalkNetwork, frame: WalkFrame, index: int, profile: Sequence[SpeedPoint]) -> float | None:
    """
    Walked distance of the walker's first stop at the excursion ``index``'s
    entry: at the end of the sidewalk before it, or on it within
    ``GUTTER_REACH`` past the curb. None if it does not wait there.

    :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
    :param WalkFrame frame: Walk frame used to map element positions to travelled distance.
    :param int index: Index of the excursion element being processed.
    :param Sequence[SpeedPoint] profile: Ordered target-speed points; positions are
        walk-element metres.
    :returns: Walked distance of the first qualifying entry wait, or ``None`` if
        the walker does not wait there.
    :rtype: float | None
    """
    element = frame.elements[index]
    previous = frame.elements[index - 1] if index else None
    curb = network.curb_distance(LineString(element.coords)) if len(element.coords) > 1 else 0.0
    stops = []
    for point in profile:
        if point.speed != 0:
            continue
        if point.lane_id == element.name and point.s <= curb + GUTTER_REACH:
            stops.append(frame.distance(element.name, point.s))
        elif previous is not None and point.lane_id == previous.name and point.s >= previous.length - 1e-6:
            stops.append(frame.offsets[index])
    return min(stops) if stops else None


@dataclass(frozen=True)
class PedestrianAgent:
    """
    Genetic representation of one pedestrian.

    :param WalkRoute route: Its walk on the walk network.
    :param Tuple[SpeedPoint, ...] speed_profile: One to four target speeds at
        (element, s) positions on the walk, in walking order: ``lane_id`` names
        an element of ``route.lanes`` and ``s`` is metres from its start, which
        is a map position (the start, an entry, an exit or a turning point). The
        last speed must not be zero; the first also applies before it.
    :param float start_t: Seconds after scenario start to move off.
    :param WalkerType walker_type: Dimensions and limits from ``WALKER_SPECS``.
    :param float walk_style: In [0, 1], from gentle to brisk: the comfortable
        acceleration and braking the replay uses (``realism.COMFORT``).
    """

    route: WalkRoute
    """Its walk on the walk network."""
    speed_profile: Tuple[SpeedPoint, ...]
    """One to four target speeds at (element, s) positions on the walk, in walking order: ``lane_id`` names an element of ``route.lanes`` and ``s`` is metres from its start, which is a map position (the start, an entry, an exit or a turning point). The last speed must not be zero; the first also applies before it."""
    start_t: float
    """Seconds after scenario start to move off."""
    walker_type: WalkerType
    """Dimensions and limits from ``WALKER_SPECS``."""
    walk_style: float = 0.5
    """In [0, 1], from gentle to brisk: the comfortable acceleration and braking the replay uses (``realism.COMFORT``)."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        :raises ValueError: If input fields or bounds violate the validation conditions of this operation.
        """
        check_profile(self.route, self.speed_profile, self.spec.max_speed, self.walker_type.name)
        require_nonnegative(start_t=self.start_t)
        if not 0 <= self.walk_style <= 1:
            raise ValueError(f"walk_style must be within [0, 1], got {self.walk_style}")

    @property
    def spec(self) -> WalkerSpec:
        """
        The walker type's dimensions and limits.

        :returns: The immutable dimensions and hard motion limits of the agent type.
        :rtype: WalkerSpec
        """
        return WALKER_SPECS[self.walker_type]

    @property
    def limits(self) -> WalkerSpec:
        """
        The spec with this walker's comfortable acceleration and braking (``walk_style``).

        :returns: The type specification with style-dependent comfortable acceleration and braking.
        :rtype: WalkerSpec
        """
        return comfort_limits(self.spec, self.walker_type.name, self.walk_style)

    @property
    def lane_change_s(self) -> Tuple[float, ...]:
        """
        Always empty: lets code shared with lane routes measure walks.

        :returns: An empty tuple; this agent does not perform lane changes.
        :rtype: Tuple[float, ...]
        """
        return ()

    def validate(self, network: WalkNetwork) -> None:
        """
        Check the walk, the speed points and the path's turning radius.

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :returns: ``None`` if all applicable validation checks pass.
        :rtype: None
        :raises InvalidRoute: With the first problem found.
        """
        self._validated_layout(network)

    def _validated_layout(self, network: WalkNetwork) -> Tuple[WalkLayout, List[Span], List[float]]:
        """
        The layout, its curvature spans, and the path metres of each speed point.

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :returns: The layout, its curvature spans, and the path metres of each speed point.
        :rtype: Tuple[WalkLayout, List[Span], List[float]]
        :raises InvalidRoute: If route geometry, topology, point placement, or map constraints fail the checks performed here.
        """
        network.validate(self.route)
        frame = network.frame(self.route)
        frame.check_windows(self.speed_profile)
        layout = walk_layout(network, frame, self.speed_profile)
        spans = curvature_spans(layout.path, layout.pivots)
        check_turning_radius(spans, self.spec.min_turn_radius, self.walker_type.name)
        positions = [layout.moved(frame.distance(point.lane_id, point.s)) for point in self.speed_profile]
        # A wait at a pivot lies exactly on it, so the replay turns there before waiting.
        positions = [next((pivot for pivot in layout.pivots if abs(pivot - position) < 1e-6), position)
                     for position in positions]
        if any(b <= a + 1e-9 for a, b in zip(positions, positions[1:])):
            raise InvalidRoute("speed points do not lie in walking order along the path")
        return layout, spans, positions

    def route_distance(self, network: WalkNetwork, lane_id: str, s: float) -> float:
        """
        Walked distance from the start to ``s`` on an element, clamped to it.

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :param str lane_id: Walk-element ID on the route.
        :param float s: Position along that element, in metres.
        :returns: Travelled distance from the walk start, in metres, clamped to
            the element window.
        :rtype: float
        :raises InvalidRoute: If the walk is invalid.
        :raises KeyError: If the element is not on the walk.
        """
        return network.frame(self.route).distance(lane_id, s)

    def route_point(self, network: WalkNetwork, distance: float) -> Tuple[str, float]:
        """
        The (element, s) at a walked distance (``route_distance``).

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :param float distance: Travelled distance, in metres.
        :returns: The walk-element ID and local position at the requested walked distance.
        :rtype: Tuple[str, float]
        :raises InvalidRoute: If the walk is invalid.
        """
        return network.frame(self.route).at(distance)

    def trajectory(self, network: WalkNetwork) -> ReplayTrajectory:
        """
        Compute the motion this agent replays.

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :returns: The trajectory describing the agent motion over scenario time.
        :rtype: ReplayTrajectory
        :raises InvalidRoute: If the agent is invalid on this network.
        """
        layout, spans, positions = self._validated_layout(network)
        s, v, dwell = speed_grid(layout.path.length, self.speed_profile, positions, self.limits,
                                 curvature_spans=spans, forced_stops=dict.fromkeys(layout.pivots, 0.0),
                                 smoothing=WALK_SPEED_SMOOTHING)
        return ReplayTrajectory(layout.path, self.start_t, s, v, dwell, layout.pivots, yaw_rate=WALKER_YAW_RATE)

    def with_route(self, route: WalkRoute, network: WalkNetwork) -> Self:
        """
        Return this agent on another walk, e.g. after crossover or mutation.

        A speed point whose element the new walk still has keeps it, its ``s``
        within the element; others move to the same share of the walked
        distance, as for lane routes (``replay.rebased``).

        :param WalkRoute route: Walk to inspect, validate, or convert.
        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :returns: An agent on the proposed walk with its speed profile adjusted.
        :rtype: PedestrianAgent
        :raises InvalidRoute: If the agent is invalid on the new walk.
        """
        candidate = replace(self, route=route, speed_profile=rebased_walk(network, self.route, self.speed_profile,
                                                                          route))
        candidate.validate(network)
        return candidate

    @classmethod
    def random(cls, network: WalkNetwork, rng: Random, *, walker_type: WalkerType, max_start_t: float = 10.0,
               min_length: float = 10.0, max_length: float = 200.0,
               count_weights: Mapping[int, float] = EXCURSION_COUNT_MIX,
               kind_weights: Mapping[str, float] = EXCURSION_MIX, max_link: float = 40.0,
               max_approach: float = 30.0, turn_back_probability: float = 0.05, min_speed: float = 0.5,
               max_attempts: int = 100, min_arrival_t: float = 0.0) -> Self:
        """
        Draw a random valid pedestrian of the given walker type.

        How many excursions and of which kinds are drawn once per call from
        ``count_weights`` and ``kind_weights`` (kinds the map has no place for
        left out): redrawing per attempt would favour whatever fits more
        often. If the drawn walk never fits in ``max_attempts``, the last
        excursion is dropped and the rest tried again. Speeds: a cruise speed
        from the walker's prior (``realism.WALKER_CRUISE``), and per excursion
        a curb wait, a wait on a median, a faster crossing and a hesitation,
        each with its chance (module constants), at most four speed points.

        :param WalkNetwork network: The map's walk network.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param WalkerType walker_type: Walker category selecting motion limits and speed priors.
        :param float max_start_t: Maximum sampled start time, in scenario seconds.
        :param float min_length: Shortest replay path in metres.
        :param float max_length: Longest replay path in metres.
        :param Mapping[int, float] count_weights: Relative sampling weights keyed by excursion count.
        :param Mapping[str, float] kind_weights: Relative sampling weights keyed by excursion kind name.
        :param float max_link: Maximum sidewalk distance between connected excursions, in metres.
        :param float max_approach: Maximum sidewalk distance from an endpoint to the adjacent excursion, in metres.
        :param float turn_back_probability: Probability that the final excursion contains a turning point.
        :param float min_speed: Minimum moving target speed, in m/s.
        :param int max_attempts: Maximum generation attempts for each fixed
            excursion-kind sequence before dropping the last excursion and retrying.
        :param float min_arrival_t: Earliest allowed end-of-walk time in scenario
            seconds; zero disables it.
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: PedestrianAgent
        :raises ValueError: If input fields or bounds violate the validation
            conditions of this operation.
        :raises RuntimeError: If the map has no sidewalks or no valid agent was found.
        """
        require_nonnegative(min_arrival_t=min_arrival_t)
        if not network.sidewalks:
            raise RuntimeError("the map has no sidewalks")
        available = {kind.name for kind in network.kinds()}
        kinds_offered = {name: weight for name, weight in kind_weights.items() if name in available and weight > 0}
        counts = {str(count): weight for count, weight in count_weights.items()
                  if 0 <= int(count) <= MAX_EXCURSIONS and weight > 0}
        count = int(draw_type(rng, counts)) if counts else 0
        kinds = [ExcursionKind[draw_type(rng, kinds_offered)] for _ in range(count)] if kinds_offered else []
        last_problem = "no candidates drawn"
        for size in range(len(kinds), -1, -1):
            try:
                return cls._random_with(network, rng, walker_type, kinds[:size], max_start_t, min_length, max_length,
                                        max_link, max_approach, turn_back_probability, min_speed, max_attempts,
                                        min_arrival_t)
            except RuntimeError as problem:
                last_problem = str(problem)
        raise RuntimeError(last_problem)

    @classmethod
    def _random_with(cls, network: WalkNetwork, rng: Random, walker_type: WalkerType, kinds: List[ExcursionKind],
                     max_start_t: float, min_length: float, max_length: float, max_link: float, max_approach: float,
                     turn_back_probability: float, min_speed: float, max_attempts: int,
                     min_arrival_t: float) -> Self:
        """
        ``random`` with the excursion kinds fixed.

        :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
        :param Random rng: Random number generator used for sampling and selection.
        :param WalkerType walker_type: Walker category selecting motion limits and speed priors.
        :param List[ExcursionKind] kinds: Excursion kinds to include, in walk order.
        :param float max_start_t: Maximum sampled start time, in scenario seconds.
        :param float min_length: Minimum accepted replay-path length, in metres.
        :param float max_length: Maximum accepted replay-path length, in metres.
        :param float max_link: Maximum sidewalk distance between connected excursions, in metres.
        :param float max_approach: Maximum sidewalk distance from an endpoint to the adjacent excursion, in metres.
        :param float turn_back_probability: Probability that the final excursion contains a turning point.
        :param float min_speed: Minimum moving target speed, in m/s.
        :param int max_attempts: Maximum number of generation attempts before failure.
        :param float min_arrival_t: Earliest accepted route-end time in scenario seconds; zero disables the horizon check.
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: PedestrianAgent
        :raises RuntimeError: If required map data is unavailable or no valid candidate is found within the attempt limit.
        """
        spec = WALKER_SPECS[walker_type]
        last_problem = "no candidates drawn"
        for _ in range(max_attempts):
            route = network.random_route(rng, kinds, max_link=max_link, max_approach=max_approach,
                                         turn_back_probability=turn_back_probability)
            if route is None:
                last_problem = f"no walk with excursions {[kind.name for kind in kinds]} fitted"
                continue
            profile = random_walk_profile(network, route, rng, walker_type.name, spec.max_speed, min_speed)
            agent = cls(route, profile, rng.uniform(0, max_start_t), walker_type, rng.random())
            try:
                trajectory = agent.trajectory(network)
            except InvalidRoute as problem:
                last_problem = str(problem)
                continue
            if not min_length <= trajectory.path.length <= max_length:
                last_problem = f"path length {trajectory.path.length:.1f} is outside [{min_length}, {max_length}]"
                continue
            if min_arrival_t and (agent.start_t >= min_arrival_t or trajectory.arrival_t < min_arrival_t):
                last_problem = f"walk must start before and finish at or after {min_arrival_t} s"
                continue
            return agent
        raise RuntimeError(f"no valid pedestrian found in {max_attempts} attempts; last candidate: {last_problem}")

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A new JSON-compatible mapping of the object fields.
        :rtype: Dict
        """
        return {
            "route": self.route.to_dict(),
            "speed_profile": [{"lane_id": point.lane_id, "s": point.s, "speed": point.speed, "dwell": point.dwell}
                              for point in self.speed_profile],
            "start_t": self.start_t,
            "walker_type": self.walker_type.name,
            "walk_style": self.walk_style,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Self:
        """
        Build an agent from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed object.
        :rtype: PedestrianAgent
        :raises KeyError: If a required serialized field is absent or ``walker_type``
            names an unknown walker type.
        """
        return cls(WalkRoute.from_dict(data["route"]), tuple(SpeedPoint(**point) for point in data["speed_profile"]),
                   data["start_t"], WalkerType[data["walker_type"]], data.get("walk_style", 0.5))


def rebased_walk(network: WalkNetwork, old_route: WalkRoute, profile: Sequence[SpeedPoint],
                 route: WalkRoute) -> Tuple[SpeedPoint, ...]:
    """
    Speed points of one walk moved onto another: a point whose element is on
    the new walk keeps it and its ``s`` if the element is long enough; any
    other moves to the same share of the walked distance it had (the start
    of the new walk if the old walk is invalid). Points are then put in
    walking order (``replay.ordered_points``).

    :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
    :param WalkRoute old_route: Original walk on which the input speed points are defined.
    :param Sequence[SpeedPoint] profile: Ordered target-speed points; positions are
        walk-element metres.
    :param WalkRoute route: Walk to inspect, validate, or convert.
    :returns: The input profile mapped onto the proposed walk.
    :rtype: Tuple[SpeedPoint, ...]
    :raises InvalidRoute: If ``route`` is invalid.
    """
    new = network.frame(route)
    try:
        old = network.frame(old_route)
    except InvalidRoute:
        old = None
    placed = []
    for point in profile:
        low, high = new.window(point.lane_id) if point.lane_id in new.index else (0.0, -1.0)
        if low <= point.s <= high:
            placed.append((new.distance(point.lane_id, point.s), 0, point))
        else:
            share = old.distance(point.lane_id, point.s) / old.total \
                if old is not None and old.total > 0 and point.lane_id in old.index else 0.0
            name, s = new.at(share * new.total)
            placed.append((new.distance(name, s), 1, replace(point, lane_id=name, s=s)))
    moving = next((point.speed for point in profile if point.speed > 0), profile[-1].speed)
    return ordered_points(placed, SpeedPoint(route.lanes[0], 0.0, moving))


def random_walk_profile(network: WalkNetwork, route: WalkRoute, rng: Random, kind: str, max_speed: float,
                        min_speed: float) -> Tuple[SpeedPoint, ...]:
    """
    A random speed profile for a walk: a cruise speed from the walker's prior
    from the start and, per excursion in priority order, a curb wait (some of
    them just off the curb), a wait on a median or island it crosses, a faster
    crossing, and a hesitation mid-road, each with its chance; every stop is
    followed by the crossing speed ``RESUME_GAP`` later. At most
    ``MAX_SPEED_POINTS`` points; groups that do not fit are left out.

    :param WalkNetwork network: Walk network supplying sidewalk and excursion geometry.
    :param WalkRoute route: Walk to inspect, validate, or convert.
    :param Random rng: Random number generator used for sampling and selection.
    :param str kind: Walker type name selecting the speed prior.
    :param float max_speed: Maximum allowed target speed, in m/s.
    :param float min_speed: Minimum moving target speed, in m/s.
    :returns: A random speed profile with a cruise speed and optional excursion
        waits, faster crossings, and hesitations, limited to ``MAX_SPEED_POINTS``.
    :rtype: Tuple[SpeedPoint, ...]
    :raises InvalidRoute: If the walk is invalid.
    """
    frame = network.frame(route)
    low = min(min_speed, max_speed)
    cruise = draw_speed(rng, kind, 1.0, low, max_speed)
    base = SpeedPoint(route.lanes[0], 0.0, cruise)
    groups: List[Tuple[int, int, List[SpeedPoint]]] = []
    for index in range(len(route.excursions)):
        name = f"x{index}"
        element = frame.elements[frame.index[name]]
        line = LineString(element.coords)
        length = element.length
        curb = min(network.curb_distance(line), length)
        crossing = min(cruise * rng.uniform(*FASTER_FACTOR), max_speed) \
            if rng.random() < FASTER_PROBABILITY else cruise

        def stop(s: float, dwell: float) -> List[SpeedPoint]:
            """
            Create a stop and its moving resume point on the current excursion.

            :param float s: Position along the current excursion, in metres.
            :param float dwell: Stop dwell duration, in seconds.
            :returns: The zero-speed stop point followed by its moving resume point.
            :rtype: List[SpeedPoint]
            """
            return [SpeedPoint(name, s, 0.0, dwell), SpeedPoint(name, s + RESUME_GAP, crossing)]

        room = length - RESUME_GAP - 0.1
        if rng.random() < CURB_WAIT_PROBABILITY:
            s = curb + rng.uniform(*GUTTER_RANGE) if rng.random() < GUTTER_SHARE else max(curb - CURB_STAND_BACK, 0.0)
            if 0 < s <= room:
                groups.append((1, index, stop(s, rng.uniform(0, MAX_CURB_WAIT))))
        elif crossing != cruise and 0 < curb <= room:
            groups.append((3, index, [SpeedPoint(name, curb, crossing)]))
        islands = network.island_spans(line)
        if islands and rng.random() < ISLAND_WAIT_PROBABILITY:
            start, end = rng.choice(islands)
            s = (start + end) / 2
            if 0 < s <= room:
                groups.append((2, index, stop(s, rng.uniform(*ISLAND_WAIT))))
        if rng.random() < HESITATION_PROBABILITY and room > curb + 2:
            s = rng.uniform(curb + 1, room - 1) if room - 1 > curb + 1 else curb + 1
            groups.append((4, index, stop(s, rng.uniform(*HESITATION))))
    chosen, stops = [base], []
    for _, _, points in sorted(groups, key=lambda group: (group[0], group[1])):
        if len(chosen) + len(points) > MAX_SPEED_POINTS:
            continue
        marks = [frame.distance(point.lane_id, point.s) for point in points if point.speed == 0]
        if any(abs(mark - other) < STOP_SPACING + RESUME_GAP for mark in marks for other in stops):
            continue
        chosen.extend(points)
        stops.extend(marks)
    placed = [(frame.distance(point.lane_id, point.s), rank, point) for rank, point in enumerate(chosen)]
    return ordered_points(placed, base)
