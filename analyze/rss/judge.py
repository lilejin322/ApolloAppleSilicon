"""
Judge a recorded drive by RSS: at every obstacle frame the ego and each
nearby road user are checked from both sides; each side's proper responses
are obligations, owed after its response time from when they arose, and the
record shows whether each party applied them. At a touch, the party that
failed its obligation is to blame; when the ego is not, it still answers for a
collision a competent and careful driver would have avoided (``preventable``).
An ego that failed only to brake is not to blame where braking as owed would
still have left it touched, the other at fault: it was run into, say at its side.
The margin ranks the ego's failures by how close they came.

The library computes only the ego's response; the other's comes from the same
checkers with the roles swapped (digest §4.6, §5.4).
"""

from dataclasses import dataclass, field, replace
from math import atan2, cos, pi, sin
from typing import Dict, List, Tuple
from shapely.affinity import translate
from shapely.geometry import Polygon

from ...map import MapService
from ...utils import ApolloVehicleConfig
from ..footprint import box, ego_footprint
from ..oracles.base import Violation
from ..replay import World
from .constellations import Party, RssMap, relative_constellation, unstructured_constellation
from .formulas import lon_stated_braking
from .physics import SPEED_MAX, stopping_distance
from .preventable import (DECISION, EVALUATION, HORIZON, Pose, Prevention, careful_driver_avoids, careful_driver_gap,
                          careful_driver_runs_into, perceived)
from .structured import IntersectionChecker, NonIntersectionChecker
from .types import (CANONICAL, ConstellationType, LatPosition, LatResponse, LonPosition, LonResponse, ObjectType,
                    RelativeConstellation, RssParameters, RssState, UnstructuredResponse, is_dangerous)
from .unstructured import UnstructuredChecker, inside_heading_range, normalize_angle

__all__ = ["KINDS", "MARGIN_CAP", "Moment", "RssJudge", "Verdict"]

MARGIN_CAP = 50.0
"""Metres: the margin when nothing lowered it; a failure leaving more room than this does not count."""
TOLERANCE = 0.3
"""m/s^2 of record noise allowed against an owed deceleration."""
STANDING = 0.1
"""m/s below which a road user stands and owes no braking."""
LATERAL_STOPPED = 0.1
"""m/s of lateral speed towards the other up to which a lateral response is met."""
WINDOW = 0.3
"""Seconds over which accelerations are taken from the recorded speeds."""
LOOKBACK = 20.0
"""Seconds before a touch in which the careful driver may perceive its risk."""
SPEED_FILTER = 2
"""Frames each side of a median over the ego's speeds: Sim Control at times freezes the ego for 0.15 s,
reporting it standing, and the speed jumps back; a median over 5 frames leaves monotone speeds as they are."""
SHORTFALL = 0.5
"""Metres by which a careful driver must have kept clearer for a danger the other caused to lower the margin."""
KINDS = ("collision", "collision_preventable", "collision_by_other")
"""
The verdict kinds: ``collision``, the ego failed RSS (unless braking as owed would still have left it
touched); else ``collision_preventable``, a competent and careful driver would have avoided it; else
``collision_by_other``.
"""
NEAR = 10.0
"""Metres within which a pair the lanes relate in no way is judged in the plane, as unstructured."""
SLACK = 0.02
"""m/s of record noise allowed on a speed change, however short the window."""
_TYPES = {"VEHICLE": ObjectType.OTHER_VEHICLE, "BICYCLE": ObjectType.BICYCLE, "PEDESTRIAN": ObjectType.PEDESTRIAN}
"""Each recorded obstacle type's RSS object type; any other type is ``OTHER_OBJECT``."""

Checked = Tuple[RelativeConstellation, RssState]
"""A party's constellation and RSS state at one frame (``RssJudge._check``)."""
Samples = List[Tuple[float, float]]
"""``(t, value)`` in time order, ``t`` in seconds since the record began."""


@dataclass(frozen=True)
class Verdict:
    """
    Who answers for one touch (``kind``, one of ``KINDS``), and why.

    :param str obstacle: The road user touched, e.g. ``"VEHICLE 1003"``.
    :param float t: Seconds since the record began: the first frame their bodies touched.
    :param str kind: ``"collision"``, ``"collision_preventable"`` or ``"collision_by_other"``.
    :param str detail: Why, for people: who failed what, and whether a careful driver would have avoided it.
    :param float x: The ego's body centre at ``t``, map frame.
    :param float y: The ego's body centre at ``t``.
    :param str constellation: The ego's constellation with it then, e.g. ``"same direction"``.
    :param str part: Where on the ego it touched: ``front``, ``rear``, ``left`` or ``right``.
    """
    obstacle: str
    """The road user touched, e.g. ``"VEHICLE 1003"``."""
    t: float
    """Seconds since the record began: the first frame their bodies touched."""
    kind: str
    """``"collision"``, ``"collision_preventable"`` or ``"collision_by_other"``."""
    detail: str
    """Why, for people: who failed what, and whether a careful driver would have avoided it."""
    x: float
    """The ego's body centre at ``t``, map frame."""
    y: float
    """The ego's body centre at ``t``."""
    constellation: str = ""
    """The ego's constellation with it then, e.g. ``"same direction"``."""
    part: str = ""
    """Where on the ego it touched: ``front``, ``rear``, ``left`` or ``right``."""

    @property
    def ego_blamed(self) -> bool:
        """
        Whether the ego answers for it: it failed RSS, or could have prevented it.

        :returns: True unless it is ``collision_by_other``: even where RSS does not blame the ego, it answers
            for a collision it could have prevented.
        :rtype: bool
        """
        return self.kind != "collision_by_other"


@dataclass(frozen=True)
class _Owed:
    """
    One obligation: from ``start`` (the last frame it was not owed) the party
    may accelerate up to ``accel_max``, from ``due`` it must apply it.

    :param float start: Seconds since the record began: the party's previous frame, or the first frame owed if it
        was not seen before.
    :param float due: Seconds since the record began: the first frame owed, plus the party's response time.
    """
    start: float
    """Seconds since the record began: the party's previous frame, or the first frame owed if it was not seen before."""
    due: float
    """Seconds since the record began: the first frame owed, plus the party's response time."""


@dataclass
class _Side:
    """
    One party's checkers against one other, and its current danger episode.
    Frames are seconds since the record began.

    :param NonIntersectionChecker structured: Checks it in a same or opposite direction constellation.
    :param IntersectionChecker intersection: Checks it in one of the three intersection constellations.
    :param UnstructuredChecker unstructured: Checks it in an unstructured constellation.
    :param int constellation_id: The id of its current constellation, which the checkers keep their history by.
    :param int | None checked: The time index it was last checked at: its frame's index plus one.
    :param ConstellationType | None kind: The type of its current constellation; None while it has none.
    :param float | None last: Its previous frame, while it is seen frame after frame.
    :param float | None threshold: The danger threshold of the current episode: its last safe frame.
    :param float | None danger: The first dangerous frame of the current episode.
    :param Dict[str, _Owed] owed: What it owes in the episode, by direction: ``"lon"``, ``"left"`` or ``"right"``.
    :param Dict[str, str] failed: How it first failed each obligation it still owes: one no longer owed takes its
        failure with it.
    :param List[str] bounds: Where it left RSS's bounds in the episode.
    :param float | None against: When it first failed or left RSS's bounds in the episode.
    :param Samples lateral: Its lateral speed in the constellation, ``(t, m/s)``, over the latest ``2 * WINDOW``
        seconds.
    """
    structured: NonIntersectionChecker = field(default_factory=NonIntersectionChecker)
    """Checks it in a same or opposite direction constellation."""
    intersection: IntersectionChecker = field(default_factory=IntersectionChecker)
    """Checks it in one of the three intersection constellations."""
    unstructured: UnstructuredChecker = field(default_factory=UnstructuredChecker)
    """Checks it in an unstructured constellation."""
    constellation_id: int = 0
    """The id of its current constellation, which the checkers keep their history by."""
    checked: int | None = None
    """The time index it was last checked at: its frame's index plus one."""
    kind: ConstellationType | None = None
    """The type of its current constellation; None while it has none."""
    last: float | None = None
    """Its previous frame, while it is seen frame after frame."""
    threshold: float | None = None
    """The danger threshold of the current episode: its last safe frame."""
    danger: float | None = None
    """The first dangerous frame of the current episode."""
    owed: Dict[str, _Owed] = field(default_factory=dict)
    """What it owes in the episode, by direction: ``"lon"``, ``"left"`` or ``"right"``."""
    failed: Dict[str, str] = field(default_factory=dict)
    """How it first failed each obligation it still owes: one no longer owed takes its failure with it."""
    bounds: List[str] = field(default_factory=list)
    """Where it left RSS's bounds in the episode."""
    against: float | None = None
    """When it first failed or left RSS's bounds in the episode."""
    lateral: Samples = field(default_factory=list)
    """Its lateral speed in the constellation, ``(t, m/s)``, over the latest ``2 * WINDOW`` seconds."""

    @property
    def failure(self) -> str | None:
        """
        The first failure of an obligation it still owes.

        :returns: How it failed, for people, e.g. ``"did not brake: ..."``; None when it failed none.
        :rtype: str | None
        """
        return next(iter(self.failed.values()), None)

    def calm(self, t: float | None) -> None:
        """
        Safe at ``t``, or not relevant (None): the danger episode, and what it owed, are over.

        :param float | None t: The frame it is safe at, kept as the next episode's threshold; None when not
            relevant, which also forgets its constellation, previous frame and lateral speeds.
        """
        self.threshold, self.danger, self.owed, self.failed, self.bounds, self.against = t, None, {}, {}, [], None
        if t is None:
            self.last, self.kind, self.lateral = None, None, []


@dataclass(frozen=True)
class Moment:
    """
    Where the margin was least: the road user's type, the ego's constellation with it, the ego's centre.

    :param str road_user: The other's type, e.g. ``"VEHICLE"``.
    :param str constellation: The ego's constellation with it then, e.g. ``"same direction"``; ``"near miss"``
        for a danger the other caused where a careful driver would have kept clearer.
    :param float x: The ego's body centre then, map frame.
    :param float y: The ego's body centre then.
    """
    road_user: str
    """The other's type, e.g. ``"VEHICLE"``."""
    constellation: str
    """
    The ego's constellation with it then, e.g. ``"same direction"``; ``"near miss"`` for a danger the other
    caused where a careful driver would have kept clearer.
    """
    x: float
    """The ego's body centre then, map frame."""
    y: float
    """The ego's body centre then."""


@dataclass
class _Caused:
    """
    A danger the other caused (it failed or left RSS's bounds), with one obstacle.

    :param int start: The index of the frame it began, in the judge's frames.
    :param float against: Seconds since the record began: when the other first failed or left RSS's bounds.
    :param str cause: How, for people: its first failure, else the first way it left RSS's bounds.
    :param bool ego_failed: Whether the ego failed RSS against it in the danger too, whose margin then counts.
    """
    start: int
    """The index of the frame it began, in the judge's frames."""
    against: float
    """Seconds since the record began: when the other first failed or left RSS's bounds."""
    cause: str
    """How, for people: its first failure, else the first way it left RSS's bounds."""
    ego_failed: bool = False
    """Whether the ego failed RSS against it in the danger too, whose margin then counts."""


@dataclass(frozen=True)
class _Frame:
    """
    One obstacle frame of the record, as the judge keeps it.

    :param float t: Seconds since the record began.
    :param Party ego: The ego, at its body centre, on its planned route.
    :param Dict[int, Party] obstacles: Each obstacle, by id.
    :param Dict[int, str] names: Each obstacle's name, by id, e.g. ``"VEHICLE 1003"``.
    :param Dict[str, str] lights: Colour name of every reported signal, e.g. ``{"signal_9": "RED"}``.
    """
    t: float
    """Seconds since the record began."""
    ego: Party
    """The ego, at its body centre, on its planned route."""
    obstacles: Dict[int, Party]
    """Each obstacle, by id."""
    names: Dict[int, str]
    """Each obstacle's name, by id, e.g. ``"VEHICLE 1003"``."""
    lights: Dict[str, str]
    """Colour name of every reported signal, e.g. ``{"signal_9": "RED"}``."""


class RssJudge:
    """
    Collect a record's obstacle frames (``add``), then judge them (``finish``):
    ``verdicts`` per touched obstacle, ``margin`` and the ``worst`` moment.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param RssParameters params: RSS dynamics.
    :param RssMap | None rss_map: The map's lanes, when already built.
    """
    map: RssMap
    """The map's lanes, as RSS reads them."""
    vehicle: ApolloVehicleConfig
    """The ego's dimensions."""
    params: RssParameters
    """RSS dynamics."""
    verdicts: List[Verdict]
    """Who answers for each touch, in time order; one per obstacle at most."""
    margin: float
    """The ego's RSS margin in metres, ``MARGIN_CAP`` until lowered; at most 0 after a touch it answers for."""
    worst: str | None
    """What happened at the least margin, for people; None when nothing lowered it."""
    worst_at: Moment | None
    """Where that was; None when nothing lowered it."""
    _frames: List[_Frame]
    """The obstacle frames taken, in time order."""
    _held: Dict[int, Tuple[float, str, Moment]]
    """
    Per obstacle, ``(margin, worst, worst_at)`` for the least margin of the ego failing RSS against it, held
    until the danger's end: a failure that a touch then shows made no difference (``_braking_in_vain``) does
    not count.
    """
    _contacts: Dict[int, float] | None
    """When each obstacle first touched the ego, seconds since the record began; built by ``_traffic``."""

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig, params: RssParameters = CANONICAL,
                 rss_map: RssMap | None = None) -> None:
        """
        Constructor

        :param MapService map_service: The map Apollo drove.
        :param ApolloVehicleConfig vehicle: The ego's dimensions.
        :param RssParameters params: RSS dynamics.
        :param RssMap | None rss_map: The map's lanes, when already built.
        """
        self.map = rss_map if rss_map is not None else RssMap.of(map_service)
        self.vehicle = vehicle
        self.params = params
        self.verdicts = []
        self.margin = MARGIN_CAP
        self.worst = None
        self.worst_at = None
        self._frames = []
        self._held = {}
        self._contacts = None

    def add(self, world: World) -> None:
        """
        Take one obstacle frame; one without the ego's state is skipped.

        :param World world: The latest state of the record's channels.
        """
        ego = world.ego
        if ego is None:
            return
        v = self.vehicle
        x, y = v.body_center(ego.x, ego.y, ego.heading)
        me = Party(-1, ObjectType.EGO_VEHICLE, x, y, ego.heading, ego.speed,
                   v.body_length, v.body_width, ego_footprint(ego.x, ego.y, ego.heading, v),
                   tuple(world.route))
        obstacles = {o.id: Party(o.id, _TYPES.get(o.type, ObjectType.OTHER_OBJECT), o.x, o.y, o.heading, o.speed,
                                 o.length, o.width, box(o.x, o.y, o.heading, o.length, o.width))
                     for o in world.obstacles}
        names = {o.id: f"{o.type} {o.id}" for o in world.obstacles}
        self._frames.append(_Frame(world.t, me, obstacles, names, dict(world.lights)))

    def finish(self) -> None:
        """
        Judge the frames taken: fill ``verdicts``, ``margin``, ``worst`` and ``worst_at``.
        """
        routes = self._routes()
        ego_route = tuple(self._path(self._lanes([frame.ego for frame in self._frames])))
        mine: Dict[int, _Side] = {}
        theirs: Dict[int, _Side] = {}
        speeds: Dict[int, Samples] = {}
        headings: Dict[int, Samples] = {}
        seen, touched = set(), set()
        caused: Dict[int, _Caused] = {}
        last: Dict[int, int] = {}
        ego_speeds = _median([frame.ego.speed for frame in self._frames], SPEED_FILTER)
        for index, frame in enumerate(self._frames):
            speeds.setdefault(-1, []).append((frame.t, ego_speeds[index]))
            headings.setdefault(-1, []).append((frame.t, frame.ego.heading))
            ego = replace(frame.ego, speed=ego_speeds[index], yaw_rate=_yaw_rate(headings[-1]),
                          route=ego_route or frame.ego.route)
            for oid, other in frame.obstacles.items():
                speeds.setdefault(oid, []).append((frame.t, other.speed))
                headings.setdefault(oid, []).append((frame.t, other.heading))
                other = replace(other, route=routes.get(oid, ()), yaw_rate=_yaw_rate(headings[oid]))
                if oid in touched:
                    continue  # After a touch the replayed bodies pass through each other: nothing more to judge.
                last[oid] = index
                ego_side, other_side = mine.setdefault(oid, _Side()), theirs.setdefault(oid, _Side())
                if not self._relevant(ego, other):
                    ego_side.calm(None), other_side.calm(None)  # Out of each other's reach: safe.
                    self._release(oid)
                    if oid in caused:
                        self._shortfall(oid, caused.pop(oid), index)
                    continue
                ego_checked = self._check(ego_side, ego, other, frame, index + 1)
                other_checked = self._check(other_side, other, ego, frame, index + 1)
                if ego_checked is not None:
                    self._comply(ego_side, ego_checked, ego, frame.t, speeds[-1])
                if other_checked is not None:
                    self._comply(other_side, other_checked, other, frame.t, speeds[oid])
                    if oid not in seen and is_dangerous(other_checked[1]):
                        _bound(other_side, frame.t, f"appeared already in danger at t={frame.t:.2f} s")
                if ego_checked is not None or other_checked is not None:
                    seen.add(oid)
                touching = ego.body.intersects(other.body)
                if ego_checked is not None and ego_side.failure is not None:
                    self._margin(ego_checked, frame, other, touching)
                elif ego_side.failure is None:
                    self._release(oid)  # Its failure is over without a touch: the margin it held counts.
                if touching:
                    touched.add(oid)
                    caused.pop(oid, None)  # Judged as a collision instead.
                    self._verdict(index, oid, ego_side, other_side)
                    continue
                cause = other_side.failure or (other_side.bounds[0] if other_side.bounds else None)
                if other_side.against is not None and cause is not None:
                    episode = caused.setdefault(oid, _Caused(index, other_side.against, cause))
                    episode.ego_failed |= ego_side.failure is not None
                elif oid in caused:
                    self._shortfall(oid, caused.pop(oid), index)
        for oid, episode in caused.items():
            self._shortfall(oid, episode, last[oid])
        for oid in list(self._held):
            self._release(oid)

    def violations(self) -> List[Violation]:
        """
        The verdicts as violations of their kinds.

        :returns: One per verdict, in time order, at the ego's body centre.
        :rtype: List[Violation]
        """
        return [Violation(v.kind, v.t, v.obstacle, v.detail, v.x, v.y) for v in self.verdicts]

    # Frames to constellations

    def _routes(self) -> Dict[int, Tuple[str, ...]]:
        """
        The lanes each obstacle drives in the record, in order: its route (``_path``).

        :returns: Each obstacle's route, by id; pedestrians drive no lanes and have none.
        :rtype: Dict[int, Tuple[str, ...]]
        """
        tracks: Dict[int, List[Party]] = {}
        for frame in self._frames:
            for oid, other in frame.obstacles.items():
                if other.object_type != ObjectType.PEDESTRIAN:
                    tracks.setdefault(oid, []).append(other)
        return {oid: tuple(self._path(self._lanes(track))) for oid, track in tracks.items()}

    def _lanes(self, track: List[Party]) -> List[str]:
        """
        The lanes a road user was matched to frame by frame, without repeats.

        :param List[Party] track: The road user, frame by frame.
        :returns: Lane ids in the order matched, each run of frames on one lane once; frames off lanes add none.
        :rtype: List[str]
        """
        lanes: List[str] = []
        for party in track:
            lane = self.map.main_lane(replace(party, route=()))
            if lane is not None and (not lanes or lanes[-1] != lane):
                lanes.append(lane)
        return lanes

    def _path(self, lanes: List[str]) -> List[str]:
        """
        The connected lanes a road user drove, from the lanes it was matched to
        frame by frame. Where lanes fan out in a junction its body overlaps a
        sibling first (one leaving the same lane) and only later the one it
        takes, so the later sibling replaces the earlier; lanes merging into the
        same lane it overlapped on the way out are not its.

        :param List[str] lanes: The lanes it was matched to frame by frame (``_lanes``).
        :returns: Lane ids, in the order it drove them.
        :rtype: List[str]
        """
        path: List[str] = []
        for lane in lanes:
            if path and lane == path[-1]:
                continue
            if lane in path:
                del path[path.index(lane) + 1:]  # Back where it was.
                continue
            if path:
                last, info = self.map.lanes[path[-1]], self.map.lanes[lane]
                if lane not in last.successors and lane not in last.forward_neighbors:
                    if set(info.predecessors) & set(last.predecessors):
                        path[-1] = lane
                        continue
                    if set(info.successors) & set(last.successors):
                        continue
            path.append(lane)
        return path

    def _relevant(self, ego: Party, other: Party) -> bool:
        """
        Whether the pair is within both parties' reach in their response times, or a fixed radius.

        :param Party ego: The ego.
        :param Party other: The road user.
        :returns: True if their bodies are at most both reaches apart, or 30 m with a pedestrian, else 100 m.
        :rtype: bool
        """
        def reach(party: Party) -> float:
            """
            How far ``party`` may go: accelerating for its response time, then braking at ``brake_min_correct``.

            :param Party party: The road user.
            :returns: Metres; 0 when it cannot be computed.
            :rtype: float
            """
            dynamics = self.map.dynamics(party, self.params, ())
            return lon_stated_braking(party.speed, SPEED_MAX, dynamics.response_time, dynamics.alpha_lon.accel_max,
                                      dynamics.alpha_lon.brake_min_correct) or 0.0
        radius = 30.0 if ObjectType.PEDESTRIAN in (ego.object_type, other.object_type) else 100.0
        return ego.body.distance(other.body) <= max(reach(ego) + reach(other), radius)

    def _check(self, side: _Side, first: Party, second: Party, frame: _Frame, time_index: int) -> Checked | None:
        """
        ``first``'s state against ``second``, from its own checkers.

        :param _Side side: ``first``'s side against ``second``, whose checkers and constellation it updates.
        :param Party first: The party checked.
        :param Party second: The other.
        :param _Frame frame: The frame, for its lights.
        :param int time_index: The checkers' time index: the frame's index plus one.
        :returns: Its constellation and state; None when not relevant, or when its checker gives no state.
        :rtype: Checked | None
        """
        constellation = relative_constellation(self.map, first, second, frame.lights, self.params)
        if constellation is None and first.body.distance(second.body) <= NEAR:
            constellation = unstructured_constellation(self.map, first, second, self.params)
        if constellation is None:
            side.calm(None)  # Not relevant, which the library takes as safe.
            return None
        kind = constellation.constellation_type
        if kind != side.kind or side.checked != time_index - 1:
            # A new constellation where the last time index had none of its type, as
            # RssConstellationIdProvider.cpp drops ids unused then: the checkers start it without history
            # (digest §4.1), but what the party owed and failed in the danger goes on: a light changing
            # colour does not undo it.
            side.kind, side.lateral, side.constellation_id = kind, [], side.constellation_id + 1
        side.checked = time_index
        constellation = replace(constellation, constellation_id=side.constellation_id)
        if kind == ConstellationType.UNSTRUCTURED:
            state = side.unstructured.check(constellation, time_index)
        elif kind.is_intersection:
            state = side.intersection.check(constellation, time_index)
        else:
            state = side.structured.check(constellation, time_index)
        return None if state is None else (constellation, state)

    # Obligations

    def _comply(self, side: _Side, checked: Checked, party: Party, t: float, speeds: Samples) -> None:
        """
        Follow the danger episode and whether ``party`` applies its proper responses.

        :param _Side side: ``party``'s side, whose episode, obligations and failures it updates.
        :param Checked checked: ``party``'s constellation and state at ``t`` (``_check``).
        :param Party party: The party, at ``t``.
        :param float t: Seconds since the record began: the frame.
        :param Samples speeds: ``party``'s speeds up to ``t``, ``(t, m/s)``; the ego's median-filtered.
        """
        constellation, state = checked
        structured = constellation.constellation_type != ConstellationType.UNSTRUCTURED
        lat_speed = constellation.ego.structured.velocity.speed_lat_max
        if structured:
            side.lateral = [(t0, v) for t0, v in side.lateral if t - t0 <= 2 * WINDOW] + [(t, lat_speed)]
        previous, side.last = side.last, t
        if not is_dangerous(state):
            side.calm(t)
            return
        if side.danger is None:
            side.danger = t
            if side.threshold is None:
                side.threshold = t  # Dangerous at first sight.
        dynamics = constellation.ego.dynamics
        acceleration = _rate(speeds)
        if acceleration is not None and acceleration > dynamics.alpha_lon.accel_max + TOLERANCE:
            _bound(side, t, f"accelerated at {acceleration:.1f} m/s^2 at t={t:.2f} s")
        if acceleration is not None and acceleration < dynamics.alpha_lon.brake_max - TOLERANCE:
            _bound(side, t, f"braked at {acceleration:.1f} m/s^2 at t={t:.2f} s")
        lon = dynamics.alpha_lon
        owed: Dict[str, Tuple[float, str]] = {}  # Obligation: what it owes after the response time, and how.
        if not structured:
            result = state.unstructured
            if result.response == UnstructuredResponse.BRAKE:
                owed["lon"] = (lon.brake_min, "brake")
            elif result.response == UnstructuredResponse.DRIVE_AWAY:
                # Driving away it may go on accelerating, inside the heading range; outside it must brake.
                inside = inside_heading_range(normalize_angle(party.heading), result.heading_range)
                owed["lon"] = (lon.accel_max if inside else lon.brake_min,
                               f"drive away, heading within [{result.heading_range[0]:.2f}, "
                               f"{result.heading_range[1]:.2f}] rad, or brake (it headed "
                               f"{normalize_angle(party.heading):.2f} rad)")
        elif state.lon.response == LonResponse.BRAKE_MIN:
            owed["lon"] = (lon.brake_min, "brake")
        elif state.lon.response == LonResponse.BRAKE_MIN_CORRECT:
            owed["lon"] = (lon.brake_min_correct, "brake")
        if structured:
            for response, name in ((state.lat_left.response, "left"), (state.lat_right.response, "right")):
                if response == LatResponse.BRAKE_MIN:
                    owed[name] = (dynamics.alpha_lat.brake_min, "stop moving sideways")
        for key in [k for k in side.owed if k not in owed]:
            # No longer owed, say as the constellation changed and the other is now behind: the danger it
            # failed to answer is not the one it is in.
            del side.owed[key]
            side.failed.pop(key, None)
        for key in owed:
            if key not in side.owed:
                side.owed[key] = _Owed(previous if previous is not None else t, t + dynamics.response_time)
        for key, (after, what) in owed.items():
            obligation = side.owed[key]
            if key == "lon":
                if party.speed <= STANDING:
                    continue  # It stands: nothing to brake.
                found = _excess(speeds, obligation, t, lon.accel_max, after)
            else:
                towards = -1.0 if key == "left" else 1.0
                if lat_speed * towards <= LATERAL_STOPPED:
                    continue  # It no longer moves towards the other.
                found = _excess([(t0, v * towards) for t0, v in side.lateral], obligation, t,
                                dynamics.alpha_lat.accel_max, after)
            if found is None:
                continue
            rate, late = found
            sideways = "" if key == "lon" else f" sideways towards it ({key})"
            if late:
                self._fail(side, key, t, f"did not {what}: at most {after:.1f} m/s^2{sideways} owed from "
                                      f"t={obligation.due:.2f} s, it kept {rate:.1f} m/s^2")
            else:
                limit = lon.accel_max if key == "lon" else dynamics.alpha_lat.accel_max
                self._fail(side, key, t, f"accelerated{sideways} at {rate:.1f} m/s^2 in danger, at most "
                                      f"{limit:.1f} m/s^2 allowed until t={obligation.due:.2f} s")

    @staticmethod
    def _fail(side: _Side, key: str, t: float, why: str) -> None:
        """
        Record that ``side``'s party failed the obligation ``key``: only its first failure of each is kept.

        :param _Side side: The party's side.
        :param str key: The obligation: ``"lon"``, ``"left"`` or ``"right"``.
        :param float t: Seconds since the record began: the frame it failed at; ``side.against`` if none yet.
        :param str why: How it failed, for people.
        """
        side.failed.setdefault(key, why)
        if side.against is None:
            side.against = t

    # Margin and verdicts

    def _margin(self, checked: Checked, frame: _Frame, other: Party, touching: bool) -> None:
        """
        How far both were from a collision that braking as hard as they can
        (``brake_max``) from now could not avoid: the room left, less what both
        still need. In a lane, overlapping along it only the room across counts,
        overlapping across it only the room along; neither, whichever leaves
        more; both, whichever leaves less. The least for each obstacle is held
        until its danger ends (``_release``).

        :param Checked checked: The ego's constellation and state against ``other``.
        :param _Frame frame: The frame.
        :param Party other: The road user.
        :param bool touching: Whether their bodies touch: the margin is then at most 0.
        """
        constellation, state = checked
        ego, obj = constellation.ego, constellation.other
        brake_ego, brake_other = ego.dynamics.alpha_lon.brake_max, obj.dynamics.alpha_lon.brake_max
        stop_ego = stopping_distance(frame.ego.speed, brake_ego) or 0.0
        stop_other = stopping_distance(other.speed, brake_other) or 0.0
        body = frame.ego.body.distance(other.body)
        kind = constellation.constellation_type
        if kind == ConstellationType.UNSTRUCTURED:
            margin, how = _sweep_margin(frame.ego, stop_ego, other, stop_other), "on their straight paths"
        elif kind.is_intersection:
            # Clear while either can still stop short of the area both may occupy, or the bodies stay apart.
            margin = max(ego.structured.distance_to_enter_intersection - stop_ego,
                         obj.structured.distance_to_enter_intersection - stop_other, body - stop_ego - stop_other)
            how = "at the intersection"
        else:
            v_ego, v_other = ego.structured.velocity, obj.structured.velocity
            lon_ego = stopping_distance(v_ego.speed_lon_max, brake_ego) or 0.0
            lon_other = stopping_distance(v_other.speed_lon_max, brake_other) or 0.0
            if kind == ConstellationType.SAME_DIRECTION:
                behind = constellation.lon_position in (LonPosition.AT_BACK, LonPosition.OVERLAP_BACK)
                rear, front = (lon_ego, lon_other) if behind else (lon_other, lon_ego)
                along = constellation.lon_distance - max(0.0, rear - front)
            else:
                along = constellation.lon_distance - lon_ego - lon_other
            towards = 1.0 if constellation.lat_position in (LatPosition.AT_LEFT, LatPosition.OVERLAP_LEFT) else -1.0
            across = constellation.lat_distance - _lateral_stop(v_ego.speed_lat_max * towards, ego.dynamics) \
                - _lateral_stop(-v_other.speed_lat_max * towards, obj.dynamics)
            # Overlapping along the lane, only room across counts; overlapping across it, only room along.
            lon_overlap = constellation.lon_position not in (LonPosition.IN_FRONT, LonPosition.AT_BACK)
            lat_overlap = constellation.lat_position not in (LatPosition.AT_LEFT, LatPosition.AT_RIGHT)
            if lon_overlap and lat_overlap:
                margin, how = min((along, "along the lane"), (across, "across the lane"))
            elif lat_overlap:
                margin, how = along, "along the lane"
            elif lon_overlap:
                margin, how = across, "across the lane"
            else:
                margin, how = max((along, "along the lane"), (across, "across the lane"))
        if touching:
            margin, how = min(margin, 0.0), "touching"
        held = self._held.get(other.id)
        if held is None or margin < held[0]:
            self._held[other.id] = (margin, f"at t={frame.t:.2f} s the ego, failing RSS against "
                                            f"{frame.names[other.id]} ({_name(kind)}), had {margin:.1f} m to spare "
                                            f"{how} if both braked hard", _moment(frame, other.id, _name(kind)))

    def _release(self, oid: int) -> None:
        """
        Count the margin held against ``oid``, if any, towards ``margin``.

        :param int oid: The obstacle's id.
        """
        held = self._held.pop(oid, None)
        if held is not None:
            self._lower(*held)

    def _lower(self, margin: float, worst: str, at: Moment) -> None:
        """
        Lower the judge's margin to ``margin`` if it is less, with what happened then and where.

        :param float margin: Metres.
        :param str worst: What happened, for people.
        :param Moment at: Where it happened.
        """
        if margin < self.margin:
            self.margin, self.worst, self.worst_at = margin, worst, at

    def _verdict(self, index: int, oid: int, ego_side: _Side, other_side: _Side) -> None:
        """
        Judge the touch with ``oid`` and append its verdict: ``collision`` if the
        ego failed RSS, unless braking as owed would still have left it touched
        (``_braking_in_vain``); else ``collision_preventable`` if a competent and
        careful driver would have avoided it; else ``collision_by_other``. The
        margin goes to 0 when the ego answers for it.

        :param int index: The index of the frame they first touched at.
        :param int oid: The obstacle's id.
        :param _Side ego_side: The ego's side against it.
        :param _Side other_side: Its side against the ego.
        """
        frame = self._frames[index]
        name = frame.names[oid]
        due = self._braking_in_vain(index, oid, ego_side, other_side)
        if due is not None:
            self._held.pop(oid, None)  # Its failure made no difference to the touch.
        else:
            self._release(oid)
        if ego_side.failure is not None and due is None:
            kind = "collision"
            detail = f"ego ({_name(ego_side.kind)} with {name}{_since(ego_side)}) {ego_side.failure}"
            if other_side.failure is not None:
                detail += f"; {name} also {other_side.failure}"
            self._lower(0.0, f"at t={frame.t:.2f} s the ego touched {name}, failing RSS",
                        _moment(frame, oid, _name(ego_side.kind)))
        else:
            if other_side.failure is not None:
                cause = f"{name} ({_name(other_side.kind)} with the ego{_since(other_side)}) {other_side.failure}"
            elif other_side.bounds:
                cause = f"{name} ({_name(other_side.kind)}) left RSS's bounds: {other_side.bounds[0]}"
            else:
                # RSS blames only a party that failed its proper response; the ego met every one it owed.
                cause = f"no failure of {name} was observed ({_name(ego_side.kind)})"
            prevention = self._prevention(index, oid, other_side)
            careful = (f"a competent and careful driver (UN R157 Annex 3), perceiving the risk at "
                       f"t={prevention.perceived:.2f} s and braking from t={prevention.brake:.2f} s,")
            lapse = (f"the ego {ego_side.failure}, but braking so from t={due:.2f} s it would still have been touched"
                     if due is not None else None)
            if prevention.avoided:
                kind = "collision_preventable"
                detail = f"{cause}; {lapse or 'the ego responded as RSS asks'}, but {careful} would have avoided it"
                self._lower(0.0, f"at t={frame.t:.2f} s the ego touched {name}, which it could have prevented",
                            _moment(frame, oid, _name(ego_side.kind)))
            else:
                kind = "collision_by_other"
                detail = (f"{cause}; {lapse or 'the ego responded properly'}, and {careful} could not have "
                          + ("avoided it either" if prevention.into is None else
                             f"avoided it without running into {prevention.into} (UN R157 5.1.1)"))
        self.verdicts.append(Verdict(name, frame.t, kind, detail, frame.ego.x, frame.ego.y, _name(ego_side.kind),
                                     _part(frame.ego, frame.obstacles[oid])))

    def _shortfall(self, oid: int, episode: _Caused, end: int) -> None:
        """
        Lower the margin for a danger the other caused without a touch, where
        the ego failed nothing but a competent and careful driver in its place
        would have kept clearer: to how close the ego came.

        :param int oid: The obstacle's id.
        :param _Caused episode: The danger it caused.
        :param int end: The index of the frame the danger ended at, or of the obstacle's last frame.
        """
        if episode.ego_failed:
            return  # Its RSS margin counts there.
        window = [f for f in self._frames[episode.start:end + 1] if oid in f.obstacles]
        if not window:
            return
        gaps = [(f.ego.body.distance(f.obstacles[oid].body), f.t) for f in window]
        gap, closest = min(gaps)
        before = [f for f in window if f.t <= closest + 1e-6]
        sweeps = []
        for f in before:
            other = f.obstacles[oid]
            brake_min = self.map.dynamics(other, self.params, ()).alpha_lon.brake_min
            sweeps.append((f.t, _sweep(other, stopping_distance(other.speed, brake_min) or 0.0)))
        when = perceived([(f.t, f.ego.body) for f in before], sweeps, episode.against)
        brake = when + EVALUATION + DECISION
        if brake >= closest - 1e-6:
            return  # The careful driver could not have braked before the closest approach.
        ego = window[0].ego
        poses = [Pose(f.t, f.ego.x, f.ego.y, f.ego.heading, f.ego.speed) for f in self._frames]
        bodies = [(f.t, f.obstacles[oid].body) for f in window]
        careful = min([careful_driver_gap(poses, ego.length, ego.width, bodies, brake, window[-1].t)]
                      + [g for g, t in gaps if t < brake - 1e-6])
        if careful <= gap + SHORTFALL:
            return
        until = window[-1].t + HORIZON
        if careful_driver_runs_into(poses, ego.length, ego.width, self._traffic(oid, brake, until), brake, until):
            return  # Keeping clearer so, it would have run into another road user: no safer driving to hold it to.
        name = window[0].names[oid]
        at = next(f for f in window if f.t == closest)
        self._lower(gap, f"at t={closest:.2f} s the ego came within {gap:.1f} m of {name}, which {episode.cause}; "
                         f"a competent and careful driver braking from t={brake:.2f} s would have kept "
                         f"{careful:.1f} m", _moment(at, oid, "near miss"))

    def _braking_in_vain(self, index: int, oid: int, ego_side: _Side, other_side: _Side) -> float | None:
        """
        When the ego owed braking, if it failed nothing else, the other failed
        or left RSS's bounds, and braking at its ``brake_min`` from then along
        its path would still have left it touched by the other as recorded:
        its failure did not cause the touch. None otherwise, also where only
        past the end of its recorded path: the record shows nothing there.
        Braking so is not capped at where the ego went: braking later but
        harder it may have stopped short where braking as owed went on by.

        :param int index: The index of the frame they first touched at.
        :param int oid: The obstacle's id.
        :param _Side ego_side: The ego's side against it.
        :param _Side other_side: Its side against the ego.
        :returns: When the braking was due, seconds since the record began; None otherwise.
        :rtype: float | None
        """
        owed = ego_side.owed.get("lon")
        if set(ego_side.failed) != {"lon"} or owed is None or not (other_side.failure or other_side.bounds):
            return None
        frame = self._frames[index]
        if owed.due >= frame.t - 1e-6:
            return None  # Not yet owed at the touch: it failed by accelerating.
        poses = [Pose(f.t, f.ego.x, f.ego.y, f.ego.heading, f.ego.speed) for f in self._frames]
        bodies = [(f.t, f.obstacles[oid].body) for f in self._frames if oid in f.obstacles]
        avoided = careful_driver_avoids(poses, frame.ego.length, frame.ego.width, bodies, owed.due, frame.t + HORIZON,
                                        deceleration=-self.params.ego.alpha_lon.brake_min, jerk_time=0.0, on_path=True)
        return None if avoided else owed.due

    def _prevention(self, index: int, oid: int, side: _Side) -> Prevention:
        """
        What a competent and careful driver in the ego's place would have done
        about the touch at ``index``: it avoided it only without running into
        another road user (UN R157 5.1.1).

        :param int index: The index of the frame they first touched at.
        :param int oid: The obstacle's id.
        :param _Side side: Its side against the ego: from when it first went against RSS the risk may be perceived.
        :returns: When it perceived the risk and braked, whether it avoided the touch, and whom avoiding it ran into.
        :rtype: Prevention
        """
        touch = self._frames[index].t
        before = [f for f in self._frames[:index + 1] if oid in f.obstacles and f.t >= touch - LOOKBACK - 1e-6]
        sweeps = []
        for f in before:
            other = f.obstacles[oid]
            brake_min = self.map.dynamics(other, self.params, ()).alpha_lon.brake_min
            sweeps.append((f.t, _sweep(other, stopping_distance(other.speed, brake_min) or 0.0)))
        when = perceived([(f.t, f.ego.body) for f in before], sweeps, side.against)
        brake = when + EVALUATION + DECISION
        if brake >= touch - 1e-6:
            return Prevention(when, brake, False)  # Too late to brake at all.
        ego = self._frames[index].ego
        poses = [Pose(f.t, f.ego.x, f.ego.y, f.ego.heading, f.ego.speed) for f in self._frames]
        bodies = [(f.t, f.obstacles[oid].body) for f in self._frames if oid in f.obstacles]
        until = touch + HORIZON
        if not careful_driver_avoids(poses, ego.length, ego.width, bodies, brake, until):
            return Prevention(when, brake, False)
        into = careful_driver_runs_into(poses, ego.length, ego.width, self._traffic(oid, brake, until), brake, until)
        return Prevention(when, brake, into is None, into)

    def _traffic(self, oid: int, since: float, until: float) -> List[Tuple[float, Polygon, str]]:
        """
        The bodies and names of the road users but ``oid`` from ``since`` to
        ``until``, each up to its first touch with the ego: after a touch the
        replayed bodies pass through each other.

        :param int oid: The obstacle to leave out: the one judged.
        :param float since: Seconds since the record began.
        :param float until: Seconds since the record began.
        :returns: ``(t, body, name)`` of each road user in each frame, in time order.
        :rtype: List[Tuple[float, Polygon, str]]
        """
        if self._contacts is None:
            self._contacts = {}
            for f in self._frames:
                for o, other in f.obstacles.items():
                    if o not in self._contacts and f.ego.body.intersects(other.body):
                        self._contacts[o] = f.t
        return [(f.t, other.body, f.names[o]) for f in self._frames if since - 1e-6 <= f.t <= until + 1e-6
                for o, other in f.obstacles.items() if o != oid and f.t < self._contacts.get(o, float("inf"))]


def _moment(frame: _Frame, oid: int, constellation: str) -> Moment:
    """
    The ``Moment`` of ``frame`` with ``oid``.

    :param _Frame frame: The frame.
    :param int oid: The obstacle's id.
    :param str constellation: The ego's constellation with it, or ``"near miss"``.
    :returns: Its type, ``constellation`` and the ego's body centre.
    :rtype: Moment
    """
    return Moment(frame.names[oid].split()[0], constellation, frame.ego.x, frame.ego.y)


def _part(ego: Party, other: Party) -> str:
    """
    Where on the ego ``other`` is: in front within 45 degrees of its heading, behind, or to a side.

    :param Party ego: The ego.
    :param Party other: The road user, by its body centre.
    :returns: ``"front"``, ``"rear"``, ``"left"`` or ``"right"``.
    :rtype: str
    """
    bearing = normalize_angle(atan2(other.y - ego.y, other.x - ego.x) - ego.heading)
    if abs(bearing) <= pi / 4:
        return "front"
    if abs(bearing) >= 3 * pi / 4:
        return "rear"
    return "left" if bearing > 0 else "right"


def _name(kind: ConstellationType | None) -> str:
    """
    A constellation type for people.

    :param ConstellationType | None kind: The type; None for no constellation.
    :returns: Its name in lower case with spaces, e.g. ``"same direction"``; ``"no constellation"`` for None.
    :rtype: str
    """
    return "no constellation" if kind is None else kind.name.lower().replace("_", " ")


def _bound(side: _Side, t: float, why: str) -> None:
    """
    Record that ``side``'s party left RSS's bounds.

    :param _Side side: The party's side.
    :param float t: Seconds since the record began: the frame; ``side.against`` if none yet.
    :param str why: How, for people, e.g. ``"accelerated at 4.2 m/s^2 at t=3.10 s"``.
    """
    side.bounds.append(why)
    if side.against is None:
        side.against = t


def _since(side: _Side) -> str:
    """
    When ``side``'s party was last safe before its danger, for a verdict's detail.

    :param _Side side: The party's side.
    :returns: ``", in danger after t=... s"`` from its threshold; empty without one.
    :rtype: str
    """
    return "" if side.threshold is None else f", in danger after t={side.threshold:.2f} s"


def _sweep(party: Party, stop: float) -> Polygon:
    """
    The ground its body covers braking straight on to a stop ``stop`` metres ahead.

    :param Party party: The road user.
    :param float stop: Metres to its stop, along its heading.
    :returns: The convex hull of its body now and at the stop.
    :rtype: Polygon
    """
    moved = translate(party.body, cos(party.heading) * stop, sin(party.heading) * stop)
    return party.body.union(moved).convex_hull


def _sweep_margin(ego: Party, stop_ego: float, other: Party, stop_other: float) -> float:
    """
    The gap between the grounds both cover braking straight on; where they
    meet, how far short of its stop the ego reaches the other's (at most 0).

    :param Party ego: The ego.
    :param float stop_ego: Metres to its stop.
    :param Party other: The road user.
    :param float stop_other: Metres to its stop.
    :returns: Metres: the gap where they do not meet; else at most 0.
    :rtype: float
    """
    mine, theirs = _sweep(ego, stop_ego), _sweep(other, stop_other)
    if mine.disjoint(theirs):
        return mine.distance(theirs)
    c, s = cos(ego.heading), sin(ego.heading)
    front = ego.x + c * ego.length / 2, ego.y + s * ego.length / 2
    meet = mine.intersection(theirs)
    points = [p for geom in getattr(meet, "geoms", (meet,)) for p in _points(geom)]
    first = min(((x - front[0]) * c + (y - front[1]) * s for x, y in points), default=0.0)
    return min(0.0, first - stop_ego)


def _points(geom) -> List[Tuple[float, float]]:
    """
    The vertices of a shapely geometry.

    :param BaseGeometry geom: A polygon, by its exterior ring, or a line or point.
    :returns: Its ``(x, y)`` coordinates.
    :rtype: List[Tuple[float, float]]
    """
    return list(geom.exterior.coords) if hasattr(geom, "exterior") else list(geom.coords)


def _lateral_stop(speed: float, dynamics) -> float:
    """
    Metres to stop moving sideways at ``speed`` (towards the other when positive) at ``|lat brake_min|``.

    :param float speed: m/s sideways, towards the other when positive.
    :param Dynamics dynamics: The party's RSS dynamics.
    :returns: Metres; 0 when it does not move towards the other.
    :rtype: float
    """
    return speed * speed / (2 * -dynamics.alpha_lat.brake_min) if speed > 0 else 0.0


def _median(values: List[float], half: int) -> List[float]:
    """
    Each value replaced by the median of the ``2 * half + 1`` around it (fewer at the ends).

    :param List[float] values: The values, in order.
    :param int half: How many values each side of each count.
    :returns: The filtered values, as many as given.
    :rtype: List[float]
    """
    out = []
    for i in range(len(values)):
        window = sorted(values[max(0, i - half):i + half + 1])
        out.append(window[len(window) // 2])
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """
    How long the intervals ``[a0, a1]`` and ``[b0, b1]`` overlap.

    :param float a0: The first's start.
    :param float a1: The first's end.
    :param float b0: The second's start.
    :param float b1: The second's end.
    :returns: The overlap's length; 0 when they do not overlap.
    :rtype: float
    """
    return max(0.0, min(a1, b1) - max(a0, b0))


def _excess(samples: Samples, owed: _Owed, t: float, before: float, after: float) -> Tuple[float, bool] | None:
    """
    Whether the sampled speed rose more over the latest window than allowed:
    ``before`` m/s^2 from the obligation's start to its due time, ``after``
    from then on. Before the due time the window starts at the obligation's
    start; after it, at the last sample before the due time, so that time the
    party was allowed to wait does not water its braking down.

    :param Samples samples: The party's speeds up to ``t``, ``(t, m/s)``: forward, or sideways towards the other.
    :param _Owed owed: The obligation.
    :param float t: Seconds since the record began: the latest sample's.
    :param float before: m/s^2 allowed until the due time: its ``accel_max``.
    :param float after: m/s^2 allowed from the due time: what it owes, e.g. its ``brake_min``.
    :returns: The mean rate over the window and whether it was after the due time; None when within.
    :rtype: Tuple[float, bool] | None
    """
    late = t > owed.due + 1e-6
    begin = owed.start
    if late:
        begin = max((t0 for t0, _ in samples if t0 <= owed.due + 1e-6), default=owed.start)
    begin = max(begin, t - WINDOW)
    first = next(((t0, v) for t0, v in samples if t0 >= begin - 1e-6 and t0 < t - 1e-6), None)
    if first is None:
        return None
    t0, v0 = first
    change, span = samples[-1][1] - v0, t - t0
    allowed = before * _overlap(t0, t, owed.start, owed.due) + after * _overlap(t0, t, owed.due, t) \
        + TOLERANCE * span + SLACK
    return (change / span, late) if change > allowed else None


def _yaw_rate(headings: Samples) -> float:
    """
    The latest yaw rate, rad/s, over about ``WINDOW`` seconds of headings.

    :param Samples headings: Headings by time, ``(t, rad)``.
    :returns: rad/s, counter-clockwise; 0 with fewer than two headings.
    :rtype: float
    """
    if len(headings) < 2:
        return 0.0
    t, heading = headings[-1]
    for t0, heading0 in reversed(headings[:-1]):
        if t - t0 >= WINDOW - 1e-6:
            break
    return normalize_angle(heading - heading0) / (t - t0) if t > t0 else 0.0


def _rate(samples: Samples) -> float | None:
    """
    How fast the latest sampled value changes, over about ``WINDOW`` seconds, else since the first sample.

    :param Samples samples: Values by time, ``(t, value)``.
    :returns: Per second; None with fewer than two samples.
    :rtype: float | None
    """
    if len(samples) < 2:
        return None
    t, value = samples[-1]
    for t0, value0 in reversed(samples[:-1]):
        if t - t0 >= WINDOW - 1e-6:
            return (value - value0) / (t - t0)
    t0, value0 = samples[0]
    return (value - value0) / (t - t0) if t > t0 else None
