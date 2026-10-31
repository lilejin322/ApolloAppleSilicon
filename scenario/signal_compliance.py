"""
Signal compliance of replayed agents: dwell at red stop lines and at signalised crosswalks.

Every NPC, bicycle and pedestrian carries an ``obeys_signals`` gene. A complying agent's
replay gets a forced stop before each signal-controlled stop line (vehicles, bicycles on
road lanes), bicycle connector across a signalised junction (bike lanes carry no signal, so
riders follow the car movement beside them) or signalised crosswalk (pedestrians) it
reaches while it may not go, and departs at the first instant it may. A violating agent replays its speed profile unchanged and
runs the red. With the signal plans fixed the result is deterministic.

Vehicles may go on GREEN and YELLOW and stop on RED (no dilemma-zone model). The maps
have no pedestrian heads, so walkers follow the vehicle light parallel to their crossing
(``SignalSection.parallel_signal``): they may step onto the road only on its GREEN;
YELLOW and RED hold them. The light is judged when the front reaches the line (the stop
line, or the curb of a crossing), so reaching the stop point on the last instant of yellow
does not let an agent cross on red.
"""
from __future__ import annotations

from itertools import groupby
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from shapely.geometry import LineString, Point

from .bike_connectors import parallel_car_lanes
from .replay import ReplayTrajectory
from .traffic_signal import SignalColor, SignalSection

COMPLIANCE_RATES: Mapping[str, float] = {"npc": 0.85, "bicycle": 0.70, "pedestrian": 0.60}
"""Probability that a randomly drawn agent of each type obeys signals."""
STOP_SETBACK: float = 0.5
"""Metres between the stop line (crosswalk edge) and the front of the stopped body."""
STOP_LINE_MATCH: float = 3.0
"""Metres within which a signal's stop line is matched to a controlled lane (``get_signal_stop_lines``)."""
_GO_ROUNDS: int = 8
"""Rounds of ``next_go`` over the stops at one place before they are taken never to allow going together."""


class SignalStop(NamedTuple):
    """
    One place on an agent's path where a signal may hold it.

    :param float distance: Path metres of the stopped body's reference point.
    :param str kind: ``"signal"`` (stop line of a traffic signal) or ``"crosswalk"``.
    :param str control_id: Signal ID or crosswalk ID.
    """
    distance: float
    kind: str
    control_id: str


def _first_crossing(path: LineString, geometry: Any) -> Optional[float]:
    """Smallest path distance where ``path`` meets ``geometry``; ``None`` if they do not meet."""
    hit = path.intersection(geometry)
    if hit.is_empty:
        return None
    points = [hit] if hit.geom_type == "Point" else [Point(c) for g in getattr(hit, "geoms", [hit])
                                                      for c in g.coords]
    return min(path.project(p) for p in points)


def lane_signal_stops(service: Any, lanes: Iterable[str], path: LineString, signals: SignalSection,
                      half_length: float) -> Tuple[SignalStop, ...]:
    """
    Stop lines of the signals in ``signals`` that control a lane of the route, on ``path``.

    :param Any service: Map service (``relations``, lane centerlines).
    :param Iterable[str] lanes: Route lane IDs; unknown (virtual) lanes are skipped.
    :param LineString path: Replay path.
    :param SignalSection signals: Plans in force; signals without a plan never hold.
    :param float half_length: Half the body length; the stop keeps the front ``STOP_SETBACK`` behind the line.
    :returns: Stops sorted by distance, one per stop line and phase (signals of one phase show one colour);
        a line the front has already passed at the start is dropped, one closer than the setback is held at 0.
    :rtype: Tuple[SignalStop, ...]
    """
    relations = service.relations
    found: Dict[str, float] = {}
    for lane in lanes:
        try:
            ids = relations.signals_for_lane[lane]
        except KeyError:
            continue
        for signal_id in ids:
            if signal_id in found:
                continue
            try:
                signals.plan_of(signal_id)
                matches = relations.get_signal_stop_lines(signal_id, lane, STOP_LINE_MATCH)
            except (KeyError, ValueError):
                continue
            centerline = service.get_lane_central_curve_by_id(lane)
            for match in matches:
                hit = path.intersection(match.geometry)
                if hit.is_empty:
                    continue
                points = [hit] if hit.geom_type == "Point" else [Point(c) for g in hit.geoms for c in g.coords]
                points = [p for p in points if p.distance(centerline) <= STOP_LINE_MATCH]
                if points:
                    line = min(path.project(p) for p in points)
                    found[signal_id] = min(found.get(signal_id, line), line)
    return _one_per_phase(signals, ((max(line - half_length - STOP_SETBACK, 0.0), signal_id)
                                    for signal_id, line in found.items() if line - half_length > 0))


def connector_signal_stops(space: Any, lanes: Iterable[str], layout: Any, signals: SignalSection,
                           half_length: float) -> Tuple[SignalStop, ...]:
    """
    Stops of a bicycle's virtual connectors across signalised junctions.

    Bike lanes have no signal relations, so each connector leg obeys the planned signals of the car
    lanes it rides beside (``bike_connectors.parallel_car_lanes``). The first leg holds the rider
    before the bike lane's dead end, where it enters the junction (the car stop line lies within
    half a metre of it for the middle half of the connectors on the bundled maps); the second leg of
    a two-stage left holds it at the corner, where it stops to turn anyway.

    :param Any space: The bicycles' ``RouteSpace`` (``virtual_lane``, ``service``).
    :param Iterable[str] lanes: Route lane IDs; map lanes are skipped (``lane_signal_stops`` covers them).
    :param Any layout: The route's ``ReplayLayout`` (``position``, ``pivots``).
    :param SignalSection signals: Plans in force; signals without a plan never hold.
    :param float half_length: Half the body length; the stop keeps the front ``STOP_SETBACK`` behind the dead end.
    :returns: Stops sorted by distance, one per place and phase.
    :rtype: Tuple[SignalStop, ...]
    """
    relations = space.service.relations
    found: List[Tuple[float, str]] = []
    for lane_id in lanes:
        connector = space.virtual_lane(lane_id)
        if connector is None:
            continue
        for leg, cars in enumerate(parallel_car_lanes(space.service, connector)):
            ids = sorted({s for car in cars for s in relations.signals_for_lane.get(car, ()) if _planned(signals, s)})
            if not ids:
                continue
            if leg == 0:
                line = layout.position(space, lane_id, 0.0)
                if line is None or line - half_length <= 0:
                    continue
                distance = max(line - half_length - STOP_SETBACK, 0.0)
            else:
                corner = layout.position(space, lane_id, connector.pivots[leg - 1])
                if corner is None:
                    continue
                distance = min(layout.pivots, key=lambda pivot: abs(pivot - corner), default=corner)
            found.extend((distance, signal_id) for signal_id in ids)
    return _one_per_phase(signals, found)


def _planned(signals: SignalSection, signal_id: str) -> bool:
    """Whether ``signal_id`` has a plan in ``signals``."""
    try:
        signals.plan_of(signal_id)
    except KeyError:
        return False
    return True


def _one_per_phase(signals: SignalSection, found: Iterable[Tuple[float, str]]) -> Tuple[SignalStop, ...]:
    """
    Signal stops from ``(distance, signal ID)`` pairs, one per distance and phase: signals of one
    phase always show one colour, so one stop decides for all of them.

    :param SignalSection signals: Plans in force; every signal has one.
    :param Iterable[Tuple[float, str]] found: Stop distances and their signals.
    :returns: Stops sorted by distance, each with the smallest signal ID of its phase there.
    :rtype: Tuple[SignalStop, ...]
    """
    stops: Dict[Tuple[float, str, int], str] = {}
    for distance, signal_id in sorted(found):
        plan = signals.plan_of(signal_id)
        phase = next(index for index, group in enumerate(plan.phases) if signal_id in group)
        stops.setdefault((distance, plan.junction_id, phase), signal_id)
    return tuple(sorted(SignalStop(distance, "signal", s) for (distance, _, _), s in stops.items()))


def crosswalk_stops(network: Any, route: Any, layout: Any, signals: SignalSection,
                    half_length: float) -> Tuple[SignalStop, ...]:
    """
    Where a walk steps onto a signalised crossing: the curb of each ``CROSSWALK`` excursion on a
    crossing with a crosswalk in ``signals.crosswalks``, and of each ``NEAR_CROSSWALK`` one beside it.

    Only the excursion's own road entry counts: walking along a sidewalk over the painted end of a
    crosswalk, or crossing mid-block or diagonally, is not governed by a signal.

    :param Any network: Walk network (``crossing_entries``).
    :param Any route: The walk (``WalkRoute``).
    :param Any layout: Its replay layout (``WalkLayout``: ``moved`` maps walked metres to path metres).
    :param SignalSection signals: Plans in force.
    :param float half_length: Half the walker's body length; the stop keeps the front ``STOP_SETBACK`` off the road.
    :returns: Stops sorted by distance; a walker already stepping onto the road at its start has none there,
        one that starts closer than the setback waits where it stands.
    :rtype: Tuple[SignalStop, ...]
    """
    controlled = set(signals.crosswalks)
    stops = []
    for crossing, walked in network.crossing_entries(route):
        ids = [c for c in crossing.crosswalks if c in controlled]
        line = layout.moved(walked)
        if ids and line - half_length > 0:
            stops.append(SignalStop(max(line - half_length - STOP_SETBACK, 0.0), "crosswalk", ids[0]))
    return tuple(sorted(stops))


def may_go(signals: SignalSection, stop: SignalStop, t: float) -> bool:
    """
    Whether an agent at ``stop`` may proceed at scenario time ``t``.

    :param SignalSection signals: Plans in force.
    :param SignalStop stop: The stop.
    :param float t: Scenario seconds.
    :returns: Vehicles: not RED. Pedestrians: the parallel vehicle light is GREEN.
    :rtype: bool
    """
    if stop.kind == "crosswalk":
        return signals.color(signals.parallel_signal(stop.control_id), t) == SignalColor.GREEN
    return signals.color(stop.control_id, t) != SignalColor.RED


def _go_together(signals: SignalSection, stops: Tuple[SignalStop, ...], t: float) -> Optional[float]:
    """
    First instant at or after ``t`` when every stop of one place may go.

    :param SignalSection signals: Plans in force.
    :param Tuple[SignalStop, ...] stops: Stops at one distance.
    :param float t: Scenario seconds.
    :returns: That instant; ``None`` if one never allows going, or they do not agree within ``_GO_ROUNDS``.
    :rtype: Optional[float]
    """
    for _ in range(_GO_ROUNDS):
        later = [next_go(signals, stop, t) for stop in stops]
        if None in later:
            return None
        if max(later) == t:
            return t
        t = max(later)
    return None


def next_go(signals: SignalSection, stop: SignalStop, t: float) -> Optional[float]:
    """
    First instant at or after ``t`` when an agent at ``stop`` may proceed.

    :param SignalSection signals: Plans in force.
    :param SignalStop stop: The stop.
    :param float t: Scenario seconds.
    :returns: ``t`` if it may go now; else the next start of green of its signal (vehicles) or of
        the parallel vehicle light (pedestrians).
    :rtype: Optional[float]
    """
    if may_go(signals, stop, t):
        return t
    if stop.kind == "crosswalk":
        return signals.next_green(signals.parallel_signal(stop.control_id), t)
    return signals.next_green(stop.control_id, t)


def comply(build: Callable[[Dict[float, float]], ReplayTrajectory], forced: Mapping[float, float],
           stops: Iterable[SignalStop], signals: SignalSection, length: float) -> ReplayTrajectory:
    """
    The replay with a dwell added at every stop reached while the agent may not go.

    Stops are settled in path order, those at one distance together: they pass untouched when
    every one may go as the front reaches the line. Otherwise the agent waits at its last
    standstill before the line (a pivot, an own stop such as a walker's curb wait, or its
    start), or stops at the stop point if it has none, until it can both leave and reach the
    line on a go (``_hold``); the wait adds to any dwell it already has there, and later stops
    see the delay. Deterministic for fixed plans.

    :param Callable build: Builds the replay from forced stops ``{path metres: dwell seconds}``.
    :param Mapping[float, float] forced: The agent's own forced stops (pivots).
    :param Iterable[SignalStop] stops: Candidate stops.
    :param SignalSection signals: Plans in force.
    :param float length: Path length; stops at or past the end are skipped.
    :returns: The compliant replay.
    :rtype: ReplayTrajectory
    """
    held: Dict[float, float] = dict(forced)
    replay = build(held)
    for distance, group in groupby(sorted(stops), key=lambda stop: stop.distance):
        group = tuple(group)
        if not 0.0 <= distance < length - 0.5:
            continue
        if all(may_go(signals, stop, replay.time_at(distance + STOP_SETBACK)) for stop in group):
            continue
        trial = dict(held)
        point = replay.last_rest(distance, distance + STOP_SETBACK)
        if point is None:                       # it would roll over the line: stop it at the stop point
            point, trial[distance] = distance, 0.0
            stopped = build(trial)
        else:                                   # it already stands still before the line: wait there
            stopped = replay
        wait = _hold(signals, group, stopped.departure_at(point), stopped.time_at(distance + STOP_SETBACK))
        if wait is None:
            continue
        trial[point] = trial.get(point, 0.0) + wait
        held = trial
        replay = build(held)
    return replay


def _hold(signals: SignalSection, stops: Tuple[SignalStop, ...], depart: float, line: float) -> Optional[float]:
    """
    Least extra dwell at one place after which the agent both leaves it and reaches the line on a go.

    A dwell there delays both instants equally. Own waits between the stop and the line (a pivot,
    a walker's wait at the curb) keep the line later than the departure.

    :param SignalSection signals: Plans in force.
    :param Tuple[SignalStop, ...] stops: Stops at one distance.
    :param float depart: When the agent, stopped there, would leave without extra dwell.
    :param float line: When its front would then reach the line.
    :returns: The extra dwell in seconds; ``None`` if no instant within ``_GO_ROUNDS`` satisfies both.
    :rtype: Optional[float]
    """
    extra = 0.0
    for _ in range(_GO_ROUNDS):
        leave = _go_together(signals, stops, depart + extra)
        if leave is None:
            return None
        extra = leave - depart
        cross = _go_together(signals, stops, line + extra)
        if cross is None:
            return None
        if cross == line + extra:
            return extra
        extra = cross - line
    return None


def active(signals: Optional[SignalSection], agent: Any) -> bool:
    """Whether ``agent``'s replay depends on ``signals``: plans exist and the agent obeys them."""
    return bool(signals is not None and signals.plans and getattr(agent, "obeys_signals", True))
