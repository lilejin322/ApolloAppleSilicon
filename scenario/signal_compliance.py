"""
Signal compliance of replayed agents: dwell at red stop lines and at crosswalks without WALK.

Every NPC, bicycle and pedestrian carries an ``obeys_signals`` gene. A complying agent's
replay gets a forced stop before each signal-controlled stop line (vehicles, bicycles on
lanes) or signalised crosswalk (pedestrians) it reaches while it may not go, and departs
at the first instant it may. A violating agent replays its speed profile unchanged and
runs the red. With the signal plans fixed the result is deterministic.

Vehicles may go on GREEN and YELLOW and stop on RED (no dilemma-zone model). Pedestrians
may enter only on WALK; FLASHING DON'T WALK and DON'T WALK both hold them.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from shapely.geometry import LineString, Point

from .replay import ReplayTrajectory
from .traffic_signal import PedState, SignalColor, SignalSection

COMPLIANCE_RATES: Mapping[str, float] = {"npc": 0.85, "bicycle": 0.70, "pedestrian": 0.60}
"""Probability that a randomly drawn agent of each type obeys signals."""
STOP_SETBACK: float = 0.5
"""Metres between the stop line (crosswalk edge) and the front of the stopped body."""
STOP_LINE_MATCH: float = 3.0
"""Metres within which a signal's stop line is matched to a controlled lane (``get_signal_stop_lines``)."""


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
    :returns: Stops sorted by distance, at most one per signal; stops at or behind the start are dropped.
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
                    distance = min(path.project(p) for p in points) - half_length - STOP_SETBACK
                    found[signal_id] = min(found.get(signal_id, distance), distance)
    return tuple(sorted(SignalStop(d, "signal", s) for s, d in found.items() if d > 0.5))


def crosswalk_stops(network: Any, path: LineString, signals: SignalSection,
                    half_length: float) -> Tuple[SignalStop, ...]:
    """
    Entries of ``path`` into crossings whose crosswalks have a pedestrian signal in ``signals``.

    :param Any network: Walk network (``crossings``).
    :param LineString path: Replay path.
    :param SignalSection signals: Plans in force.
    :param float half_length: Half the walker's body length.
    :returns: Stops sorted by distance; a crossing entered at or before the start is dropped.
    :rtype: Tuple[SignalStop, ...]
    """
    controlled = set(signals.crosswalks)
    stops = []
    for crossing in network.crossings.values():
        ids = [c for c in crossing.crosswalks if c in controlled]
        if not ids or not path.intersects(crossing.polygon) or crossing.polygon.contains(Point(path.coords[0])):
            continue
        entry = _first_crossing(path, crossing.polygon.boundary)
        if entry is None:
            continue
        distance = entry - half_length - STOP_SETBACK
        if distance > 0.5:
            stops.append(SignalStop(distance, "crosswalk", ids[0]))
    return tuple(sorted(stops))


def may_go(signals: SignalSection, stop: SignalStop, t: float) -> bool:
    """
    Whether an agent at ``stop`` may proceed at scenario time ``t``.

    :param SignalSection signals: Plans in force.
    :param SignalStop stop: The stop.
    :param float t: Scenario seconds.
    :returns: Vehicles: not RED. Pedestrians: WALK.
    :rtype: bool
    """
    if stop.kind == "crosswalk":
        return signals.ped_state(stop.control_id, t) == PedState.WALK
    return signals.color(stop.control_id, t) != SignalColor.RED


def next_go(signals: SignalSection, stop: SignalStop, t: float) -> Optional[float]:
    """
    First instant at or after ``t`` when an agent at ``stop`` may proceed.

    :param SignalSection signals: Plans in force.
    :param SignalStop stop: The stop.
    :param float t: Scenario seconds.
    :returns: ``t`` if it may go now; else the next start of green (vehicles) or WALK (pedestrians);
        ``None`` for a crosswalk that never shows WALK.
    :rtype: Optional[float]
    """
    if may_go(signals, stop, t):
        return t
    if stop.kind == "crosswalk":
        return signals.next_walk(stop.control_id, t)
    return signals.next_green(stop.control_id, t)


def comply(build: Callable[[Dict[float, float]], ReplayTrajectory], forced: Mapping[float, float],
           stops: Iterable[SignalStop], signals: SignalSection, length: float) -> ReplayTrajectory:
    """
    The replay with a dwell added at every stop reached while the agent may not go.

    Stops are settled in path order: a stop reached on a go passes untouched; one reached
    otherwise becomes a forced stop with dwell until the next go, and later stops see the
    delay. Deterministic for fixed plans.

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
    for stop in sorted(stops):
        if not 0.5 < stop.distance < length - 0.5:
            continue
        if may_go(signals, stop, replay.time_at(stop.distance)):
            continue
        trial = dict(held)
        trial[stop.distance] = max(trial.get(stop.distance, 0.0), 0.0)
        arrival = build(trial).time_at(stop.distance)
        go = next_go(signals, stop, arrival)
        if go is None:
            continue
        trial[stop.distance] = max(held.get(stop.distance, 0.0), round(go - arrival, 6))
        held = trial
        replay = build(held)
    return replay


def active(signals: Optional[SignalSection], agent: Any) -> bool:
    """Whether ``agent``'s replay depends on ``signals``: plans exist and the agent obeys them."""
    return bool(signals is not None and signals.plans and getattr(agent, "obeys_signals", True))
