"""
Planning's own verdicts: emergency stops, lane changes in junctions, and a
Planning that fails or falls silent.
"""

from dataclasses import dataclass
from math import isfinite
from typing import List, Set, Tuple
from google.protobuf.message import Message
from shapely.geometry import Point, Polygon

from ...map import MapService
from ...map.geometry import polygon_to_shape
from ..replay import COMMAND_STATUS, PLANNING, PLANNING_COMMAND, World
from .base import Oracle, Violation

__all__ = ["EStopOracle", "JunctionLaneChangeOracle", "PlanningFailureOracle"]


def _enum_name(message: Message, field: str) -> str:
    """
    Resolve the symbolic name of a protobuf enum field at its current value.

    :param Message message: The message.
    :param str field: The name of one of its enum fields.
    :returns: The name of the field's value, e.g. ``"ERROR"``.
    :rtype: str
    """
    return message.DESCRIPTOR.fields_by_name[field].enum_type.values_by_number[getattr(message, field)].name


class EStopOracle(Oracle):
    """
    Report each reason for which Planning decided an emergency stop, once.

    Read both ``ADCTrajectory.decision.main_decision.estop`` and the independent
    ``ADCTrajectory.estop`` used when Planning publishes an emergency stop on failure.
    """
    kind = "estop"
    """The ``Violation.kind`` it reports; the subject is the main reason code or top-level reason text."""
    channels = frozenset({PLANNING})
    """The planning messages, which carry the decision."""
    _reasons: Set[str]
    """The reason codes or top-level reason texts reported."""

    def __init__(self) -> None:
        """
        Start with no reported emergency-stop reasons.
        """
        super().__init__()
        self._reasons = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Report each distinct emergency-stop reason found in Planning messages.

        :param World world: The world at the planning message.
        :param str channel: ``PLANNING``.
        :param Message message: The ``ADCTrajectory``.
        """
        decision = message.decision.main_decision
        if decision.HasField("estop"):
            reason = _enum_name(decision.estop, "reason_code")
            detail = decision.estop.reason or reason
        elif message.estop.is_estop:
            reason = message.estop.reason or "ESTOP"
            detail = reason
        else:
            return
        if reason not in self._reasons:
            self._reasons.add(reason)
            self.report(world, reason, f"emergency stop: {detail}")


class JunctionLaneChangeOracle(Oracle):
    """
    Report each junction in which Planning decided to change lanes, once.

    The decision is the ``change_lane_type`` of the main cruise or stop
    decision; the ego is in a junction when its reference point lies in the
    junction's polygon.

    :param MapService map_service: The map Apollo drives.
    """
    kind = "junction_lane_change"
    """The ``Violation.kind`` it reports; the subject is the junction's id."""
    channels = frozenset({PLANNING})
    """The planning messages, which carry the decision."""
    _areas: List[Tuple[str, Polygon]]
    """Each junction with a polygon: its id and area."""
    _junctions: Set[str]
    """The junctions reported."""

    def __init__(self, map_service: MapService) -> None:
        """
        Cache usable junction polygons and initialize per-junction reporting state.

        :param MapService map_service: The map Apollo drives.
        """
        super().__init__()
        self._areas = [(junction_id, area) for junction_id, junction in map_service.junction_table.items()
                       if (area := polygon_to_shape(junction.polygon)) is not None]
        self._junctions = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Report a left or right lane-change decision inside each junction at most once.

        :param World world: The world at the planning message.
        :param str channel: ``PLANNING``.
        :param Message message: The ``ADCTrajectory``.
        """
        decision = message.decision.main_decision
        main = decision.cruise if decision.HasField("cruise") else decision.stop if decision.HasField("stop") else None
        if main is None or world.ego is None:
            return
        change = _enum_name(main, "change_lane_type")
        if change not in ("LEFT", "RIGHT"):
            return
        here = Point(world.ego.x, world.ego.y)
        for junction, area in self._areas:
            if area.covers(here) and junction not in self._junctions:
                self._junctions.add(junction)
                self.report(world, junction, f"lane change to the {change.lower()} inside junction {junction}")


@dataclass
class _CommandError:
    """
    One command's error, timed on one clock and located at its first reception.

    :param float start: The first ERROR status' stamp on the episode's clock: the status' source time, or record
        time for statuses without a finite timestamp.
    :param float end: The latest ERROR status' stamp, on the same clock.
    :param float received: Seconds since the record began at which the first ERROR status was received.
    :param Tuple[float | None, float | None] position: The ego's reference point ``(x, y)`` at ``received``, map
        frame; ``(None, None)`` without an ego state.
    :param str message: The latest ERROR status' message.
    """
    start: float
    """The first ERROR status' stamp on the episode's clock: the status' source time, or record time for statuses
    without a finite timestamp."""
    end: float
    """The latest ERROR status' stamp, on the same clock."""
    received: float
    """Seconds since the record began at which the first ERROR status was received."""
    position: Tuple[float | None, float | None]
    """The ego's reference point ``(x, y)`` at ``received``, map frame; ``(None, None)`` without an ego state."""
    message: str
    """The latest ERROR status' message."""


class PlanningFailureOracle(Oracle):
    """
    Report Planning silence and sustained errors for the current command.

    Silence is measured in receive time from the first valid command or the
    latest Planning message. Command changes do not postpone an existing silence.
    Errors are measured in status source time, with receive time as a fallback
    for old messages without a finite timestamp. Switching commands or clocks
    closes the error episode. Duplicate and older statuses on the same clock
    are ignored. Report times remain relative to record reception.

    :param float max_gap: Seconds of Planning silence still tolerated.
    :param float max_error: Seconds of ERROR status required for a report.
    """
    kind = "planning_error"
    """The ``Violation.kind`` of a sustained error; silence is reported as ``planning_gap``. The subject of both
    is ``"planning"``."""
    channels = frozenset({PLANNING, COMMAND_STATUS, PLANNING_COMMAND})
    """Planning's trajectories (silence), the command statuses (errors) and the commands (which one is current)."""
    max_gap: float
    """Seconds of Planning silence still tolerated."""
    max_error: float
    """Seconds of ERROR status required for a report."""
    _last: float | None
    """Record time silence is counted from: the latest Planning message, or the first command when none came
    before it; None until either."""
    _command_id: int | None
    """The current command's ID; None before the first command."""
    _last_status: Tuple[bool, float] | None
    """The latest status accepted for the current command: whether its stamp is source time, and the stamp."""
    _error: _CommandError | None
    """The current command's open error episode; None while its latest status is not ERROR."""

    def __init__(self, *, max_gap: float = 2.0, max_error: float = 1.0) -> None:
        """
        Configure silence and error thresholds in seconds.

        :param float max_gap: Seconds of Planning silence still tolerated.
        :param float max_error: Seconds of ERROR status required for a report.
        """
        super().__init__()
        self.max_gap = max_gap
        self.max_error = max_error
        self._last = None
        self._command_id = None
        self._last_status = None
        self._error = None

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Observe the active command, Planning reception, or its matching status.

        A new command closes the open error and starts the silence clock if nothing has yet. A Planning message
        checks the silence before it and restarts the clock. A status of the current command opens, extends or
        closes the error episode; one of another command, or not newer than the last on its clock, is ignored.

        :param World world: The world at the message; ``world.command`` is the current command.
        :param str channel: ``PLANNING``, ``COMMAND_STATUS`` or ``PLANNING_COMMAND``.
        :param Message message: The ``ADCTrajectory``, ``CommandStatus`` or ``PlanningCommand``.
        """
        command_id = None if world.command is None else world.command.command_id
        if command_id != self._command_id:
            self._close_error()
            self._command_id = command_id
            self._last_status = None
            if command_id is not None and self._last is None:
                self._last = world.t
        if channel == PLANNING:
            self._gap(world)
            self._last = world.t
            return
        if channel != COMMAND_STATUS or command_id is None or message.command_id != command_id:
            return
        source_time = message.header.HasField("timestamp_sec") and isfinite(message.header.timestamp_sec)
        stamp = message.header.timestamp_sec if source_time else world.t
        if self._last_status is not None:
            previous_source, previous_stamp = self._last_status
            if previous_source != source_time:
                self._close_error()
            elif stamp <= previous_stamp:
                return
        self._last_status = source_time, stamp
        if _enum_name(message, "status") != "ERROR":
            self._close_error()
            return
        if self._error is None:
            ego = world.ego
            position = (None, None) if ego is None else (ego.x, ego.y)
            self._error = _CommandError(stamp, stamp, world.t, position, message.message)
        else:
            self._error.end = stamp
            self._error.message = message.message

    def finish(self, world: World) -> None:
        """
        Check final silence and close the current command's error episode.

        :param World world: The world at the last message.
        """
        self._gap(world)
        self._close_error()

    def _gap(self, world: World) -> None:
        """
        Report silence since the first command or the latest Planning message, if longer than ``max_gap``.

        :param World world: The world now; the silence runs from ``_last`` to ``world.t``.
        """
        if self._last is not None and world.t - self._last > self.max_gap:
            self.report(world, "planning", f"no planning message for {world.t - self._last:.1f} s "
                                           f"after t={self._last:.1f} s", kind="planning_gap")

    def _close_error(self) -> None:
        """
        Report a qualifying error at its first receive time, then clear it.
        """
        error, self._error = self._error, None
        if error is None:
            return
        duration = error.end - error.start
        if duration >= self.max_error - 1e-6:
            self.violations.append(Violation(
                self.kind, error.received, "planning",
                f"Planning failed command {self._command_id} for {duration:.1f} s: {error.message}",
                *error.position))
