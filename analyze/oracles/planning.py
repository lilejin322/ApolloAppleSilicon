"""
Planning's own verdicts: emergency stops, lane changes in junctions, and a
Planning that fails or falls silent.
"""

from google.protobuf.message import Message
from shapely.geometry import Point

from ...map import MapService
from ...map.geometry import polygon_to_shape
from ..replay import COMMAND_STATUS, PLANNING, World
from .base import Episodes, Oracle

__all__ = ["EStopOracle", "JunctionLaneChangeOracle", "PlanningFailureOracle"]


def _enum_name(message: Message, field: str) -> str:
    return message.DESCRIPTOR.fields_by_name[field].enum_type.values_by_number[getattr(message, field)].name


class EStopOracle(Oracle):
    """
    Report each reason for which Planning decided an emergency stop, once.

    The decision is ``ADCTrajectory.decision.main_decision.estop``.
    """

    kind = "estop"
    channels = frozenset({PLANNING})

    def __init__(self) -> None:
        super().__init__()
        self._reasons: set = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        decision = message.decision.main_decision
        if not decision.HasField("estop"):
            return
        reason = _enum_name(decision.estop, "reason_code")
        if reason not in self._reasons:
            self._reasons.add(reason)
            self.report(world, reason, f"emergency stop: {decision.estop.reason or reason}")


class JunctionLaneChangeOracle(Oracle):
    """
    Report each junction in which Planning decided to change lanes, once.

    The decision is the ``change_lane_type`` of the main cruise or stop
    decision; the ego is in a junction when its reference point lies in the
    junction's polygon.

    :param MapService map_service: The map Apollo drives.
    """

    kind = "junction_lane_change"
    channels = frozenset({PLANNING})

    def __init__(self, map_service: MapService) -> None:
        super().__init__()
        self._areas = [(junction_id, area) for junction_id, junction in map_service.junction_table.items()
                       if (area := polygon_to_shape(junction.polygon)) is not None]
        self._junctions: set = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
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


class PlanningFailureOracle(Oracle):
    """
    Report Planning falling silent or failing the ego's command for too long.

    ``planning_gap``: no ``/apollo/planning`` message for more than ``max_gap``
    seconds after the first one, up to the end of the record. External Command
    runs in Planning's process, and planning a session's first route held
    Planning for 0.91 s on Apple Silicon; other gaps reached 0.63 s.
    ``planning_error``: Planning reports ``ERROR`` for the current command for
    ``max_error`` seconds or more; a single failed cycle, as right after a new
    command arrives, is not reported.

    :param float max_gap: Seconds between planning messages still normal.
    :param float max_error: Seconds of ``ERROR`` status still tolerated.
    """

    kind = "planning_error"
    channels = frozenset({PLANNING, COMMAND_STATUS})

    def __init__(self, *, max_gap: float = 2.0, max_error: float = 1.0) -> None:
        super().__init__()
        self.max_gap = max_gap
        self.max_error = max_error
        self._last: float | None = None
        self._errors = Episodes()
        self._message = ""

    def on_message(self, world: World, channel: str, message: Message) -> None:
        if channel == PLANNING:
            self._gap(world)
            self._last = world.t
            return
        failing = (world.command is not None and message.command_id == world.command.command_id
                   and _enum_name(message, "status") == "ERROR")
        if failing:
            self._message = message.message
        self._close(self._errors.update(world, {"error": 1.0} if failing else {}))

    def finish(self, world: World) -> None:
        self._gap(world)
        self._close(self._errors.close_all())

    def _gap(self, world: World) -> None:
        if self._last is not None and world.t - self._last > self.max_gap:
            self.report(world, "planning", f"no planning message for {world.t - self._last:.1f} s "
                                           f"after t={self._last:.1f} s", kind="planning_gap")

    def _close(self, closed) -> None:
        for _, episode in closed:
            if episode.duration >= self.max_error:
                self.report_episode(episode, "planning", f"Planning failed the command for {episode.duration:.1f} s: "
                                                         f"{self._message}")
