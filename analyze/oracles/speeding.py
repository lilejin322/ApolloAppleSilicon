"""
Speeding: the ego drives faster than its lane's speed limit.
"""

from typing import Hashable, List, Tuple
from google.protobuf.message import Message
from shapely.geometry import LineString, Point

from ...map import MapService
from ..replay import LOCALIZATION, PLANNING_COMMAND, World
from .base import Episode, Episodes, Oracle

__all__ = ["SpeedingOracle"]


class SpeedingOracle(Oracle):
    """
    Report each stretch the ego drives above its lane's limit by more than ``tolerance``.

    The ego's lane is the nearest lane of the route Apollo planned, so the
    lanes crossing a junction never count; a position farther than
    ``max_offset`` from every route lane, or a lane without a limit, is not
    judged. One violation per stretch, at its start, with the top speed.

    :param MapService map_service: The map Apollo drives.
    :param float tolerance: Fraction above the limit that is allowed.
    :param float max_offset: Metres from a route lane's centerline still on it.
    """
    kind = "speeding"
    """The ``Violation.kind`` it reports; the subject is the lane."""
    channels = frozenset({LOCALIZATION, PLANNING_COMMAND})
    """The localization messages, which carry the ego's speed, and the commands, which carry the route."""
    map: MapService
    """The map Apollo drives."""
    tolerance: float
    """Fraction above the limit that is allowed."""
    max_offset: float
    """Metres from a route lane's centerline still on it."""
    _route: Tuple[str, ...]
    """The lanes of the route the limits were collected for."""
    _lanes: List[Tuple[str, LineString, float]]
    """Each route lane on the map: its ID, centerline and speed limit in m/s (0 without one)."""
    _episodes: Episodes
    """The open stretches of speeding, by ``(lane, limit)``, each with its top speed in m/s."""

    def __init__(self, map_service: MapService, *, tolerance: float = 0.05, max_offset: float = 3.0) -> None:
        """
        Configure the relative speed tolerance and maximum lane offset in metres.

        :param MapService map_service: The map Apollo drives.
        :param float tolerance: Fraction above the limit that is allowed.
        :param float max_offset: Metres from a route lane's centerline still on it.
        """
        super().__init__()
        self.map = map_service
        self.tolerance = tolerance
        self.max_offset = max_offset
        self._route = ()
        self._lanes = []
        self._episodes = Episodes()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Refresh route speed limits or update speeding episodes on the nearest eligible route lane.

        :param World world: The world at the message; ``world.route`` and ``world.ego`` are read from it.
        :param str channel: ``PLANNING_COMMAND`` (a new route) or ``LOCALIZATION`` (a new pose and speed).
        :param Message message: The ``PlanningCommand`` or the localization; both are read through ``world``.
        """
        if channel == PLANNING_COMMAND:
            if world.route != self._route:
                self._route = world.route
                self._lanes = [(lane, self.map.get_lane_central_curve_by_id(lane),
                                self.map.get_lane_by_id(lane).speed_limit)
                               for lane in self._route if lane in self.map.lane_table]
            return
        holding = {}
        ego = world.ego
        if self._lanes:
            point = Point(ego.x, ego.y)
            offset, lane, limit = min((line.distance(point), lane, limit) for lane, line, limit in self._lanes)
            if offset <= self.max_offset and limit > 0 and ego.speed > limit * (1 + self.tolerance):
                holding[(lane, limit)] = ego.speed
        self._close(self._episodes.update(world, holding))

    def finish(self, world: World) -> None:
        """
        Report any speeding episodes still open at the end of replay.

        :param World world: The world at the last message; unused.
        """
        self._close(self._episodes.close_all())

    def _close(self, closed: List[Tuple[Hashable, Episode]]) -> None:
        """
        Report each closed speeding episode with its peak speed, lane limit and duration.

        :param List[Tuple[Hashable, Episode]] closed: Episodes just closed (``Episodes.update`` or ``close_all``),
            keyed ``(lane, limit)``.
        """
        for (lane, limit), episode in closed:
            self.report_episode(episode, lane, f"{episode.peak:.2f} m/s on {lane}, limit {limit:.2f} m/s "
                                               f"(+{episode.peak / limit - 1:.0%}) for {episode.duration:.1f} s")
