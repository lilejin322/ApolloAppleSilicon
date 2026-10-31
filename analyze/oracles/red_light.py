"""
Red light: the ego's front crosses the stop line of a signal that shows red.
"""

from typing import List, Tuple

from google.protobuf.message import Message
from shapely.geometry import LineString

from ...map import MapService
from ...utils import ApolloVehicleConfig
from ..footprint import front_point
from ..replay import LOCALIZATION, PLANNING_COMMAND, World
from .base import Oracle

__all__ = ["RedLightOracle"]


class RedLightOracle(Oracle):
    """
    Report each crossing of a stop line on red, once per signal: the red
    signals of one stop line are one violation, its subject their IDs joined
    by ``+`` (on Borregas Ave four signal heads share each stop line).

    The stop lines are those of the signals controlling the lanes of the route
    Apollo planned (``/apollo/planning/command``), each matched to its lane by
    ``MapService.get_signal_stop_lines``, so a stop line of another approach
    never counts. The crossing is the ego's front-edge midpoint moving across
    the line between two localization messages; the colour is the latest
    ``/apollo/perception/traffic_light`` report. Crossing on yellow is allowed,
    as in most jurisdictions; a signal not reported is not judged.

    :param MapService map_service: The map Apollo drives.
    :param ApolloVehicleConfig vehicle: ``front_edge_to_center`` places the front.
    """

    kind = "red_light"
    channels = frozenset({LOCALIZATION, PLANNING_COMMAND})

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig) -> None:
        super().__init__()
        self.map = map_service
        self.vehicle = vehicle
        self._route: Tuple[str, ...] = ()
        self._stop_lines: List[Tuple[LineString, str, List[str]]] = []
        """Each stop line of the route, its lane and the signals it stops for."""
        self._front: Tuple[float, float] | None = None
        self._run: set = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        if channel == PLANNING_COMMAND:
            if world.route != self._route:
                self._route = world.route
                self._stop_lines = []
                for lane in self._route:
                    if lane not in self.map.lane_table:
                        continue
                    for signal in self.map.get_signals_for_lane(lane):
                        for stop in self.map.get_signal_stop_lines(signal, lane):
                            same = next((signals for line, at, signals in self._stop_lines
                                         if at == lane and line.equals(stop.geometry)), None)
                            if same is None:
                                self._stop_lines.append((stop.geometry, lane, [signal]))
                            elif signal not in same:
                                same.append(signal)
            return
        ego = world.ego
        front = front_point(ego.x, ego.y, ego.heading, self.vehicle)
        previous, self._front = self._front, front
        if previous is None or previous == front or not self._stop_lines:
            return
        step = LineString([previous, front])
        for line, lane, signals in self._stop_lines:
            red = sorted(signal for signal in signals
                         if signal not in self._run and world.lights.get(signal) == "RED")
            if not red or not step.intersects(line):
                continue
            self._run.update(red)
            subject = "+".join(red)
            self.report(world, subject, f"front crossed the stop line of {subject} on red, entering past {lane} "
                                        f"at {ego.speed:.2f} m/s")
