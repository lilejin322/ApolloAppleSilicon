"""
Lane marking: the ego's body crosses a line it may not cross.
"""

from math import copysign
from typing import Dict, List, Tuple

from google.protobuf.message import Message
from shapely.geometry import LineString, Point, Polygon

from ...map import MapService
from ...map.types import LaneMarking
from ...utils import ApolloVehicleConfig
from ..footprint import ego_footprint
from ..replay import LOCALIZATION, World
from .base import Episodes, Oracle

__all__ = ["LaneMarkingOracle"]

_SAME_LINE = 0.3
"""Metres (Hausdorff) within which two lanes' markings are one painted line."""


class LaneMarkingOracle(Oracle):
    """
    Report each stretch the ego's body reaches ``min_depth`` past an uncrossable
    line for ``min_duration`` or more, with the deepest reach.

    Uncrossable is ``LaneMarking.prohibits_crossing``: a real solid white,
    solid yellow, double yellow or curb marking; dotted, unknown and virtual
    boundaries never count. The depth is how far the body reaches into the
    far side of the line, seen from the ego's reference point, alongside the
    line only: a body passing a line's end, as when turning off a lane, does
    not cross it.
    Apollo plans with lane widths rather than the boundary curves, and a body
    has round corners, so a normal drive hugging a curb reaches a few
    centimetres past it (3.2 cm on Borregas Ave); ``min_depth`` keeps that out.
    A line two adjacent lanes share is reported once.

    :param MapService map_service: The map Apollo drives.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param float min_depth: Metres past the line below which the body is still on its side.
    :param float min_duration: Seconds below which a crossing is not reported.
    """

    kind = "lane_marking"
    channels = frozenset({LOCALIZATION})

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig, *, min_depth: float = 0.1,
                 min_duration: float = 0.2) -> None:
        super().__init__()
        self.map = map_service
        self.vehicle = vehicle
        self.min_depth = min_depth
        self.min_duration = min_duration
        self._reach = vehicle.length + vehicle.width
        """Metres a strip beside a line must reach to hold any part of the body."""
        self._lines: Dict[LaneMarking, LaneMarking] = {}
        """Each marking seen, mapped to the first marking of its painted line."""
        self._known: List[LaneMarking] = []
        self._episodes = Episodes()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        ego = world.ego
        body = ego_footprint(ego.x, ego.y, ego.heading, self.vehicle)
        reference = Point(ego.x, ego.y)
        holding: Dict[LaneMarking, float] = {}
        for marking in self.map.get_lane_markings(body):
            if marking.prohibits_crossing:
                depth = _depth(marking.geometry, reference, body, self._reach)
                if depth >= self.min_depth:
                    line = self._line(marking)
                    holding[line] = max(depth, holding.get(line, 0.0))
        self._close(self._episodes.update(world, holding))

    def finish(self, world: World) -> None:
        self._close(self._episodes.close_all())

    def _line(self, marking: LaneMarking) -> LaneMarking:
        line = self._lines.get(marking)
        if line is None:
            line = next((known for known in self._known
                         if known.geometry.hausdorff_distance(marking.geometry) <= _SAME_LINE), marking)
            if line is marking:
                self._known.append(marking)
            self._lines[marking] = line
        return line

    def _close(self, closed: List[Tuple[LaneMarking, object]]) -> None:
        for line, episode in closed:
            if episode.duration >= self.min_duration:
                types = "+".join(line.types)
                self.report_episode(episode, f"{line.lane_id} {line.side} {types}",
                                    f"body up to {episode.peak:.2f} m past the {types} {line.side} line of "
                                    f"{line.lane_id} (s {line.start_s:.0f}-{line.end_s:.0f}) "
                                    f"for {episode.duration:.1f} s")


def _side(line: LineString, point: Point) -> float:
    """Positive left of ``line`` at the point's projection, negative right of it, zero on it."""
    s = line.project(point)
    a, b = line.interpolate(max(s - 0.05, 0.0)), line.interpolate(min(s + 0.05, line.length))
    return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)


def _depth(line: LineString, reference: Point, body: Polygon, reach: float) -> float:
    """
    How far ``body`` reaches past ``line``, on the side away from ``reference``.

    The far side is a strip ``reach`` wide alongside the line, with flat ends,
    so nothing beyond the line's ends counts. A reference point on the line
    has both sides as far sides.
    """
    here = _side(line, reference)
    sides = (-1.0, 1.0) if here == 0 else (-copysign(1.0, here),)
    deepest = 0.0
    for side in sides:
        part = body.intersection(line.buffer(side * reach, single_sided=True))
        for piece in getattr(part, "geoms", (part,)):
            if isinstance(piece, Polygon) and not piece.is_empty:
                deepest = max(deepest, max(line.distance(Point(c)) for c in piece.exterior.coords))
    return deepest
