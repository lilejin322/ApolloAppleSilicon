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
_OUTSIDE = {"left": 1.0, "right": -1.0}
"""The side of a curb away from its lane, as ``_side`` signs it: boundaries run in the lane's direction."""


class LaneMarkingOracle(Oracle):
    """
    Report each stretch the ego's body reaches ``min_depth`` past an uncrossable
    line for ``min_duration`` or more, with the deepest reach.

    Uncrossable is ``LaneMarking.prohibits_crossing``: a real solid white,
    solid yellow, double yellow or curb marking; dotted, unknown and virtual
    boundaries never count. The depth is how far the body reaches into the
    far side of the line, alongside the line only: a body passing a line's
    end, as when turning off a lane, does not cross it. A curb's far side is
    the one away from its lane, wherever the ego is: a wide turn on Borregas
    Ave put the reference point itself past one. Any other line's far side is
    the one away from the ego's reference point, since a line two opposite
    lanes share may be crossed either way.
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
    """The ``Violation.kind`` it reports; the subject is the line's lane, side and types."""
    channels = frozenset({LOCALIZATION})
    """The localization messages, at which the ego's body is placed."""
    map: MapService
    """The map Apollo drives."""
    vehicle: ApolloVehicleConfig
    """The ego's dimensions."""
    min_depth: float
    """Metres past the line below which the body is still on its side."""
    min_duration: float
    """Seconds below which a crossing is not reported."""
    _reach: float
    """Metres a strip beside a line must reach to hold any part of the body."""
    _lines: Dict[LaneMarking, LaneMarking]
    """Each marking seen, mapped to the first marking of its painted line."""
    _known: List[LaneMarking]
    """The first marking of each painted line seen, in the order seen."""
    _episodes: Episodes
    """The open crossings, by painted line, each with its deepest reach in metres."""

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig, *, min_depth: float = 0.1,
                 min_duration: float = 0.2) -> None:
        """
        Configure body penetration thresholds in metres and seconds and initialize marking caches.

        :param MapService map_service: The map Apollo drives.
        :param ApolloVehicleConfig vehicle: The ego's dimensions.
        :param float min_depth: Metres past the line below which the body is still on its side.
        :param float min_duration: Seconds below which a crossing is not reported.
        """
        super().__init__()
        self.map = map_service
        self.vehicle = vehicle
        self.min_depth = min_depth
        self.min_duration = min_duration
        self._reach = vehicle.length + vehicle.width
        self._lines = {}
        self._known = []
        self._episodes = Episodes()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Update episodes where the ego body crosses a prohibiting marking beyond the depth threshold.

        :param World world: The world at the localization message.
        :param str channel: ``LOCALIZATION``.
        :param Message message: The localization; the pose is read from ``world.ego``.
        """
        ego = world.ego
        body = ego_footprint(ego.x, ego.y, ego.heading, self.vehicle)
        reference = Point(ego.x, ego.y)
        holding: Dict[LaneMarking, float] = {}
        for marking in self.map.get_lane_markings(body):
            if marking.prohibits_crossing:
                outside = _OUTSIDE[marking.side] if "CURB" in marking.types else None
                depth = _depth(marking.geometry, reference, body, self._reach, outside)
                if depth >= self.min_depth:
                    line = self._line(marking)
                    holding[line] = max(depth, holding.get(line, 0.0))
        self._close(self._episodes.update(world, holding))

    def finish(self, world: World) -> None:
        """
        Report qualifying marking-crossing episodes still open at the end of replay.

        :param World world: The world at the last message; unused.
        """
        self._close(self._episodes.close_all())

    def _line(self, marking: LaneMarking) -> LaneMarking:
        """
        Reuse a canonical marking for geometrically coincident painted lines.

        :param LaneMarking marking: A lane's marking.
        :returns: The first marking seen within ``_SAME_LINE`` (Hausdorff) of it; itself if none.
        :rtype: LaneMarking
        """
        line = self._lines.get(marking)
        if line is None:
            line = next((known for known in self._known
                         if known.geometry.hausdorff_distance(marking.geometry) <= _SAME_LINE), marking)
            if line is marking:
                self._known.append(marking)
            self._lines[marking] = line
        return line

    def _close(self, closed: List[Tuple[LaneMarking, object]]) -> None:
        """
        Report closed marking-crossing episodes meeting the minimum duration.

        :param List[Tuple[LaneMarking, Episode]] closed: Episodes just closed, by painted line (``_line``).
        """
        for line, episode in closed:
            if episode.duration >= self.min_duration:
                types = "+".join(line.types)
                self.report_episode(episode, f"{line.lane_id} {line.side} {types}",
                                    f"body up to {episode.peak:.2f} m past the {types} {line.side} line of "
                                    f"{line.lane_id} (s {line.start_s:.0f}-{line.end_s:.0f}) "
                                    f"for {episode.duration:.1f} s")


def _side(line: LineString, point: Point) -> float:
    """
    Positive left of ``line`` at the point's projection, negative right of it, zero on it.

    :param LineString line: A marking, in its lane's direction.
    :param Point point: The point.
    :returns: A cross product, signed by the side; its size is not a distance.
    :rtype: float
    """
    s = line.project(point)
    a, b = line.interpolate(max(s - 0.05, 0.0)), line.interpolate(min(s + 0.05, line.length))
    return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)


def _depth(line: LineString, reference: Point, body: Polygon, reach: float, outside: float | None = None) -> float:
    """
    How far ``body`` reaches past ``line``, on its ``outside`` (``1.0`` left,
    ``-1.0`` right), or by default on the side away from ``reference``.

    The far side is a strip ``reach`` wide alongside the line, with flat ends,
    so nothing beyond the line's ends counts. A reference point on the line
    has both sides as far sides.

    :param LineString line: The marking, in its lane's direction.
    :param Point reference: The ego's reference point.
    :param Polygon body: The ego's footprint.
    :param float reach: Metres: the strip's width.
    :param float | None outside: The far side, ``1.0`` left or ``-1.0`` right; None to take the side away from
        ``reference``.
    :returns: Metres from the line to the farthest point of the body within the strip; 0 when none is in it.
    :rtype: float
    """
    here = _side(line, reference)
    if outside is not None:
        sides = (outside,)
    else:
        sides = (-1.0, 1.0) if here == 0 else (-copysign(1.0, here),)
    deepest = 0.0
    for side in sides:
        part = body.intersection(line.buffer(side * reach, single_sided=True))
        for piece in getattr(part, "geoms", (part,)):
            if isinstance(piece, Polygon) and not piece.is_empty:
                deepest = max(deepest, max(line.distance(Point(c)) for c in piece.exterior.coords))
    return deepest
