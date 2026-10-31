"""
Collision: the ego's body touches another road user's, and who is to blame.

Blame follows Responsibility-Sensitive Safety (Shalev-Shwartz, Shammah and
Shashua, 2017), which splits a conflict into a longitudinal and a lateral
part. A road user that runs into another is to blame if the other was in its
way early enough to stop; otherwise the other is, for getting into its way too
late, as by cutting in, pulling out or appearing. Side by side, the one that
moved sideways into the other is. The replayed agents do not react, so without
this every agent running into a slow ego from behind would count against the
ego, and the ego cutting in just ahead of a fast agent would not.
"""

from collections import deque
from dataclasses import dataclass
from math import cos, hypot, pi, sin
from typing import Deque, Dict, List, Tuple

from google.protobuf.message import Message
from shapely.geometry import Polygon

from ...utils import ApolloVehicleConfig
from ..footprint import box, ego_footprint
from ..replay import OBSTACLES, Obstacle, World
from .base import Oracle

__all__ = ["CollisionOracle"]

_MOVING = 0.1
"""Metres per second (or metres sideways) up to which a road user counts as not moving toward the other."""
_SIDEWAYS = 2.0
"""Seconds over which sideways movement is measured."""


@dataclass(frozen=True)
class _Pose:
    """One road user at one obstacle message."""

    t: float
    x: float
    y: float
    heading: float
    speed: float
    body: Polygon
    front: float
    """Metres from ``x, y`` ahead to the front edge."""
    width: float


@dataclass(frozen=True)
class _Blame:
    """
    Whether a road user that ran into another is to blame, and why: ``how``
    is ``"all_along"`` (the other was in its way from its first pose on),
    ``"onto"`` (its way came onto the other, as by turning toward it),
    ``"room"`` (the other got into its way with room to stop) or ``"late"``
    (the other got into its way too late: the other is to blame).
    """

    how: str
    since: float
    """When the other got into the striker's way."""
    touch: float
    """When they touched."""
    gap: float
    """Metres the striker had to go then to touch the other."""
    closing: float
    """Metres per second the striker closed on the other then."""
    needed: float
    """Metres the striker needed then to stop short of the other."""

    @property
    def striker_blamed(self) -> bool:
        return self.how != "late"


class CollisionOracle(Oracle):
    """
    Report each obstacle the ego touches, once, as ``collision`` when the ego
    is to blame and ``collision_by_other`` when the obstacle is.

    Checked at every obstacle message, against the latest localization, so
    the obstacles are where perception put them. Blame is settled at the
    first touch, from the last ``history`` seconds. Road users move along
    their headings.

    1. Strikers: a road user that moves and closes on the other along its
       own heading at more than 0.1 m/s runs into it. Within 45 degrees of
       the same or the opposite heading, as in rear-end and head-on
       collisions, it must also close faster than it moves sideways
       relative to the other, so that side by side neither does; at a
       crossing, as at a junction, both do.
    2. A striker's way at a moment is the strip ahead of its front edge,
       ``lookahead`` metres long and as wide as its body, together with the
       ground its body covers from that moment until the touch: the strip
       holds what lies on its line of travel, also coming the other way; the
       covered ground follows curves and turns as driven. Scanning back from
       the touch, the other got into the striker's way where it last was not
       in it. If, where the other just was, it already lay in the striker's
       way then, the way came onto it, as by the striker turning toward it.
       Otherwise the striker had room if the distance it had to go to touch
       the other was at least what it needed to stop: closing speed times
       ``reaction_time``, plus braking at ``braking``.
    3. A striker is to blame if the other was in its way all along, its way
       came onto the other, or it had room; otherwise the other is, for
       getting into its way too late. When each struck and each got into the
       other's way too late, the one that did so first is to blame, as the
       one entering a junction against the light; when each is to blame for
       striking, the ego is.
    4. With no striker, the one that moved more sideways toward the other
       over the last 2 s is to blame, measured across the heading of whichever
       turned less: the lane changer. An obstacle that appears on the ego,
       or comes onto it with neither moving toward the other, is to blame.

    ``reaction_time`` and ``braking`` are the common Responsibility-Sensitive
    Safety values for the response time and the minimum braking a road user
    is expected to apply.

    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param float reaction_time: Seconds a road user takes to start braking.
    :param float braking: Metres per second squared a road user is expected to brake at.
    :param float history: Seconds of poses kept to settle blame.
    :param float lookahead: Metres a road user's way runs ahead of its front edge.
    """

    kind = "collision"
    channels = frozenset({OBSTACLES})

    def __init__(self, vehicle: ApolloVehicleConfig, *, reaction_time: float = 0.5, braking: float = 4.0,
                 history: float = 10.0, lookahead: float = 50.0) -> None:
        super().__init__()
        self.vehicle = vehicle
        self.reaction_time = reaction_time
        self.braking = braking
        self.history = history
        self.lookahead = lookahead
        self._reach = hypot(max(vehicle.front_edge_to_center, vehicle.back_edge_to_center),
                            max(vehicle.left_edge_to_center, vehicle.right_edge_to_center))
        self._frames: Deque[Tuple[_Pose, Dict[int, Obstacle]]] = deque()
        self._hit: set = set()
        """Obstacles touched, blamed on either: only the first touch is judged."""

    def on_message(self, world: World, channel: str, message: Message) -> None:
        ego = world.ego
        if ego is None:
            return
        vehicle = self.vehicle
        pose = _Pose(world.t, ego.x, ego.y, ego.heading, ego.speed,
                     ego_footprint(ego.x, ego.y, ego.heading, vehicle), vehicle.front_edge_to_center,
                     vehicle.left_edge_to_center + vehicle.right_edge_to_center)
        self._frames.append((pose, {obstacle.id: obstacle for obstacle in world.obstacles}))
        while self._frames[0][0].t < world.t - self.history:
            self._frames.popleft()
        for obstacle in world.obstacles:
            if obstacle.id in self._hit:
                continue
            if hypot(obstacle.x - ego.x, obstacle.y - ego.y) > self._reach + hypot(obstacle.length, obstacle.width) / 2:
                continue
            if pose.body.intersects(_obstacle_pose(world.t, obstacle).body):
                self._hit.add(obstacle.id)
                self._judge(world, obstacle)

    def _judge(self, world: World, obstacle: Obstacle) -> None:
        ego, other = self._tracks(obstacle.id)
        name = f"{obstacle.type} {obstacle.id}"
        if len(other) < 2 or other[-2] is None:
            self.report(world, name, f"{name} appeared on the ego", kind="collision_by_other")
            return
        by_ego = self._blame(ego, other) if _strikes(ego[-1], other[-1]) else None
        by_other = self._blame(other, ego) if _strikes(other[-1], ego[-1]) else None
        if by_ego is None and by_other is None:
            self._sideways(world, name, ego, other)
            return
        if by_ego is not None and by_other is not None and not by_ego.striker_blamed \
                and not by_other.striker_blamed:
            # Each got into the other's way too late: the first to do so is to blame.
            ego_blamed = by_other.since <= by_ego.since
        else:
            ego_blamed = (by_ego is not None and by_ego.striker_blamed) or \
                         (by_other is not None and not by_other.striker_blamed)
        if ego_blamed and by_ego is not None and by_ego.striker_blamed:
            self.report(world, name, f"ego at {ego[-1].speed:.2f} m/s ran into {name}, {_why(by_ego)}")
        elif ego_blamed:
            self.report(world, name, f"ego got into the way of {name} at t={by_other.since:.2f} s, "
                                     f"{_room(by_other)}")
        elif by_other is not None and by_other.striker_blamed:
            self.report(world, name, f"{name} at {other[-1].speed:.2f} m/s ran into the ego, {_why(by_other)}",
                        kind="collision_by_other")
        else:
            self.report(world, name, f"{name} got into the ego's way at t={by_ego.since:.2f} s, {_room(by_ego)}",
                        kind="collision_by_other")

    def _sideways(self, world: World, name: str, ego: List[_Pose], other: List[_Pose | None]) -> None:
        ego_moved, other_moved = _moved_toward(ego, other, world.t - _SIDEWAYS)
        if max(ego_moved, other_moved) <= _MOVING:
            self.report(world, name, f"{name} came onto the ego with neither moving toward the other",
                        kind="collision_by_other")
        elif ego_moved >= other_moved:
            self.report(world, name, f"ego moved {_metres(ego_moved)} m sideways into {name} in the last "
                                     f"{_SIDEWAYS:g} s, {name} {_metres(other_moved)} m")
        else:
            self.report(world, name, f"{name} moved {_metres(other_moved)} m sideways into the ego in the "
                                     f"last {_SIDEWAYS:g} s, the ego {_metres(ego_moved)} m",
                        kind="collision_by_other")

    def _tracks(self, obstacle_id: int) -> Tuple[List[_Pose], List[_Pose | None]]:
        """The ego's poses and the obstacle's at the same frames, None where it was not reported."""
        ego, other = [], []
        for pose, obstacles in self._frames:
            ego.append(pose)
            obstacle = obstacles.get(obstacle_id)
            other.append(None if obstacle is None else _obstacle_pose(pose.t, obstacle))
        return ego, other

    def _blame(self, striker: List[_Pose | None], other: List[_Pose | None]) -> _Blame:
        last = len(striker) - 1
        covered, entry = striker[last].body, last
        for i in range(last - 1, -1, -1):
            if striker[i] is None or other[i] is None:
                break
            swept = covered.union(striker[i].body)
            body = other[i].body
            if not (body.intersects(swept) or body.intersects(self._ahead(striker[i]))):
                break
            covered, entry = swept, i
        start, there = striker[entry], other[entry]
        closing = start.speed - there.speed * cos(there.heading - start.heading)
        needed = closing * self.reaction_time + closing * closing / (2 * self.braking) if closing > 0 else 0.0
        touching = next((i for i in range(entry, last + 1) if striker[i].body.intersects(there.body)), None)
        gap = 0.0
        if touching != entry:
            before = last if touching is None else touching - 1
            gap = sum(hypot(striker[i + 1].x - striker[i].x, striker[i + 1].y - striker[i].y)
                      for i in range(entry, before)) + striker[before].body.distance(there.body)
        if entry == 0 or striker[entry - 1] is None:
            how = "all_along"
        elif other[entry - 1] is not None and other[entry - 1].body.intersects(
                covered.union(self._ahead(start))):
            how = "onto"
        else:
            how = "room" if gap >= needed else "late"
        return _Blame(how, there.t, striker[last].t, gap, closing, needed)

    def _ahead(self, pose: _Pose) -> Polygon:
        """The strip ``lookahead`` metres ahead of the front edge, as wide as the body."""
        middle = pose.front + self.lookahead / 2
        return box(pose.x + cos(pose.heading) * middle, pose.y + sin(pose.heading) * middle, pose.heading,
                   self.lookahead, pose.width)


def _obstacle_pose(t: float, obstacle: Obstacle) -> _Pose:
    return _Pose(t, obstacle.x, obstacle.y, obstacle.heading, obstacle.speed,
                 box(obstacle.x, obstacle.y, obstacle.heading, obstacle.length, obstacle.width),
                 obstacle.length / 2, obstacle.width)


def _strikes(pose: _Pose, other: _Pose) -> bool:
    """
    Whether ``pose`` moves and closes on ``other`` along its own heading, and
    unless crossing it, faster than sideways relative to it.
    """
    if pose.speed <= _MOVING:
        return False
    c, s = cos(pose.heading), sin(pose.heading)
    rx = pose.speed * c - other.speed * cos(other.heading)
    ry = pose.speed * s - other.speed * sin(other.heading)
    closing, sideways = rx * c + ry * s, abs(ry * c - rx * s)
    angle = abs((other.heading - pose.heading + pi) % (2 * pi) - pi)
    crossing = pi / 4 <= angle <= 3 * pi / 4
    return closing > _MOVING and (crossing or closing > sideways)


def _moved_toward(a: List[_Pose | None], b: List[_Pose | None], since: float) -> Tuple[float, float]:
    """
    Metres ``a`` and ``b`` each moved toward the other since ``since`` (or
    since it appeared), across the heading of the one that turned less.
    """
    def turned(track):
        return max(abs((pose.heading - track[-1].heading + pi) % (2 * pi) - pi)
                   for pose in track if pose is not None and pose.t >= since)

    def first(track):
        return next(pose for pose in track if pose is not None and pose.t >= since)

    heading = a[-1].heading if turned(a) <= turned(b) else b[-1].heading
    nx, ny = -sin(heading), cos(heading)
    toward = 1.0 if (b[-1].x - a[-1].x) * nx + (b[-1].y - a[-1].y) * ny > 0 else -1.0
    start_a, start_b = first(a), first(b)
    moved_a = ((a[-1].x - start_a.x) * nx + (a[-1].y - start_a.y) * ny) * toward
    moved_b = -((b[-1].x - start_b.x) * nx + (b[-1].y - start_b.y) * ny) * toward
    return moved_a, moved_b


def _metres(value: float) -> str:
    return f"{round(value, 2) + 0.0:.2f}"  # no "-0.00"


def _room(blame: _Blame) -> str:
    return (f"{blame.gap:.1f} m ahead of it closing at {blame.closing:.2f} m/s; "
            f"it needed {blame.needed:.1f} m to stop")


def _why(blame: _Blame) -> str:
    if blame.how == "all_along":
        return f"which had been in its way for {blame.touch - blame.since:.1f} s, as far back as known"
    if blame.how == "onto":
        return f"its way coming onto it at t={blame.since:.2f} s"
    return f"which got into its way at t={blame.since:.2f} s, {_room(blame)}"
