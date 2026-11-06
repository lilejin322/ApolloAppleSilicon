"""
Replay a record: its messages in time order, and the world they describe.

The record's receive times are the one clock every channel shares; headers
are on the wall clock too (our managers stamp the run's start plus scenario
time), but each sender stamps them its own way. ``Replay`` sorts the messages
by receive time and keeps a ``World``, the latest state of the ego, the
obstacles, the lights and Planning, which every oracle reads instead of
keeping its own copies.
"""

from dataclasses import dataclass, field
from math import cos, hypot, isfinite, sin
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Self
from google.protobuf.message import Message

from ..manager.routing import planned_lanes
from .record import RecordReader

if TYPE_CHECKING:  # ``oracles.base`` imports ``World`` from here: only the type checker may import it back.
    from .oracles.base import Oracle, Violation

__all__ = ["COMMAND_STATUS", "EgoState", "LOCALIZATION", "OBSTACLES", "Obstacle", "PLANNING", "PLANNING_COMMAND",
           "PREDICTION", "Replay", "TRAFFIC_LIGHT", "World"]

LOCALIZATION = "/apollo/localization/pose"
"""The ego's pose, velocity and acceleration (``LocalizationEstimate``)."""
OBSTACLES = "/apollo/perception/obstacles"
"""Every road user around the ego (``PerceptionObstacles``); our managers publish it."""
TRAFFIC_LIGHT = "/apollo/perception/traffic_light"
"""The colour of each signal (``TrafficLightDetection``); our managers publish it."""
PLANNING = "/apollo/planning"
"""Planning's trajectory, decision and scenario (``ADCTrajectory``)."""
PREDICTION = "/apollo/prediction"
"""Prediction's trajectories for each obstacle (``PredictionObstacles``)."""
PLANNING_COMMAND = "/apollo/planning/command"
"""The route Planning drives (``PlanningCommand``)."""
COMMAND_STATUS = "/apollo/planning/command_status"
"""Whether Planning is carrying out its command (``CommandStatus``)."""
_WORLD_CHANNELS = (LOCALIZATION, OBSTACLES, TRAFFIC_LIGHT, PLANNING, PLANNING_COMMAND, COMMAND_STATUS)
"""The channels ``World`` keeps the latest message of, whether or not an oracle listens to them."""


@dataclass(frozen=True)
class EgoState:
    """
    The ego at one localization message.

    Apollo's vector fields default to NaN; a velocity or acceleration
    component a publisher leaves out counts as 0, so a missing acceleration
    reads as none rather than making every comparison false.

    :param float x: Reference point, map frame.
    :param float y: Reference point, map frame.
    :param float heading: Radians.
    :param float speed: Metres per second.
    :param float acceleration: Longitudinal, metres per second squared; negative when braking.
    """
    x: float
    """Reference point, map frame."""
    y: float
    """Reference point, map frame."""
    heading: float
    """Radians."""
    speed: float
    """Metres per second."""
    acceleration: float
    """Longitudinal, metres per second squared; negative when braking."""

    @classmethod
    def of(cls, localization: Message) -> Self:
        """
        Read a ``LocalizationEstimate``.

        :param Message localization: The message.
        :returns: The ego's state, its speed the size of its velocity and its acceleration the part along its
            heading.
        :rtype: EgoState
        """
        pose = localization.pose
        heading = pose.heading
        vx, vy = _set(pose.linear_velocity.x), _set(pose.linear_velocity.y)
        ax, ay = _set(pose.linear_acceleration.x), _set(pose.linear_acceleration.y)
        return cls(pose.position.x, pose.position.y, heading, hypot(vx, vy), ax * cos(heading) + ay * sin(heading))


def _set(value: float) -> float:
    """
    Replace a non-finite recorded scalar with zero; preserve finite values.

    :param float value: A recorded component, NaN where the publisher left it out.
    :returns: ``value``, or 0 when it is NaN or infinite.
    :rtype: float
    """
    return value if isfinite(value) else 0.0


@dataclass(frozen=True)
class Obstacle:
    """
    One perception obstacle.

    :param int id: Its ID.
    :param str type: ``PerceptionObstacle.Type`` name, e.g. ``"VEHICLE"``.
    :param float x: Centre, map frame.
    :param float y: Centre, map frame.
    :param float heading: Radians.
    :param float length: Along the heading.
    :param float width: Across it.
    :param float speed: Metres per second.
    """
    id: int
    """Its ID."""
    type: str
    """``PerceptionObstacle.Type`` name, e.g. ``"VEHICLE"``."""
    x: float
    """Centre, map frame."""
    y: float
    """Centre, map frame."""
    heading: float
    """Radians."""
    length: float
    """Along the heading."""
    width: float
    """Across it."""
    speed: float
    """Metres per second."""

    @classmethod
    def all_of(cls, obstacles: Message) -> Tuple["Obstacle", ...]:
        """
        Read every obstacle of a ``PerceptionObstacles``.

        :param Message obstacles: The message.
        :returns: The obstacles, in the message's order.
        :rtype: Tuple[Obstacle, ...]
        """
        result = []
        for item in obstacles.perception_obstacle:
            type_name = item.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[item.type].name
            result.append(cls(item.id, type_name, item.position.x, item.position.y, item.theta,
                              item.length, item.width, hypot(_set(item.velocity.x), _set(item.velocity.y))))
        return tuple(result)


@dataclass
class World:
    """
    The latest state of every channel the oracles read, at time ``t``.

    :param float t: Seconds since the record began.
    :param float begin: When the record began, in seconds since the epoch; 0 when unknown.
    :param EgoState | None ego: The ego at the latest localization message; None before the first.
    :param Tuple[Obstacle, ...] obstacles: The obstacles of the latest frame; none before the first.
    :param Dict[str, str] lights: Colour name of every reported signal, e.g. ``{"signal_9": "RED"}``.
    :param Message | None planning: The latest ``ADCTrajectory``.
    :param Message | None command: The latest ``PlanningCommand`` created since the record began.
    :param Tuple[str, ...] route: The lanes of ``command`` in driving order.
    :param Message | None command_status: The latest ``CommandStatus`` Planning published.
    """
    t: float = 0.0
    """Seconds since the record began."""
    begin: float = 0.0
    """When the record began, in seconds since the epoch; 0 when unknown."""
    ego: EgoState | None = None
    """The ego at the latest localization message; None before the first."""
    obstacles: Tuple[Obstacle, ...] = ()
    """The obstacles of the latest frame; none before the first."""
    lights: Dict[str, str] = field(default_factory=dict)
    """Colour name of every reported signal, e.g. ``{"signal_9": "RED"}``."""
    planning: Message | None = None
    """The latest ``ADCTrajectory``."""
    command: Message | None = None
    """
    The latest ``PlanningCommand`` created since the record began: the route
    Apollo planned. Cyber hands a new subscriber the last message a channel
    cached, so a record started between scenarios opens with the previous
    scenario's command; it is ignored. A record started while the ego already
    drives therefore has no route until the next command.
    """
    route: Tuple[str, ...] = ()
    """The lanes of ``command`` in driving order."""
    command_status: Message | None = None
    """The latest ``CommandStatus`` Planning published."""

    def update(self, channel: str, message: Message) -> None:
        """
        Take one message of a world channel; other channels change nothing.

        :param str channel: Its channel.
        :param Message message: The message.
        """
        if channel == LOCALIZATION:
            self.ego = EgoState.of(message)
        elif channel == OBSTACLES:
            self.obstacles = Obstacle.all_of(message)
        elif channel == TRAFFIC_LIGHT:
            names = message.DESCRIPTOR.fields_by_name["traffic_light"].message_type.fields_by_name["color"].enum_type
            self.lights = {light.id: names.values_by_number[light.color].name for light in message.traffic_light}
        elif channel == PLANNING:
            self.planning = message
        elif channel == PLANNING_COMMAND:
            if message.header.timestamp_sec >= self.begin:
                self.command = message
                self.route = planned_lanes(message)
        elif channel == COMMAND_STATUS:
            self.command_status = message


class Replay:
    """
    Play a record's messages, in receive-time order, to oracles.

    Every message first updates the ``World``; then each oracle that listens
    to its channel sees the world. When the record ends each oracle finishes,
    closing what it still has open.

    :param RecordReader reader: The record.
    """
    reader: RecordReader
    """The record."""

    def __init__(self, reader: RecordReader) -> None:
        """
        Bind the record reader used to reconstruct successive world snapshots.

        :param RecordReader reader: The record.
        """
        self.reader = reader

    def run(self, oracles: Sequence["Oracle"]) -> List["Violation"]:
        """
        Replay the record to ``oracles`` and collect their violations.

        :param Sequence[Oracle] oracles: Fresh oracles; each is used for one record only.
        :returns: Every violation, in time order.
        :rtype: List[Violation]
        """
        listeners: Dict[str, List["Oracle"]] = {}
        for oracle in oracles:
            for channel in oracle.channels:
                listeners.setdefault(channel, []).append(oracle)
        wanted = set(_WORLD_CHANNELS) | set(listeners)
        messages = sorted(self.reader.messages(wanted), key=lambda m: m.time)
        start = self.reader.header.begin_time or (messages[0].time if messages else 0)
        world = World(begin=start / 1e9)
        for message in messages:
            world.t = (message.time - start) / 1e9
            world.update(message.channel, message.message)
            for oracle in listeners.get(message.channel, ()):
                oracle.on_message(world, message.channel, message.message)
        for oracle in oracles:
            oracle.finish(world)
        return sorted((v for oracle in oracles for v in oracle.violations), key=lambda v: v.t)
