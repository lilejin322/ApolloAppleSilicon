"""
What an oracle is and what it reports, abstract class's definition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, FrozenSet, Hashable, List, Mapping, Tuple
from google.protobuf.message import Message

from ..replay import World

__all__ = ["Episode", "Episodes", "Oracle", "Violation"]


@dataclass(frozen=True)
class Violation:
    """
    One thing the ADS did wrong.

    :param str kind: The oracle's verdict, e.g. ``"collision"`` or ``"red_light"``.
    :param float t: Seconds since the record began, when it started.
    :param str subject: What was violated or hit, e.g. ``"VEHICLE 1003"`` or ``"signal_9"``.
    :param str detail: What happened, for people.
    :param float | None x: The ego at ``t``, map frame: its reference point, or its body centre for the RSS
        collision kinds; None without an ego state.
    :param float | None y: The same, its Y coordinate.
    """
    kind: str
    """The oracle's verdict, e.g. ``"collision"`` or ``"red_light"``."""
    t: float
    """Seconds since the record began, when it started."""
    subject: str
    """What was violated or hit, e.g. ``"VEHICLE 1003"`` or ``"signal_9"``."""
    detail: str
    """What happened, for people."""
    x: float | None = None
    """
    The ego at ``t``, map frame: its reference point, or its body centre for the RSS collision kinds; None
    without an ego state.
    """
    y: float | None = None
    """The same, its Y coordinate."""


class Oracle(ABC):
    """
    Judge one kind of violation while a record is replayed.

    ``Replay`` calls ``on_message`` for every message of ``channels`` after
    updating the world, then ``finish`` once at the end. An oracle judges one
    record only.

    A subclass sets ``channels``, and ``kind`` if it reports, and implements
    ``on_message``; ``finish`` does nothing unless it overrides it.
    """
    kind: ClassVar[str]
    """The ``Violation.kind`` it reports."""
    channels: ClassVar[FrozenSet[str]]
    """The channels it listens to."""
    violations: List[Violation]
    """What it reported, in the order reported."""

    def __init__(self) -> None:
        """
        Initialize an empty collection of reported violations.
        """
        self.violations = []

    @abstractmethod
    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        See one message, after it updated ``world``.

        :param World world: The world at the message.
        :param str channel: Its channel, one of ``channels``.
        :param Message message: The message.
        """

    def finish(self, world: World) -> None:
        """
        The record has ended; report what is still open.

        :param World world: The world at the last message.
        """

    def report(self, world: World, subject: str, detail: str, *, kind: str | None = None) -> None:
        """
        Record a violation now, at the ego's current position.

        :param World world: The world now.
        :param str subject: What was violated.
        :param str detail: What happened.
        :param str | None kind: ``kind`` by default.
        """
        ego = world.ego
        self.violations.append(Violation(kind or self.kind, world.t, subject, detail,
                                         None if ego is None else ego.x, None if ego is None else ego.y))

    def report_episode(self, episode: "Episode", subject: str, detail: str, *, kind: str | None = None) -> None:
        """
        Record a violation that lasted a while, at its start.

        :param Episode episode: When and where it started.
        :param str subject: What was violated.
        :param str detail: What happened.
        :param str | None kind: ``kind`` by default.
        """
        self.violations.append(Violation(kind or self.kind, episode.start, subject, detail, *episode.position))


@dataclass
class Episode:
    """
    One stretch of time during which a condition held.

    :param float start: Seconds since the record began: the first time it held.
    :param float end: Seconds since the record began: the last time it held.
    :param Tuple[float | None, float | None] position: The ego's reference point ``(x, y)`` at ``start``, map
        frame; ``(None, None)`` without an ego state.
    :param float peak: The largest value seen, by the oracle's measure.
    """
    start: float
    """Seconds since the record began: the first time it held."""
    end: float
    """Seconds since the record began: the last time it held."""
    position: Tuple[float | None, float | None]
    """The ego's reference point ``(x, y)`` at ``start``, map frame; ``(None, None)`` without an ego state."""
    peak: float
    """The largest value seen, by the oracle's measure."""

    @property
    def duration(self) -> float:
        """
        Seconds from ``start`` to ``end``.

        :returns: Seconds; 0 for an episode seen at one time only.
        :rtype: float
        """
        return self.end - self.start


class Episodes:
    """
    Track stretches of time, by key, during which a condition holds.

    ``update`` takes the keys that hold now with a value each; a key that
    stops holding closes its episode, which is returned.
    """
    open: Dict[Hashable, Episode]
    """The episodes still open, by key."""

    def __init__(self) -> None:
        """
        Initialize an empty mapping of active episodes by subject key.
        """
        self.open = {}

    def update(self, world: World, holding: Mapping[Hashable, float]) -> List[Tuple[Hashable, Episode]]:
        """
        Advance to ``world.t``.

        :param World world: The world now.
        :param Mapping[Hashable, float] holding: Each key that holds now, with its value.
        :returns: The episodes that just ended, with their keys; each ends at the last time it held.
        :rtype: List[Tuple[Hashable, Episode]]
        """
        ego = world.ego
        here = (None, None) if ego is None else (ego.x, ego.y)
        for key, value in holding.items():
            episode = self.open.get(key)
            if episode is None:
                self.open[key] = Episode(world.t, world.t, here, value)
            else:
                episode.end = world.t
                episode.peak = max(episode.peak, value)
        closed = [(key, episode) for key, episode in self.open.items() if key not in holding]
        for key, _ in closed:
            del self.open[key]
        return closed

    def close_all(self) -> List[Tuple[Hashable, Episode]]:
        """
        End every open episode, as at the end of the record.

        :returns: The episodes that were open, with their keys.
        :rtype: List[Tuple[Hashable, Episode]]
        """
        closed, self.open = list(self.open.items()), {}
        return closed
