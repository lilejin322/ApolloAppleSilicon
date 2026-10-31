"""
Comfort: the ego accelerates or brakes harder than passengers accept.
"""

from google.protobuf.message import Message

from ..replay import LOCALIZATION, World
from .base import Episodes, Oracle

__all__ = ["ComfortOracle"]

_TOLERANCE = 1e-3
"""Metres per second squared past a limit still within it: Planning brakes at exactly -4.0, which Sim
Control's perfect control reproduces with float error."""


class ComfortOracle(Oracle):
    """
    Report each stretch of harsh longitudinal acceleration or braking, with its peak.

    The acceleration is the localization's, along the ego's heading. A value
    a rounding error past a limit is within it: Planning's comfortable
    braking is exactly the default limit, and runs on Borregas Ave braked at
    -4.000000 m/s^2 but never below -4.001.

    :param float max_acceleration: Metres per second squared allowed when speeding up.
    :param float max_deceleration: Metres per second squared allowed when braking, positive.
    """

    kind = "comfort"
    channels = frozenset({LOCALIZATION})

    def __init__(self, *, max_acceleration: float = 4.0, max_deceleration: float = 4.0) -> None:
        super().__init__()
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self._episodes = Episodes()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        a = world.ego.acceleration
        holding = {}
        if a > self.max_acceleration + _TOLERANCE:
            holding["acceleration"] = a
        elif a < -self.max_deceleration - _TOLERANCE:
            holding["braking"] = -a
        self._close(self._episodes.update(world, holding))

    def finish(self, world: World) -> None:
        self._close(self._episodes.close_all())

    def _close(self, closed) -> None:
        for what, episode in closed:
            limit = self.max_acceleration if what == "acceleration" else self.max_deceleration
            self.report_episode(episode, what, f"harsh {what} up to {episode.peak:.2f} m/s^2 "
                                               f"(limit {limit:.1f}) for {episode.duration:.1f} s")
