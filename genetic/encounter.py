"""
Where and when the ego will be, roughly, so that new agents can be drawn to
meet it, on any map: agents drawn at random spread over the whole map and
minute and seldom meet the ego (on borregas_ave it had one road user within
30 m on average, and none for 40 % of its driving time).

The ego's position is known only within an envelope: Apollo moves off about
1.4 s after the ego's ``start_t`` (1.2-1.9 s), then waits at lights and yields
as the traffic makes it, so 10 to 40 s after moving off it had covered between
0.7 and 7.5 m a second of its route. These are the 10th and 90th percentiles
of the first 30 records of the 2026-11-02 borregas_ave search, the only map
measured; other maps use the same figures. An agent's encounter score is how
near it comes, at any sample time, to the stretch of the ego's route the ego
may be on then (``EgoEnvelope.range``): its distance from the route plus how
far along the route it lies outside that stretch. Lower meets the ego better.
"""

from dataclasses import dataclass
from math import inf
from typing import Tuple
import numpy as np
from shapely.geometry import LineString, Point

from ..scenario import ADAgent, RouteSpace
from ..scenario.sections import Body

__all__ = ["EgoEnvelope", "MOVE_OFF", "REACH", "SLOW", "FAST", "SLACK", "encounter_score"]

MOVE_OFF = 1.4
"""Seconds from the ego's ``start_t`` until it moves."""
SLOW = 0.7
"""m/s of route the slowest egos covered on average once moving."""
FAST = 7.5
"""m/s of route the fastest egos covered on average once moving."""
SLACK = 15.0
"""Metres along the route either side of the envelope still counted as meeting the ego."""
REACH = 60.0
"""Metres from the route beyond which a sample is not scored."""
STEP = 0.5
"""Seconds between the sample times scored."""


@dataclass(frozen=True)
class EgoEnvelope:
    """
    The stretch of its route the ego may be on at each time.

    :param LineString path: The ego's route, as it drives it.
    :param float start_t: When the ego may move, its gene's ``start_t``.
    """
    path: LineString
    """The ego's route, as it drives it."""
    start_t: float
    """When the ego may move, its gene's ``start_t``."""

    @classmethod
    def of(cls, ego: ADAgent, ego_space: RouteSpace) -> "EgoEnvelope":
        """
        The envelope of ``ego`` on its route space.

        :param ADAgent ego: The ego gene: its route and ``start_t``.
        :param RouteSpace ego_space: The ego's route space (``ScenarioContext.ego_space``), which lays out the route.
        :returns: The envelope along the route ``ego_space`` builds for the ego.
        :rtype: EgoEnvelope
        """
        return cls(ego_space.path(ego.route), ego.start_t)

    def range(self, t: float) -> Tuple[float, float]:
        """
        Metres along the route between which the ego may be at ``t``.

        :param float t: Scenario time in seconds.
        :returns: ``(nearest, farthest)`` route metres: ``SLOW`` and ``FAST`` times the seconds since the ego moved
            off (``start_t + MOVE_OFF``), both capped at the route's length; ``(0, 0)`` before it moves.
        :rtype: Tuple[float, float]
        """
        moving = max(0.0, t - self.start_t - MOVE_OFF)
        return min(SLOW * moving, self.path.length), min(FAST * moving, self.path.length)


def encounter_score(body: Body, times: np.ndarray, envelope: EgoEnvelope) -> float:
    """
    How near ``body`` comes to where the ego may be: at the best sample time,
    metres from the route plus metres along it outside the envelope (``SLACK``
    excepted). ``inf`` when it never comes within ``REACH`` of the route.

    :param Body body: The agent's sampled poses, on ``times``.
    :param np.ndarray times: The sample times of ``body``.
    :param EgoEnvelope envelope: Where the ego may be.
    :rtype: float
    """
    every = max(1, int(round(STEP / (times[1] - times[0])))) if len(times) > 1 else 1
    best = inf
    for i in range(0, len(times), every):
        point = Point(float(body.x[i]), float(body.y[i]))
        off = envelope.path.distance(point)
        if off > REACH:
            continue
        low, high = envelope.range(float(times[i]))
        along = envelope.path.project(point)
        best = min(best, off + max(0.0, low - SLACK - along, along - high - SLACK))
    return best
