"""
Whether a collision the other road user caused was still preventable.

UN Regulation No. 157 (ALKS), 5.1.1: the system "shall not cause any
collisions that are reasonably foreseeable and preventable. If a collision can
be safely avoided without causing another one, it shall be avoided." Its
Annex 3 draws the line between preventable and unpreventable with a competent
and careful human driver who only brakes: once it perceives the risk it takes
0.4 s to evaluate it and 0.75 s more until it brakes, then jerks to 0.774 g in
0.6 s (Table 1).

Annex 3 perceives a cut-in when the other's centre deviates 0.375 m from its
lane centre, and a leading vehicle's braking beyond 5 m/s^2. To cover every
constellation alike, the risk here is perceived once the other both behaves
against RSS (fails a proper response or leaves RSS's bounds) and can no longer
keep out of the ground the ego covers until the touch, braking at its
``brake_min`` from then: it is visibly getting into the ego's way. A car cutting
in at 1.75 m/s sideways is perceived so as it turns towards the ego's lane,
0.2 s before Annex 3's point (0.375 m at 1.75 m/s): a little stricter on the ego.

The careful driver brakes along the ego's recorded path; the other moves as
recorded (replayed agents do not react). Braking so, it must not run into any
other road user (5.1.1: avoiding one collision by causing another is no safe
avoidance), but one running into its back is to blame for that: road users
behind the ego are not considered.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import Callable, Dict, List, Sequence, Tuple
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..footprint import box
from .unstructured import normalize_angle

__all__ = ["DECELERATION", "DECISION", "EVALUATION", "HORIZON", "JERK_TIME", "Prevention", "careful_driver_avoids",
           "careful_driver_gap", "careful_driver_runs_into", "perceived"]

EVALUATION = 0.4
"""Seconds to evaluate a perceived risk (R157 Annex 3, Table 1)."""
DECISION = 0.75
"""Seconds from the evaluated risk to the start of braking (Table 1)."""
JERK_TIME = 0.6
"""Seconds to full deceleration (Table 1)."""
DECELERATION = 0.774 * 9.81
"""m/s^2 of full deceleration, road friction 1.0 (Table 1)."""
HORIZON = 10.0
"""Seconds after the touch, or the danger's end, the braked ego is checked against the road users."""


@dataclass(frozen=True)
class Pose:
    """
    The ego's body centre at one frame.

    :param float t: Seconds since the record began.
    :param float x: Body centre, map frame.
    :param float y: Body centre.
    :param float heading: Radians.
    :param float speed: m/s.
    """
    t: float
    """Seconds since the record began."""
    x: float
    """Body centre, map frame."""
    y: float
    """Body centre."""
    heading: float
    """Radians."""
    speed: float
    """m/s."""


@dataclass(frozen=True)
class Prevention:
    """
    What the careful driver would have done: perceived the risk at
    ``perceived``, braked from ``brake``; whether that avoided the touch
    without running into another road user, else the one it ran into (``into``).

    :param float perceived: Seconds since the record began: when it perceived the risk (``perceived``).
    :param float brake: Seconds since the record began: when it began braking, ``EVALUATION + DECISION`` later.
    :param bool avoided: Whether braking so kept it clear of the other without running into another road user.
    :param str | None into: The road user it ran into avoiding the touch, e.g. ``"VEHICLE 1003"``; None otherwise.
    """
    perceived: float
    """Seconds since the record began: when it perceived the risk (``perceived``)."""
    brake: float
    """Seconds since the record began: when it began braking, ``EVALUATION + DECISION`` later."""
    avoided: bool
    """Whether braking so kept it clear of the other without running into another road user."""
    into: str | None = None
    """The road user it ran into avoiding the touch, e.g. ``"VEHICLE 1003"``; None otherwise."""


def perceived(ego_bodies: Sequence[Tuple[float, Polygon]], other: Sequence[Tuple[float, Polygon]],
              since: float | None) -> float:
    """
    When the risk is perceived: the first frame, from ``since`` on, at which the
    other's ground braking at ``brake_min`` meets the ground the ego's body
    covers from then until the touch (the last of ``ego_bodies``).

    :param Sequence[Tuple[float, Polygon]] ego_bodies: The ego's bodies up to the touch, ``(t, body)`` by time.
    :param Sequence[Tuple[float, Polygon]] other: At the same frames, the ground the other covers braking at its
        ``brake_min`` (``judge._sweep``).
    :param float | None since: When it first behaved against RSS; None when it was not seen to.
    :returns: Seconds since the record began; the touch's time when never before.
    :rtype: float
    """
    covered: List[Polygon] = []
    ground = Polygon()
    for _, body in reversed(ego_bodies):
        ground = unary_union([ground, body])
        covered.append(ground)
    covered.reverse()
    sweeps = dict(other)
    for (t, _), ahead in zip(ego_bodies, covered):
        if since is not None and t < since - 1e-6:
            continue
        sweep = sweeps.get(t)
        if sweep is not None and sweep.intersects(ahead):
            return t
    return ego_bodies[-1][0]


def careful_driver_avoids(poses: Sequence[Pose], length: float, width: float,
                          other: Sequence[Tuple[float, Polygon]], brake: float, until: float, *,
                          deceleration: float = DECELERATION, jerk_time: float = JERK_TIME,
                          on_path: bool = False) -> bool:
    """
    Whether the ego, braking from ``brake`` as the careful driver does along
    its recorded path, keeps clear of the other's recorded bodies until ``until``.

    :param Sequence[Pose] poses: The ego's recorded poses, by time.
    :param float length: The ego's length, m.
    :param float width: The ego's width, m.
    :param Sequence[Tuple[float, Polygon]] other: The other's bodies, ``(t, body)`` by time.
    :param float brake: Seconds since the record began: when the careful driver starts braking.
    :param float until: Seconds since the record began: the last time to check.
    :param float deceleration: m/s^2 of full deceleration, positive.
    :param float jerk_time: Seconds to full deceleration; 0 for all of it at once.
    :param bool on_path: Look for a touch only while the braked ego is on its recorded path: past its end
        the record shows no road, so nothing there proves one.
    :returns: True if the braked ego never touches the other's bodies from ``brake`` to ``until``.
    :rtype: bool
    """
    return careful_driver_gap(poses, length, width, other, brake, until, stop_at_touch=True,
                              deceleration=deceleration, jerk_time=jerk_time, on_path=on_path) > 0.0


def careful_driver_gap(poses: Sequence[Pose], length: float, width: float, other: Sequence[Tuple[float, Polygon]],
                       brake: float, until: float, *, stop_at_touch: bool = False,
                       deceleration: float = DECELERATION, jerk_time: float = JERK_TIME, on_path: bool = False) -> float:
    """
    The least distance between the careful driver's body, braking from
    ``brake`` along the ego's recorded path, and the other's recorded bodies,
    from ``brake`` to ``until``; 0 at a touch.

    :param Sequence[Pose] poses: The ego's recorded poses, by time.
    :param float length: The ego's length, m.
    :param float width: The ego's width, m.
    :param Sequence[Tuple[float, Polygon]] other: The other's bodies, ``(t, body)`` by time.
    :param float brake: Seconds since the record began: when the careful driver starts braking.
    :param float until: Seconds since the record began: the last time to check.
    :param bool stop_at_touch: Return at the first touch.
    :param float deceleration: As for ``careful_driver_avoids``.
    :param float jerk_time: As for ``careful_driver_avoids``.
    :param bool on_path: As for ``careful_driver_avoids``.
    :returns: Metres; 0 at a touch; infinite when none of the other's bodies is checked in the window.
    :rtype: float
    """
    pose = _braking(poses, brake, deceleration, jerk_time)
    gap = float("inf")
    for t, body in other:
        if t < brake - 1e-6 or t > until + 1e-6:
            continue
        x, y, heading, recorded = pose(t)
        if on_path and not recorded:
            continue
        gap = min(gap, box(x, y, heading, length, width).distance(body))
        if gap == 0.0 and stop_at_touch:
            break
    return gap


def careful_driver_runs_into(poses: Sequence[Pose], length: float, width: float,
                             traffic: Sequence[Tuple[float, Polygon, str]], brake: float, until: float, *,
                             deceleration: float = DECELERATION, jerk_time: float = JERK_TIME) -> str | None:
    """
    The first of ``traffic`` the ego, braking from ``brake`` as the careful
    driver does along its recorded path, touches until ``until``; None when
    none. One first touching it at its back ran into the braking ego, and
    replayed it then passes through: it does not count.

    :param Sequence[Pose] poses: The ego's recorded poses, by time.
    :param float length: The ego's length, m.
    :param float width: The ego's width, m.
    :param Sequence[Tuple[float, Polygon, str]] traffic: Road users' ``(t, body, name)``, in time order, several
        at a time.
    :param float brake: Seconds since the record began: when the careful driver starts braking.
    :param float until: Seconds since the record began: the last time to check.
    :param float deceleration: As for ``careful_driver_avoids``.
    :param float jerk_time: As for ``careful_driver_avoids``.
    :returns: The name of the first it runs into, e.g. ``"VEHICLE 1003"``; None when none.
    :rtype: str | None
    """
    pose = _braking(poses, brake, deceleration, jerk_time)
    braked: Dict[float, Tuple[float, float, float, Polygon]] = {}
    behind = set()
    for t, body, name in traffic:
        if t < brake - 1e-6 or t > until + 1e-6 or name in behind:
            continue
        if t not in braked:
            x, y, heading, _ = pose(t)
            braked[t] = x, y, heading, box(x, y, heading, length, width)
        x, y, heading, mine = braked[t]
        if mine.intersects(body):
            centre = body.centroid
            if abs(normalize_angle(atan2(centre.y - y, centre.x - x) - heading)) < 3 * pi / 4:
                return name
            behind.add(name)
    return None


def _braking(poses: Sequence[Pose], brake: float, deceleration: float,
             jerk_time: float) -> Callable[[float], Tuple[float, float, float, bool]]:
    """
    The careful driver's centre and heading at a time from ``brake`` on, along
    the recorded path, and whether it is still on it.

    :param Sequence[Pose] poses: The ego's recorded poses, by time.
    :param float brake: Seconds since the record began: when it starts braking, from its recorded place and speed.
    :param float deceleration: m/s^2 of full deceleration, positive.
    :param float jerk_time: Seconds to full deceleration; 0 for all of it at once.
    :returns: Its pose at a time: ``(x, y, heading, on_path)``.
    :rtype: Callable[[float], Tuple[float, float, float, bool]]
    """
    times = [p.t for p in poses]
    arc = [0.0]
    for a, b in zip(poses, poses[1:]):
        arc.append(arc[-1] + ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5)
    start = _interpolate(times, arc, brake)
    speed = _interpolate(times, [p.speed for p in poses], brake)

    def pose(t: float) -> Tuple[float, float, float, bool]:
        """
        Return a braked pose along the recorded path and whether it remains within that path.

        :param float t: Seconds since the record began, from ``brake`` on.
        :returns: ``(x, y, heading, on_path)``: its body centre, map frame, its heading in radians, and whether
            it has not yet passed the end of the recorded path.
        :rtype: Tuple[float, float, float, bool]
        """
        s = start + _braked(speed, t - brake, deceleration, jerk_time)
        return (*_along(poses, arc, s), s <= arc[-1] + 1e-6)
    return pose


def _braked(speed: float, elapsed: float, deceleration: float = DECELERATION, jerk_time: float = JERK_TIME) -> float:
    """
    Metres covered in ``elapsed`` seconds from ``speed``, jerking to full ``deceleration`` in ``jerk_time``.

    :param float speed: m/s when braking starts.
    :param float elapsed: Seconds since.
    :param float deceleration: m/s^2 of full deceleration, positive.
    :param float jerk_time: Seconds to full deceleration; 0 or less for all of it at once.
    :returns: Metres, no further than where it stops.
    :rtype: float
    """
    if jerk_time <= 0.0:
        rest = min(elapsed, speed / deceleration)
        return speed * rest - deceleration * rest * rest / 2
    jerk = deceleration / jerk_time
    ramp = min(elapsed, jerk_time)
    v = speed - jerk * ramp * ramp / 2
    if v <= 0.0:  # Stops while the brake builds up: v(tau) = speed - jerk tau^2 / 2.
        stop = (2 * speed / jerk) ** 0.5
        tau = min(elapsed, stop)
        return speed * tau - jerk * tau ** 3 / 6
    covered = speed * ramp - jerk * ramp ** 3 / 6
    rest = max(elapsed - jerk_time, 0.0)
    if rest == 0.0:
        return covered
    rest = min(rest, v / deceleration)
    return covered + v * rest - deceleration * rest * rest / 2


def _interpolate(times: Sequence[float], values: Sequence[float], t: float) -> float:
    """
    Linearly interpolate a nonempty sampled series, clamping queries outside its time range.

    :param Sequence[float] times: The sample times, ascending.
    :param Sequence[float] values: The value at each.
    :param float t: The time asked for.
    :returns: The value at ``t``; the first or last value outside the samples' range.
    :rtype: float
    """
    i = bisect_right(times, t)
    if i == 0:
        return values[0]
    if i == len(times):
        return values[-1]
    t0, t1 = times[i - 1], times[i]
    return values[i - 1] + (values[i] - values[i - 1]) * (t - t0) / (t1 - t0) if t1 > t0 else values[i]


def _along(poses: Sequence[Pose], arc: Sequence[float], s: float) -> Tuple[float, float, float]:
    """
    The pose ``s`` metres along the recorded path; straight on past its end.

    :param Sequence[Pose] poses: The ego's recorded poses, by time.
    :param Sequence[float] arc: Metres along the path at each pose.
    :param float s: Metres along the path.
    :returns: ``(x, y, heading)``: the body centre, map frame, interpolated between poses, and the heading in
        radians; past the end, on along the last heading.
    :rtype: Tuple[float, float, float]
    """
    i = bisect_right(arc, s)
    if i >= len(arc):
        last = poses[-1]
        beyond = s - arc[-1]
        return last.x + cos(last.heading) * beyond, last.y + sin(last.heading) * beyond, last.heading
    i = max(i, 1)
    a, b = poses[i - 1], poses[i]
    span = arc[i] - arc[i - 1]
    k = (s - arc[i - 1]) / span if span > 0 else 0.0
    heading = a.heading + normalize_angle(b.heading - a.heading) * k
    return a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, atan2(sin(heading), cos(heading))
