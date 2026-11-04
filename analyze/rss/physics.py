"""
RSS physics, ported from ad-rss-lib 5.0.0 ``ad_rss/src/core/Physics.cpp``.

Every function returns None where the C++ returns false. Comparisons are
fuzzy, as ad_physics' operators are: two values are equal within 1e-3 for
distances, speeds and durations and 1e-4 for accelerations, and ``a < b``
holds only if they are not equal. Decelerations are negative.
"""

from math import copysign, sqrt
from typing import Tuple

__all__ = ["ACCELERATION", "DISTANCE", "DISTANCE_MAX", "DURATION", "DURATION_MAX", "SPEED", "SPEED_MAX",
           "accelerated_limited_movement", "distance_offset_accelerated", "feq", "fge", "fgt", "fle", "flt", "fmax",
           "fmin", "required_time_accelerated", "speed_accelerated", "speed_and_distance_offset", "stopping_distance",
           "time_for_distance", "time_to_cover_distance", "time_to_stop"]

DISTANCE = SPEED = DURATION = 1e-3
"""Precision of distances (m), speeds (m/s) and durations (s)."""
ACCELERATION = 1e-4
"""Precision of accelerations (m/s^2)."""
DISTANCE_MAX = 1e9
"""Metres: the ad_physics distance maximum, which the library uses as "infinite"."""
DURATION_MAX = 1e6
"""Seconds: the ad_physics duration maximum, which the library uses as "infinite"."""
SPEED_MAX = 100.0
"""m/s: the ad_physics speed maximum, which the library uses as "infinite"."""


def feq(a: float, b: float, precision: float) -> bool:
    """
    ``a == b`` within ``precision``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: True if ``|a - b| < precision``.
    :rtype: bool
    """
    return abs(a - b) < precision


def fgt(a: float, b: float, precision: float) -> bool:
    """
    ``a > b`` and not equal within ``precision``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: True if ``a`` exceeds ``b`` by at least ``precision``.
    :rtype: bool
    """
    return a > b and not feq(a, b, precision)


def flt(a: float, b: float, precision: float) -> bool:
    """
    ``a < b`` and not equal within ``precision``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: True if ``b`` exceeds ``a`` by at least ``precision``.
    :rtype: bool
    """
    return a < b and not feq(a, b, precision)


def fge(a: float, b: float, precision: float) -> bool:
    """
    ``a >= b`` within ``precision``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: True if ``a > b``, or they are equal within ``precision``.
    :rtype: bool
    """
    return a > b or feq(a, b, precision)


def fle(a: float, b: float, precision: float) -> bool:
    """
    ``a <= b`` within ``precision``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: True if ``a < b``, or they are equal within ``precision``.
    :rtype: bool
    """
    return a < b or feq(a, b, precision)


def fmax(a: float, b: float, precision: float) -> float:
    """
    ``std::max`` on ad_physics values: ``b`` if ``a < b``, else ``a``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: The larger; ``a`` when they are equal within ``precision``.
    :rtype: float
    """
    return b if flt(a, b, precision) else a


def fmin(a: float, b: float, precision: float) -> float:
    """
    ``std::min`` on ad_physics values: ``b`` if ``b < a``, else ``a``.

    :param float a: The left value.
    :param float b: The right value.
    :param float precision: How close counts as equal, e.g. ``SPEED``.
    :returns: The smaller; ``a`` when they are equal within ``precision``.
    :rtype: float
    """
    return b if flt(b, a, precision) else a


def _signbit(value: float) -> bool:
    """
    Return the floating-point sign bit, including the sign of negative zero.

    :param float value: The value.
    :returns: True if it is negative, ``-0.0`` included.
    :rtype: bool
    """
    return copysign(1.0, value) < 0


def distance_offset_accelerated(v: float, a: float, t: float) -> float | None:
    """
    Distance covered in ``t`` from ``v`` at ``a``, unclamped (Physics.cpp:25-38).

    :param float v: The initial speed, m/s.
    :param float a: The acceleration, m/s^2.
    :param float t: The duration, s.
    :returns: Metres, ``v t + a t^2 / 2``, even past a standstill; None for a negative ``t``.
    :rtype: float | None
    """
    if flt(t, 0.0, DURATION):
        return None
    return 0.5 * a * t * t + v * t


def speed_accelerated(v: float, a: float, t: float) -> float | None:
    """
    Speed after ``t`` from ``v`` at ``a``, unclamped (Physics.cpp:40-53).

    :param float v: The initial speed, m/s.
    :param float a: The acceleration, m/s^2.
    :param float t: The duration, s.
    :returns: m/s, ``v + a t``, even below 0; None for a negative ``t``.
    :rtype: float | None
    """
    if flt(t, 0.0, DURATION):
        return None
    return v + a * t


def stopping_distance(v: float, a: float) -> float | None:
    """
    Distance to stop from ``v`` braking at ``a`` (Physics.cpp:55-79); signed:
    a negative speed stopped by a positive ``a`` gives a negative distance.

    :param float v: The speed, m/s.
    :param float a: The acceleration, m/s^2: of the opposite sign to ``v``.
    :returns: Metres, ``v^2 / (-2 a)``; 0 standing at no acceleration; None when ``a`` is 0 while moving, or
        has ``v``'s sign bit.
    :rtype: float | None
    """
    if feq(a, 0.0, ACCELERATION):
        return 0.0 if feq(v, 0.0, SPEED) else None
    if _signbit(a) == _signbit(v):
        return None
    return v * v / (2 * -a)


def accelerated_limited_movement(v0: float, v_max: float | None, a: float, t: float
                                 ) -> Tuple[float, float] | None:
    """
    Speed and distance after ``t`` from ``v0`` at ``a``, never below standstill
    and, accelerating, never above ``max(v_max, v0)`` (Physics.cpp:81-128).

    :param float v0: The initial speed, m/s.
    :param float | None v_max: The speed cap, m/s; None for none.
    :param float a: The acceleration, m/s^2.
    :param float t: The duration, s.
    :returns: ``(speed, distance)``, m/s and m; None for a negative ``t`` or ``v0``.
    :rtype: Tuple[float, float] | None
    """
    if flt(t, 0.0, DURATION) or flt(v0, 0.0, SPEED):
        return None
    v = fmax(0.0, v0 + a * t, SPEED)
    if v_max is not None:
        v = fmin(fmax(v_max, v0, SPEED), v, SPEED)
    t_accelerated = (v - v0) / a if not feq(a, 0.0, ACCELERATION) else 0.0
    t_constant = t - t_accelerated
    return v, v0 * t + 0.5 * a * t_accelerated * t_accelerated + (v - v0) * t_constant


def required_time_accelerated(v0: float, a: float, s: float) -> float:
    """
    The first positive time to cover ``s`` from ``v0`` at ``a`` (Physics.cpp:130-153).

    :param float v0: The initial speed, m/s.
    :param float a: The acceleration, m/s^2; not 0.
    :param float s: The distance, m.
    :returns: Seconds.
    :rtype: float
    :raises ValueError: When braking at ``a`` stops it short of ``s``.
    """
    root = sqrt((v0 / a) ** 2 + 2 * s / a)
    t1, t2 = -v0 / a + root, -v0 / a - root
    return t2 if fgt(t2, 0.0, DURATION) else t1


def time_for_distance(v0: float, v_max: float, a: float, s: float) -> float | None:
    """
    Time to cover ``s`` from ``v0`` at ``a``, accelerating up to ``v_max`` (Physics.cpp:155-220).

    :param float v0: The initial speed, m/s.
    :param float v_max: The speed cap while accelerating, m/s.
    :param float a: The acceleration, m/s^2.
    :param float s: The distance, m.
    :returns: Seconds; ``DURATION_MAX`` standing at no acceleration; None for a negative ``v0``.
    :rtype: float | None
    :raises ValueError: When braking at ``a`` stops it short of ``s``, or the movement is inconsistent.
    """
    if flt(v0, 0.0, SPEED):
        return None
    if feq(a, 0.0, ACCELERATION) or (fge(v0, v_max, SPEED) and fgt(a, 0.0, ACCELERATION)):
        return DURATION_MAX if feq(v0, 0.0, SPEED) else s / v0
    t = required_time_accelerated(v0, a, s)
    if fgt(a, 0.0, ACCELERATION):
        t_to_max = (v_max - v0) / a
        if fgt(t, t_to_max, DURATION):
            s_accelerated = v0 * t_to_max + 0.5 * a * t_to_max * t_to_max
            if s_accelerated > s:
                raise ValueError(f"inconsistent movement: {s_accelerated} m accelerating, {s} m to cover")
            return t_to_max + (s - s_accelerated) / v_max
    return t


def time_to_cover_distance(v0: float, v_max: float, rho: float, a_response: float, a_after: float, s: float
                           ) -> float | None:
    """
    Time to cover ``s``: ``a_response`` during the response time ``rho`` (speed
    capped by ``v_max``), then ``a_after`` uncapped; ``DURATION_MAX`` when it stops
    short (Physics.cpp:222-289).

    :param float v0: The initial speed, m/s.
    :param float v_max: The speed cap during the response time, m/s.
    :param float rho: The response time, s.
    :param float a_response: The acceleration during it, m/s^2.
    :param float a_after: The acceleration after it, m/s^2.
    :param float s: The distance, m.
    :returns: Seconds; ``DURATION_MAX`` when it stops short; None for a negative ``s``, ``rho`` or ``v0``.
    :rtype: float | None
    """
    if flt(s, 0.0, DISTANCE):
        return None
    response = accelerated_limited_movement(v0, v_max, a_response, rho)
    if response is None:
        return None
    v_rho, s_rho = response
    if fge(s_rho, s, DISTANCE):
        return time_for_distance(v0, v_max, a_response, s)
    if feq(v_rho, 0.0, SPEED):
        return DURATION_MAX
    stop = stopping_distance(v_rho, a_after)
    if stop is None or fgt(s_rho + stop, s, DISTANCE):
        after = time_for_distance(v_rho, SPEED_MAX, a_after, s - s_rho)
        return None if after is None else rho + after
    return DURATION_MAX


def speed_and_distance_offset(t: float, v0: float, rho: float, v_max: float, a_response: float, a_after: float
                              ) -> Tuple[float, float] | None:
    """
    Speed and distance after ``t``: ``a_response`` during ``rho``, then ``a_after``,
    both capped by ``v_max`` (Physics.cpp:291-321).

    :param float t: The duration, s.
    :param float v0: The initial speed, m/s.
    :param float rho: The response time, s.
    :param float v_max: The speed cap, m/s.
    :param float a_response: The acceleration during the response time, m/s^2.
    :param float a_after: The acceleration after it, m/s^2.
    :returns: ``(speed, distance)``, m/s and m, never below standstill; None for a negative ``t``, ``rho`` or
        ``v0``.
    :rtype: Tuple[float, float] | None
    """
    first = accelerated_limited_movement(v0, v_max, a_response, fmin(t, rho, DURATION))
    if first is None:
        return None
    v, s = first
    if fgt(t, rho, DURATION):
        second = accelerated_limited_movement(v, v_max, a_after, t - rho)
        if second is None:
            return None
        v, s = second[0], s + second[1]
    return v, s


def time_to_stop(v0: float, rho: float, v_max: float, a_response: float, a_after: float) -> float | None:
    """
    Time until standstill: ``a_response`` during ``rho``, then ``a_after``
    (Physics.cpp:323-375); None when standing already while braking, or never stopping.

    :param float v0: The initial speed, m/s.
    :param float rho: The response time, s.
    :param float v_max: The speed cap during the response time, m/s.
    :param float a_response: The acceleration during it, m/s^2.
    :param float a_after: The acceleration after it, m/s^2.
    :returns: Seconds; None as above, or for a negative ``rho`` or ``v0``.
    :rtype: float | None
    """
    if flt(a_response, 0.0, ACCELERATION) and fle(v0, 0.0, SPEED):
        return None
    response = accelerated_limited_movement(v0, v_max, a_response, rho)
    if response is None:
        return None
    v_rho = response[0]
    if fge(a_after, 0.0, ACCELERATION) and fgt(v_rho, 0.0, SPEED):
        return None
    if feq(v_rho, 0.0, SPEED):
        return -v0 / a_response if fgt(v0, 0.0, SPEED) and flt(a_response, 0.0, ACCELERATION) else 0.0
    return rho - v_rho / a_after
