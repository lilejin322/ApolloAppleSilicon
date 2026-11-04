"""
RSS safe distances, ported from ad-rss-lib 5.0.0
``ad_rss/src/structured/RssFormulas.cpp`` (digest §2). Functions return None
where the C++ returns false; the ``check_*`` functions return
``(ok, safe_distance, is_safe)``.
"""

from math import copysign
from typing import Tuple

from .physics import (DISTANCE, DISTANCE_MAX, accelerated_limited_movement, distance_offset_accelerated, fgt, flt,
                      fmax, speed_accelerated, stopping_distance)
from .types import RelativeObjectState

__all__ = ["Check", "check_lat", "check_lon_opposite", "check_lon_same", "check_stop_in_front", "lat_stated_braking",
           "lon_stated_braking", "safe_lat", "safe_lon_opposite", "safe_lon_same", "valid"]

Check = Tuple[bool, float, bool]
"""``(ok, safe_distance, is_safe)``."""


def valid(obj: RelativeObjectState) -> bool:
    """The generated input-range check plus ``speed_lon_min >= 0`` (RssFormulas.cpp:24-35)."""
    d, v = obj.dynamics, obj.structured.velocity
    lon, lat = d.alpha_lon, d.alpha_lat
    return (0 <= lon.accel_max <= 100 and -100 <= lon.brake_max <= lon.brake_min <= lon.brake_min_correct < 0
            and 0 <= lat.accel_max <= 100 and -100 <= lat.brake_min < 0
            and 0 <= d.lateral_fluctuation_margin <= 1 and 0 < d.response_time <= 10
            and all(-100 <= s <= 100 for s in (v.speed_lon_min, v.speed_lon_max, v.speed_lat_min, v.speed_lat_max))
            and v.speed_lon_min >= 0)


def _signbit(value: float) -> bool:
    """Return the floating-point sign bit, including the sign of negative zero."""
    return copysign(1.0, value) < 0


def lon_stated_braking(v: float, v_max: float, rho: float, accel: float, brake: float) -> float | None:
    """
    Distance covered accelerating at ``accel`` (speed capped) for ``rho``, then
    braking at ``brake`` to standstill (RssFormulas.cpp:37-67).
    """
    response = accelerated_limited_movement(v, v_max, accel, rho)
    if response is None:
        return None
    v_rho, s_rho = response
    stop = 0.0
    if _signbit(v_rho) != _signbit(brake):
        stop = stopping_distance(v_rho, brake)
        if stop is None:
            return None
    return s_rho + stop


def lat_stated_braking(v: float, rho: float, accel: float, brake: float) -> float | None:
    """
    Lateral distance covered accelerating at ``accel`` for ``rho``, then braking
    at ``brake``, unclamped: the motion may reverse (RssFormulas.cpp:69-97).
    """
    v_rho, s_rho = speed_accelerated(v, accel, rho), distance_offset_accelerated(v, accel, rho)
    if v_rho is None or s_rho is None:
        return None
    stop = 0.0
    if _signbit(v_rho) != _signbit(brake):
        stop = stopping_distance(v_rho, brake)
        if stop is None:
            return None
    return s_rho + stop


def safe_lon_same(leading: RelativeObjectState, following: RelativeObjectState) -> float | None:
    """Safe distance with ``following`` behind ``leading`` in the same direction (RssFormulas.cpp:99-131)."""
    if not valid(leading) or not valid(following):
        return None
    f, l = following.dynamics, leading.dynamics
    braking = lon_stated_braking(following.structured.velocity.speed_lon_max, f.max_speed_on_acceleration,
                                 f.response_time, f.alpha_lon.accel_max, f.alpha_lon.brake_min)
    stop = stopping_distance(leading.structured.velocity.speed_lon_min, l.alpha_lon.brake_max)
    if braking is None or stop is None:
        return None
    return fmax(braking - stop, 0.0, DISTANCE) + f.min_longitudinal_safety_distance


def safe_lon_opposite(correct: RelativeObjectState, opposite: RelativeObjectState) -> float | None:
    """
    Safe distance between ``correct`` (in its correct lane, braking with
    ``brake_min_correct``) and ``opposite`` approaching it (RssFormulas.cpp:156-197).
    """
    if not valid(correct) or not valid(opposite):
        return None
    c, o = correct.dynamics, opposite.dynamics
    first = lon_stated_braking(correct.structured.velocity.speed_lon_max, c.max_speed_on_acceleration,
                               c.response_time, c.alpha_lon.accel_max, c.alpha_lon.brake_min_correct)
    second = lon_stated_braking(opposite.structured.velocity.speed_lon_max, o.max_speed_on_acceleration,
                                o.response_time, o.alpha_lon.accel_max, o.alpha_lon.brake_min)
    if first is None or second is None:
        return None
    return first + second + fmax(c.min_longitudinal_safety_distance, o.min_longitudinal_safety_distance, DISTANCE)


def safe_lat(left: RelativeObjectState, right: RelativeObjectState) -> float | None:
    """
    Safe lateral distance between ``left`` and ``right`` of it; lateral speed is
    positive to the right (RssFormulas.cpp:252-290).
    """
    if not valid(left) or not valid(right):
        return None
    l, r = left.dynamics, right.dynamics
    to_left = lat_stated_braking(left.structured.velocity.speed_lat_max, l.response_time, l.alpha_lat.accel_max,
                                 l.alpha_lat.brake_min)
    to_right = lat_stated_braking(right.structured.velocity.speed_lat_min, r.response_time, -r.alpha_lat.accel_max,
                                  -r.alpha_lat.brake_min)
    if to_left is None or to_right is None:
        return None
    margin = 0.5 * (l.lateral_fluctuation_margin + r.lateral_fluctuation_margin)
    return fmax(to_left - to_right + margin, 0.0, DISTANCE)


def _check(distance: float, safe: float | None) -> Check:
    """Return validity, safe distance and strict safety, using RSS distance tolerances."""
    if flt(distance, 0.0, DISTANCE) or safe is None:
        return False, DISTANCE_MAX, False
    return True, safe, fgt(distance, safe, DISTANCE)


def check_lon_same(leading: RelativeObjectState, following: RelativeObjectState, distance: float) -> Check:
    """Whether ``distance`` is safe in the same direction (RssFormulas.cpp:133-154)."""
    if flt(distance, 0.0, DISTANCE):
        return False, DISTANCE_MAX, False
    return _check(distance, safe_lon_same(leading, following))


def check_lon_opposite(correct: RelativeObjectState, opposite: RelativeObjectState, distance: float) -> Check:
    """Whether ``distance`` is safe in opposite directions (RssFormulas.cpp:199-219)."""
    if flt(distance, 0.0, DISTANCE):
        return False, DISTANCE_MAX, False
    return _check(distance, safe_lon_opposite(correct, opposite))


def check_lat(left: RelativeObjectState, right: RelativeObjectState, distance: float) -> Check:
    """Whether the lateral ``distance`` is safe (RssFormulas.cpp:292-313)."""
    if flt(distance, 0.0, DISTANCE):
        return False, DISTANCE_MAX, False
    return _check(distance, safe_lat(left, right))


def check_stop_in_front(obj: RelativeObjectState) -> Check:
    """
    Whether ``obj`` can stop in front of the intersection: its stated braking
    distance below ``distance_to_enter_intersection`` (RssFormulas.cpp:221-250).
    """
    if not valid(obj):
        return False, 0.0, False
    d = obj.dynamics
    braking = lon_stated_braking(obj.structured.velocity.speed_lon_max, d.max_speed_on_acceleration, d.response_time,
                                 d.alpha_lon.accel_max, d.alpha_lon.brake_min)
    safe = (braking if braking is not None else 0.0) + d.min_longitudinal_safety_distance
    return braking is not None, safe, flt(safe, obj.structured.distance_to_enter_intersection, DISTANCE)
