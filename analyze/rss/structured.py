"""
The structured RSS checkers, ported from ad-rss-lib 5.0.0
``RssNonIntersectionConstellationChecker.cpp`` and
``RssIntersectionConstellationChecker.cpp`` (digest §4, §5). Each instance
keeps the danger-threshold memory of its constellations from one time index
to the next, so one instance must see every frame in order. A check returns
None where the C++ returns false.
"""

from typing import Dict, Tuple

from .formulas import check_lat, check_lon_opposite, check_lon_same, check_stop_in_front
from .physics import DISTANCE, DISTANCE_MAX, DURATION, DURATION_MAX, feq, fgt, flt, time_to_cover_distance
from .types import (ConstellationType, Evaluator, IntersectionState, LatPosition, LatResponse, LatState, LonPosition,
                    LonResponse, LonState, RelativeConstellation, RssState, is_dangerous, is_lateral_safe,
                    is_longitudinal_safe)

__all__ = ["IntersectionChecker", "NonIntersectionChecker"]

_OVERLAPS = (LatPosition.OVERLAP_LEFT, LatPosition.OVERLAP, LatPosition.OVERLAP_RIGHT)


class NonIntersectionChecker:
    """Same and opposite direction (RssNonIntersectionConstellationChecker.cpp)."""

    def __init__(self) -> None:
        """Initialize previous-frame and current-frame same/opposite-direction safety histories."""
        self._time: int | None = None
        self._last: Dict[int, Tuple[bool, bool]] = {}
        self._new: Dict[int, Tuple[bool, bool]] = {}

    def check(self, constellation: RelativeConstellation, time_index: int) -> RssState | None:
        """
        The ego's state and proper response in ``constellation`` at ``time_index``.

        :returns: The state, or None for invalid input.
        """
        if time_index != self._time:
            self._last, self._new, self._time = self._new, {}, time_index
        c = constellation
        state = RssState(c.constellation_id, c.constellation_type)
        if c.constellation_type == ConstellationType.SAME_DIRECTION:
            ok = _lon_same(c, state.lon) and _lateral(c, state)
        elif c.constellation_type == ConstellationType.OPPOSITE_DIRECTION:
            ok = _lon_opposite(c, state.lon) and _lateral(c, state)
        else:
            ok = False
        if not ok:
            return None
        if is_dangerous(state):
            if c.constellation_id in self._last:
                remembered = self._last[c.constellation_id]
            elif c.constellation_type == ConstellationType.SAME_DIRECTION and c.lat_position in _OVERLAPS:
                remembered = (True, False)  # Danger at first sight with lateral overlap: brake, do not swerve.
            else:
                remembered = (False, False)
        else:
            remembered = (is_longitudinal_safe(state), is_lateral_safe(state))
        lon_safe, lat_safe = remembered
        if lat_safe and not lon_safe:
            state.lon.response = LonResponse.NONE
        elif lon_safe and not lat_safe:
            state.lat_left.response = state.lat_right.response = LatResponse.NONE
        if c.constellation_id in self._new:
            return None
        self._new[c.constellation_id] = remembered
        return state


def _lon_same(c: RelativeConstellation, lon: LonState) -> bool:
    """Update same-direction longitudinal safety and response; return whether the calculation succeeded."""
    lon.current_distance = c.lon_distance
    if c.lon_position in (LonPosition.IN_FRONT, LonPosition.OVERLAP_FRONT):
        lon.evaluator = Evaluator.LONGITUDINAL_DISTANCE_SAME_DIRECTION_EGO_FRONT
        lon.response = LonResponse.NONE
        ok, lon.safe_distance, lon.is_safe = check_lon_same(c.ego, c.other, c.lon_distance)
    else:
        lon.evaluator = Evaluator.LONGITUDINAL_DISTANCE_SAME_DIRECTION_OTHER_IN_FRONT
        ok, lon.safe_distance, lon.is_safe = check_lon_same(c.other, c.ego, c.lon_distance)
    if lon.is_safe:
        lon.response = LonResponse.NONE
    return ok


def _lon_opposite(c: RelativeConstellation, lon: LonState) -> bool:
    """Update opposing-traffic longitudinal safety using the correct-lane braking rule when applicable."""
    lon.current_distance = c.lon_distance
    if c.ego.structured.is_in_correct_lane:
        lon.evaluator = Evaluator.LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION_EGO_CORRECT_LANE
        ok, lon.safe_distance, lon.is_safe = check_lon_opposite(c.ego, c.other, c.lon_distance)
        lon.response = LonResponse.BRAKE_MIN_CORRECT
    else:
        lon.evaluator = Evaluator.LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION
        ok, lon.safe_distance, lon.is_safe = check_lon_opposite(c.other, c.ego, c.lon_distance)
    if lon.is_safe:
        lon.response = LonResponse.NONE
    return ok


def _lateral(c: RelativeConstellation, state: RssState) -> bool:
    """Update both lateral responses from relative position; return whether the distance calculation succeeded."""
    left, right = state.lat_left, state.lat_right
    position = c.lat_position
    if position == LatPosition.AT_LEFT:
        left.evaluator, left.current_distance, left.safe_distance = Evaluator.NONE, DISTANCE_MAX, DISTANCE_MAX
        right.evaluator, right.current_distance = Evaluator.LATERAL_DISTANCE, c.lat_distance
        ok, right.safe_distance, is_safe = check_lat(c.ego, c.other, c.lat_distance)
    elif position == LatPosition.AT_RIGHT:
        right.evaluator, right.current_distance, right.safe_distance = Evaluator.NONE, DISTANCE_MAX, DISTANCE_MAX
        left.evaluator, left.current_distance = Evaluator.LATERAL_DISTANCE, c.lat_distance
        ok, left.safe_distance, is_safe = check_lat(c.other, c.ego, c.lat_distance)
    else:
        for side in (left, right):
            side.evaluator, side.current_distance, side.safe_distance = Evaluator.LATERAL_DISTANCE, 0.0, 0.0
        ok, is_safe = True, False
    if is_safe:
        _safe(left), _safe(right)
    elif position in (LatPosition.AT_LEFT, LatPosition.OVERLAP_LEFT):
        _safe(left)
    elif position in (LatPosition.AT_RIGHT, LatPosition.OVERLAP_RIGHT):
        _safe(right)
    return ok


def _safe(side: LatState) -> None:
    """Mark one lateral component safe and remove its braking response."""
    side.is_safe, side.response = True, LatResponse.NONE


class IntersectionChecker:
    """The three intersection types (RssIntersectionConstellationChecker.cpp)."""

    def __init__(self) -> None:
        """Initialize previous-frame and current-frame intersection-state histories."""
        self._time: int | None = None
        self._last: Dict[int, IntersectionState] = {}
        self._new: Dict[int, IntersectionState] = {}

    def check(self, constellation: RelativeConstellation, time_index: int) -> RssState | None:
        """
        The ego's state and proper response in an intersection ``constellation``.

        :returns: The state, or None for invalid input (both with priority, leave before enter).
        """
        c = constellation
        if c.ego.structured.has_priority and c.other.structured.has_priority:
            return None
        if time_index != self._time:
            self._last, self._new, self._time = self._new, {}, time_index
        state = RssState(c.constellation_id, c.constellation_type)
        for side in (state.lat_left, state.lat_right):
            side.response, side.evaluator, side.current_distance, side.safe_distance = (
                LatResponse.NONE, Evaluator.LATERAL_DISTANCE, 0.0, 0.0)
        ok, safe_state = _intersection_safe(c, state.lon)
        if not ok:
            return None
        state.lon.is_safe = safe_state is not None
        if safe_state is None:
            previous = self._last.get(c.constellation_id)
            if previous is None:
                state.lon.response = LonResponse.BRAKE_MIN
            else:
                if previous == IntersectionState.NON_PRIO_ABLE_TO_BREAK:
                    state.lon.response = LonResponse.NONE if c.ego.structured.has_priority else LonResponse.BRAKE_MIN
                elif previous == IntersectionState.SAFE_LONGITUDINAL_DISTANCE:
                    state.lon.response = (LonResponse.NONE if c.lon_position == LonPosition.IN_FRONT
                                          else LonResponse.BRAKE_MIN)
                else:
                    state.lon.response = LonResponse.BRAKE_MIN
                self._new[c.constellation_id] = previous
        else:
            state.lon.response = LonResponse.NONE
            self._new[c.constellation_id] = safe_state
        return state


def _intersection_safe(c: RelativeConstellation, lon: LonState) -> Tuple[bool, IntersectionState | None]:
    """Evaluate intersection safety and update longitudinal diagnostics; return validity and a safe-state reason."""
    ego, other = c.ego.structured, c.other.structured
    if (flt(ego.distance_to_leave_intersection, ego.distance_to_enter_intersection, DISTANCE)
            or flt(other.distance_to_leave_intersection, other.distance_to_enter_intersection, DISTANCE)):
        return False, None
    is_safe = False
    if not ego.has_priority:
        lon.evaluator = Evaluator.INTERSECTION_OTHER_PRIORITY_EGO_ABLE_TO_STOP
        lon.current_distance = ego.distance_to_enter_intersection
        ok, lon.safe_distance, is_safe = check_stop_in_front(c.ego)
        if not ok:
            return False, None
    if not is_safe and not other.has_priority:
        lon.evaluator = Evaluator.INTERSECTION_EGO_PRIORITY_OTHER_ABLE_TO_STOP
        lon.current_distance = other.distance_to_enter_intersection
        ok, lon.safe_distance, is_safe = check_stop_in_front(c.other)
        if not ok:
            return False, None
    if is_safe:
        return True, IntersectionState.NON_PRIO_ABLE_TO_BREAK
    lon.current_distance = c.lon_distance
    if c.lon_position == LonPosition.IN_FRONT:
        lon.evaluator = Evaluator.INTERSECTION_EGO_IN_FRONT
        ok, lon.safe_distance, is_safe = check_lon_same(c.ego, c.other, c.lon_distance)
    else:
        lon.evaluator = Evaluator.INTERSECTION_OTHER_IN_FRONT
        ok, lon.safe_distance, is_safe = check_lon_same(c.other, c.ego, c.lon_distance)
    if not ok:
        return False, None
    if is_safe:
        return True, IntersectionState.SAFE_LONGITUDINAL_DISTANCE
    lon.evaluator, lon.current_distance, lon.safe_distance = Evaluator.INTERSECTION_OVERLAP, 0.0, 0.0
    ok, is_safe = _no_time_overlap(c)
    if not ok:
        return False, None
    return True, IntersectionState.NO_TIME_OVERLAP if is_safe else None


def _no_time_overlap(c: RelativeConstellation) -> Tuple[bool, bool]:
    """Whether one has left the conflict area before the other can reach it (checkLateralIntersect, :22-91)."""
    e, o = c.ego, c.other
    v_max = e.dynamics.max_speed_on_acceleration  # The library uses the ego's cap for both (lines 49, 69).

    def reach(obj):
        """Compute time to the intersection entrance using maximum speed and response-then-minimum braking."""
        d, s = obj.dynamics, obj.structured
        return time_to_cover_distance(s.velocity.speed_lon_max, v_max, d.response_time, d.alpha_lon.accel_max,
                                      d.alpha_lon.brake_min, s.distance_to_enter_intersection)

    def leave(obj):
        """Compute time to leave the intersection using minimum speed and maximum braking."""
        d, s = obj.dynamics, obj.structured
        return time_to_cover_distance(s.velocity.speed_lon_min, v_max, d.response_time, d.alpha_lon.brake_max,
                                      d.alpha_lon.brake_max, s.distance_to_leave_intersection)

    times = reach(e), reach(o), leave(e), leave(o)
    if any(t is None for t in times):
        return False, False
    reach_ego, reach_other, leave_ego, leave_other = times
    return True, (fgt(reach_ego, leave_other, DURATION) or fgt(reach_other, leave_ego, DURATION)
                  or (feq(reach_ego, DURATION_MAX, DURATION) and feq(reach_other, DURATION_MAX, DURATION)))
