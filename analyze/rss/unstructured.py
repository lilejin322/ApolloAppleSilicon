"""
The RSS checker of unstructured constellations (pedestrians, road users off
the lanes), ported from ad-rss-lib 5.0.0
``ad_rss/src/unstructured/RssUnstructuredConstellationChecker.cpp``
(digest §6.4): Definition 22 decides from the brake and continue-forward
trajectory sets who must brake, who may continue and when to drive away.
"""

from math import atan2, pi
from typing import Dict, Tuple

from shapely.geometry.base import BaseGeometry

from .physics import feq
from .trajectories import sets
from .types import RelativeConstellation, RssState, UnstructuredResponse

__all__ = ["DRIVE_AWAY_STOPPED_MAX_SPEED", "UnstructuredChecker", "drive_away_range", "inside_heading_range",
           "normalize_angle"]

DRIVE_AWAY_STOPPED_MAX_SPEED = 0.1
"""m/s up to which a road user counts as stopped (checker line 21)."""

Sets = Tuple[BaseGeometry, BaseGeometry]


def normalize_angle(angle: float) -> float:
    """``normalizeAngleSigned``: into (-pi, pi]."""
    angle = (angle + pi) % (2 * pi) - pi
    return pi if angle == -pi else angle


def drive_away_range(ego: Tuple[float, float], other: Tuple[float, float], max_angle: float) -> Tuple[float, float]:
    """Headings that lead away from ``other``, within ``max_angle`` of its opposite (checker 376-390)."""
    away = atan2(ego[1] - other[1], ego[0] - other[0])
    return normalize_angle(away - max_angle), normalize_angle(away + max_angle)


def inside_heading_range(angle: float, heading_range: Tuple[float, float]) -> bool:
    """Whether ``angle`` lies in ``[begin, end]``, wrapping through pi when ``end < begin``."""
    begin, end = heading_range
    if end >= begin:
        return begin <= angle <= end
    return angle >= begin or angle <= end


def _collides(a: BaseGeometry, b: BaseGeometry) -> bool:
    """Return whether two reachable sets intersect; boundary contact counts as collision."""
    return not a.disjoint(b)  # Touching counts.


class UnstructuredChecker:
    """
    Definition 22 per constellation, with the library's memories: whether the
    other had to brake at the last safe time, and a drive-away until it is
    safe, the other moves or the ego steers out of the range.
    """

    def __init__(self) -> None:
        """Initialize per-object safety history, drive-away positions and the per-frame ego-set cache."""
        self._time: int | None = None
        self._last: Dict[int, bool] = {}
        self._new: Dict[int, bool] = {}
        self._drive_away: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._ego_sets: Sets | None = None

    def check(self, constellation: RelativeConstellation, time_index: int, *, ego_sets: Sets | None = None,
              other_sets: Sets | None = None) -> RssState | None:
        """
        The ego's state and proper response.

        :param ego_sets: The ego's ``(brake, continue_forward)`` sets; computed once per time index by default.
        :param other_sets: The other's; computed by default.
        :returns: The state, or None when the sets cannot be calculated or are empty.
        """
        c = constellation
        if time_index != self._time:
            self._last, self._new, self._time, self._ego_sets = self._new, {}, time_index, None
        if ego_sets is None:
            if self._ego_sets is None:
                self._ego_sets = sets(c.ego)
            ego_sets = self._ego_sets
        if other_sets is None:
            other_sets = sets(c.other)
        if ego_sets is None or other_sets is None or any(s.is_empty for s in (*ego_sets, *other_sets)):
            return None
        state = RssState(c.constellation_id, c.constellation_type)
        result = state.unstructured
        ego_brake, ego_forward = ego_sets
        other_brake, other_forward = other_sets
        ego_hits_other = _collides(ego_brake, other_forward)
        other_hits_ego = _collides(other_brake, ego_forward)
        ego_must_brake = not ego_hits_other and other_hits_ego
        other_must_brake = not other_hits_ego and ego_hits_other
        brake_both = not _collides(ego_brake, other_brake)
        ego, other = c.ego.unstructured, c.other.unstructured
        if ego_must_brake or other_must_brake or brake_both:
            self._new[c.constellation_id] = other_must_brake
            response = UnstructuredResponse.CONTINUE_FORWARD
        else:
            last = self._last.get(c.constellation_id, False)
            if c.constellation_id in self._last:
                self._new[c.constellation_id] = last
            if ego.speed_max <= DRIVE_AWAY_STOPPED_MAX_SPEED and other.speed_max <= DRIVE_AWAY_STOPPED_MAX_SPEED:
                response = UnstructuredResponse.DRIVE_AWAY  # Rule 1.
            elif last:  # Rule 2: the other had to brake at the last safe time.
                response = (UnstructuredResponse.CONTINUE_FORWARD if ego.speed_max > DRIVE_AWAY_STOPPED_MAX_SPEED
                            else UnstructuredResponse.DRIVE_AWAY)
            else:
                response = UnstructuredResponse.BRAKE  # Rule 3.
        result.response = response
        result.is_safe = response == UnstructuredResponse.CONTINUE_FORWARD  # Also under Rule 2, as the library.
        if response == UnstructuredResponse.DRIVE_AWAY:
            result.heading_range = drive_away_range((ego.x, ego.y), (other.x, other.y),
                                                    c.ego.dynamics.unstructured.drive_away_max_angle)
            self._drive_away[c.constellation_id] = (result.heading_range, (other.x, other.y))
        elif c.constellation_id in self._drive_away:
            heading_range, center = self._drive_away[c.constellation_id]
            steering = normalize_angle(ego.yaw + ego.steering_angle)
            moved = not (feq(center[0], other.x, 1e-3) and feq(center[1], other.y, 1e-3))
            if result.is_safe or moved or not inside_heading_range(steering, heading_range):
                del self._drive_away[c.constellation_id]
            else:
                result.response, result.is_safe, result.heading_range = UnstructuredResponse.DRIVE_AWAY, False, \
                    heading_range
        return state
