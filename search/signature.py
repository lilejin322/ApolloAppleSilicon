"""
What a scenario found, as a set of keys: the search's selection takes two
scenarios that found the same as duplicates (``nsga2.select``).

A key names a finding and where on the map it happened: the road user, the
constellation and the part of the ego for a collision; the signal, lane or
line an oracle names; else the junction, or the stretch of lane (``SPAN``
metres), the ego was in. Many slight variations of one scenario that find one
failure of Apollo share its key, while the same failure elsewhere on the map,
or with another road user, does not. Places are map elements rather than
coordinates, so no distance threshold has to be tuned.

Checked on the first 36 records of the 2026-11-02 search: records sharing a
key found it within 1.6 m of each other, their ego paths within 1.2 m over the
3 s around it, but for keys naming a whole lane, whose records lay up to 137 m
apart on it; hence the stretches. No two records with different keys found the
same within 8 m and 2 m.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ..analyze import Violation
from ..analyze.rss.judge import Moment, Verdict
from ..map import MapService
from .objectives import EGO_KINDS

__all__ = ["Places", "signature"]

SPAN = 20.0
"""Metres of lane one place covers."""
_NAMED = {"red_light", "speeding", "junction_lane_change"}
"""Kinds whose subject already names the place: the signal, the lane, the junction."""
_NOWHERE = {"planning_gap"}
"""Kinds that are one finding wherever they happen: Planning falling silent is its latency, not the place's (gaps of
2.0-2.2 s came at random places while the host was loaded), and keyed by place every one would count as new."""


class Places:
    """The junction a point lies in, or else the ``SPAN``-metre stretch of its nearest lane, as ``lane_3@40``."""

    def __init__(self, map_service: MapService) -> None:
        self.map_service = map_service
        areas = map_service.relations.junction_areas
        self._junctions: Dict[str, BaseGeometry] = {j: area for j, area in areas.items() if area is not None}

    def __call__(self, x: Optional[float], y: Optional[float]) -> str:
        if x is None or y is None:
            return "?"
        point = Point(x, y)
        for junction, area in self._junctions.items():
            if area.covers(point):
                return junction
        try:
            lane = self.map_service.get_nearest_lane(point)
        except (ValueError, RuntimeError):
            return "?"
        s = self.map_service.get_lane_central_curve_by_id(lane).project(point)
        return f"{lane}@{int(s // SPAN * SPAN)}"


def signature(violations: Iterable[Violation], verdicts: Sequence[Verdict], worst_at: Moment | None,
              places: Places) -> Tuple[str, ...]:
    """
    The keys of the ego's findings in one record, sorted.

    :param violations: What the oracles found; only the ego's kinds (``EGO_KINDS``) count.
    :param verdicts: The RSS oracle's verdicts, for the collisions' constellation and part.
    :param worst_at: Where the margin was least, a near miss when no collision; None when nothing lowered it.
    :param Places places: Locates the ego.
    :rtype: Tuple[str, ...]
    """
    by_touch = {(v.obstacle, round(v.t, 3)): v for v in verdicts}
    keys = set()
    for v in violations:
        if v.kind not in EGO_KINDS:
            continue
        if v.kind in ("collision", "collision_preventable"):
            verdict = by_touch.get((v.subject, round(v.t, 3)))
            road_user = v.subject.split()[0]
            how = f"{verdict.constellation}:{verdict.part}" if verdict is not None else "?"
            keys.add(f"{v.kind}:{road_user}:{how}:{places(v.x, v.y)}")
        elif v.kind in _NAMED:
            keys.add(f"{v.kind}:{v.subject}")
        elif v.kind in _NOWHERE:
            keys.add(v.kind)
        elif v.kind == "lane_marking":  # The line, and where along it: a curb is crossed at one corner.
            keys.add(f"{v.kind}:{v.subject}:{places(v.x, v.y)}")
        else:  # comfort (acceleration or braking), estop (its reason), planning_error
            subject = "" if v.subject == "planning" else f"{v.subject}:"
            keys.add(f"{v.kind}:{subject}{places(v.x, v.y)}")
    if worst_at is not None and not any(k.startswith(("collision:", "collision_preventable:")) for k in keys):
        keys.add(f"margin:{worst_at.road_user}:{worst_at.constellation}:{places(worst_at.x, worst_at.y)}")
    return tuple(sorted(keys))
