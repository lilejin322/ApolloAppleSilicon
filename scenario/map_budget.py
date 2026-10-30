"""
Map-dependent agent count caps (``MapBudget``).

The scenario chromosome never names its map: a section's size gene is a
relative ``density`` in ``[0, 1]``. The absolute cap per agent type comes from
the map the scenario replays on, so one density yields more agents on a large
map and fewer on a small one.

Formula (per section type, base cap ``B`` = that section's ``RULES.max_count``)::

    L      = drivable length of the map: sum of CAR-lane centrelines, metres
    scale  = clamp(sqrt(L / reference_length), min_scale, max_scale)
    cap    = max(min_count, round(B * scale))
    count  = min_count + round(density * (cap - min_count))

``reference_length`` defaults to borregas_ave (about 2.7 km), so that map keeps
the historical caps (6 NPC / 4 bicycles / 6 pedestrians). Square root because
agents cluster near the ego, so the useful count grows slower than the network.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

REFERENCE_LENGTH: float = 2729.0
"""Drivable CAR-lane length of borregas_ave in metres; its scale is 1."""


@dataclass(frozen=True)
class MapBudget:
    """
    Absolute per-type count caps of one map, derived from its size.

    :param float scale: Multiplier on each section's base ``max_count``.
    :param float drivable_length: CAR-lane length the scale came from, metres (0 when given directly).
    :param int junctions: Junction count of the map (informational).
    """
    scale: float = 1.0
    """Multiplier on each section's base ``max_count``."""
    drivable_length: float = 0.0
    """CAR-lane length the scale came from, in metres."""
    junctions: int = 0
    """Junction count of the map (informational)."""

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    @classmethod
    def from_size(cls, drivable_length: float, junctions: int = 0, *, reference_length: float = REFERENCE_LENGTH,
                  min_scale: float = 1.0, max_scale: float = 3.0) -> "MapBudget":
        """
        Budget of a map with ``drivable_length`` metres of CAR lanes.

        :returns: ``MapBudget`` with ``scale = clamp(sqrt(L / reference_length), min_scale, max_scale)``.
        :rtype: MapBudget
        """
        scale = sqrt(max(drivable_length, 0.0) / reference_length)
        return cls(min(max(scale, min_scale), max_scale), float(drivable_length), int(junctions))

    @classmethod
    def of(cls, context: Any, **kwargs: float) -> "MapBudget":
        """
        Budget of a ``ScenarioContext``: its own ``budget`` if set, otherwise from its CAR ``RouteSpace``.

        :param ScenarioContext context: Map context.
        :returns: The map's budget.
        :rtype: MapBudget
        """
        if getattr(context, "budget", None) is not None:
            return context.budget
        spaces: Mapping[Any, Any] = context.npc_spaces
        space = next((space for vt, space in spaces.items() if getattr(vt, "name", "") == "CAR"),
                     next(iter(spaces.values()), context.ego_space))
        length = getattr(space, "_budget_length", None)
        if length is None:
            length = float(sum(space._lengths.values()))
            space._budget_length = length
        junctions = len(getattr(space.service, "junction_table", {}) or {})
        return cls.from_size(length, junctions, **kwargs)

    def cap(self, base_max: int, min_count: int = 0) -> int:
        """
        Absolute cap for a section whose base cap is ``base_max``.

        :returns: ``max(min_count, round(base_max * scale))``.
        :rtype: int
        """
        return max(min_count, int(round(base_max * self.scale)))

    def rules(self, rules: Any) -> Any:
        """
        ``rules`` (a ``SectionRules``) with ``max_count`` replaced by this map's cap.
        """
        from dataclasses import replace
        return replace(rules, max_count=self.cap(rules.max_count, rules.min_count))


def count_for(density: float, min_count: int, max_count: int) -> int:
    """
    Section size of a relative ``density`` gene under ``min_count .. max_count``.

    :returns: ``min_count + round(clamp(density, 0, 1) * (max_count - min_count))``.
    :rtype: int
    """
    density = min(max(float(density), 0.0), 1.0)
    return min_count + int(round(density * (max_count - min_count)))


def density_for(count: int, min_count: int, max_count: int) -> float:
    """
    Relative density of a section of ``count`` agents under ``min_count .. max_count`` (inverse of ``count_for``).

    :rtype: float
    """
    if max_count <= min_count:
        return 0.0
    return min(max((count - min_count) / (max_count - min_count), 0.0), 1.0)
