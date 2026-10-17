"""
Results returned by map queries.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp
"""

from typing import Tuple
from dataclasses import dataclass, field
from shapely.geometry import LineString


@dataclass
class PositionEstimate:
    lane_id: str | None = None
    s: float | None = None


@dataclass(frozen=True)
class LaneSignalOverlap:
    """A signal association and its longitudinal interval on one lane.

    Missing protobuf coordinates remain None, rather than becoming zero.
    """

    overlap_id: str
    lane_id: str
    signal_id: str
    start_s: float | None
    end_s: float | None


@dataclass(frozen=True)
class LaneStopLine:
    """A signal stop-line candidate matched to explicit lane overlap intervals.

    Geometry is the complete original stop line, not clipped to the lane.
    distance is the minimum distance to the supporting lane intervals in metres.
    """

    lane_id: str
    signal_id: str
    stop_line_index: int
    geometry: LineString
    overlaps: tuple[LaneSignalOverlap, ...]
    distance: float


PROHIBITING_MARKINGS = frozenset({"SOLID_WHITE", "SOLID_YELLOW", "DOUBLE_YELLOW", "CURB"})
"""``LaneBoundaryType.Type`` names that a vehicle may not cross."""


@dataclass(frozen=True, order=True)
class LaneMarking:
    """
    A stretch of one lane boundary whose markings do not change.

    Adjacent lanes describe a shared line separately, each as its own boundary.
    Markings order by lane, side, and position; geometry is not compared.
    """

    lane_id: str
    """The lane Id"""
    side: str
    """``"left"`` or ``"right"`` of the lane, in its driving direction."""
    start_s: float
    """Start along the boundary curve, in metres from its first point."""
    end_s: float
    """End along the boundary curve, in metres from its first point."""
    types: Tuple[str, ...]
    """``LaneBoundaryType.Type`` names, such as ``("SOLID_WHITE",)``."""
    virtual: bool
    """Whether the boundary does not exist in the real world, as inside junctions."""
    geometry: LineString = field(compare=False, repr=False)

    @property
    def prohibits_crossing(self) -> bool:
        """
        Whether a vehicle may not cross this marking: a real solid line, double
        yellow line, or curb. Virtual boundaries and dotted or UNKNOWN markings
        do not prohibit crossing; with several types, any prohibiting one does.

        :rtype: bool
        """
        return not self.virtual and not PROHIBITING_MARKINGS.isdisjoint(self.types)
