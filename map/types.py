"""Results returned by map queries."""

from dataclasses import dataclass

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
