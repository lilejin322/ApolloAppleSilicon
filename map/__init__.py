"""
Apollo map loading and queries, backed by one explicitly configured service.

Re-implemented using shapely STRtree rather than the Apollo's original KDTree, will be faster.
"""

from .map_service import MapService
from .types import LaneSignalOverlap, LaneStopLine, PositionEstimate

__all__ = ["MapService", "PositionEstimate", "LaneSignalOverlap", "LaneStopLine"]
