"""
Analyze one record: replay it to every violation oracle and read off what
else the search scores, the ego's RSS margin and the states Planning went
through. ``RecordAnalyzer`` is the one entry point; the oracles judge
violations only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..map import MapService
from ..utils import ApolloVehicleConfig
from .oracles import (ComfortOracle, EStopOracle, JunctionLaneChangeOracle, LaneMarkingOracle, Oracle,
                      PlanningFailureOracle, PredictionNaNOracle, RedLightOracle, RssOracle, SpeedingOracle, Violation)
from .planning_states import planning_transitions
from .record import RecordReader
from .replay import PLANNING, Replay
from .rss.judge import Moment, Verdict

__all__ = ["RecordAnalysis", "RecordAnalyzer", "default_oracles"]


def default_oracles(map_service: MapService, vehicle: ApolloVehicleConfig) -> List[Oracle]:
    """
    A fresh set of every oracle, with default thresholds; the RSS oracle first.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :rtype: List[Oracle]
    """
    return [RssOracle(map_service, vehicle), RedLightOracle(map_service, vehicle), SpeedingOracle(map_service),
            LaneMarkingOracle(map_service, vehicle), ComfortOracle(), EStopOracle(),
            JunctionLaneChangeOracle(map_service), PlanningFailureOracle(), PredictionNaNOracle()]


@dataclass(frozen=True)
class RecordAnalysis:
    """
    What one record shows.

    :param violations: Every violation, in time order.
    :param float margin: The ego's RSS margin in metres (``RssOracle.margin``).
    :param str | None worst: What happened at the least margin; None when nothing lowered it.
    :param Moment | None worst_at: Where that was.
    :param verdicts: The RSS verdict of each collision.
    :param transitions: Each change of Planning's scenario or stage (``planning_transitions``).
    """

    violations: Tuple[Violation, ...]
    margin: float
    worst: str | None = None
    worst_at: Moment | None = None
    verdicts: Tuple[Verdict, ...] = ()
    transitions: Tuple[str, ...] = ()


class RecordAnalyzer:
    """
    Analyze records of one map and ego.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions, e.g. ``utils.get_vehicle_params``.
    :param descriptor_path: The protobuf descriptor of the Apollo that recorded them.
    """

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig, descriptor_path: str | Path) -> None:
        self.map_service = map_service
        self.vehicle = vehicle
        self.descriptor_path = descriptor_path

    def analyze(self, record: str | Path) -> RecordAnalysis:
        """
        Replay ``record`` to fresh ``default_oracles`` and read Planning's states.

        :param record: The record file, e.g. ``ApolloManager.record_file``.
        :rtype: RecordAnalysis
        """
        reader = RecordReader(record, descriptor_path=self.descriptor_path)
        oracles = default_oracles(self.map_service, self.vehicle)
        rss = oracles[0]
        violations = Replay(reader).run(oracles)
        planning = sorted(reader.messages({PLANNING}), key=lambda m: m.time)
        return RecordAnalysis(tuple(violations), rss.margin, rss.worst, rss.worst_at, tuple(rss.verdicts),
                              planning_transitions(m.message for m in planning))
