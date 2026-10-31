"""
Judge one record: replay it to every oracle and collect the violations.
"""

from pathlib import Path
from typing import List, Sequence

from ..map import MapService
from ..utils import ApolloVehicleConfig
from .oracles import (CollisionOracle, ComfortOracle, EStopOracle, JunctionLaneChangeOracle, LaneMarkingOracle,
                      Oracle, PlanningFailureOracle, RedLightOracle, SpeedingOracle, Violation)
from .record import RecordReader
from .replay import Replay

__all__ = ["analyze", "default_oracles"]


def default_oracles(map_service: MapService, vehicle: ApolloVehicleConfig) -> List[Oracle]:
    """
    A fresh set of every oracle, with default thresholds.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :rtype: List[Oracle]
    """
    return [CollisionOracle(vehicle), RedLightOracle(map_service, vehicle), SpeedingOracle(map_service),
            LaneMarkingOracle(map_service, vehicle), ComfortOracle(), EStopOracle(),
            JunctionLaneChangeOracle(map_service), PlanningFailureOracle()]


def analyze(record: str | Path, *, map_service: MapService, vehicle: ApolloVehicleConfig,
            descriptor_path: str | Path, oracles: Sequence[Oracle] | None = None) -> List[Violation]:
    """
    Replay a record and report what the ADS did wrong.

    :param record: The record file, e.g. ``ApolloManager.record_file``.
    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions, e.g. ``utils.get_vehicle_params``.
    :param descriptor_path: The protobuf descriptor of the Apollo that recorded it.
    :param oracles: Fresh oracles; ``default_oracles`` by default.
    :returns: Every violation, in time order.
    :rtype: List[Violation]
    """
    reader = RecordReader(record, descriptor_path=descriptor_path)
    return Replay(reader).run(oracles if oracles is not None else default_oracles(map_service, vehicle))
