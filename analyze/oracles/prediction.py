"""
Prediction: trajectories Planning cannot use.
"""

from math import isnan
from typing import Set

from google.protobuf.message import Message

from ..replay import PREDICTION, World
from .base import Oracle

__all__ = ["PredictionNaNOracle"]


class PredictionNaNOracle(Oracle):
    """
    Report each obstacle Prediction predicted a trajectory with NaN points
    for, once, at the first such prediction.

    Planning takes no decision for such an obstacle and drives as if it were
    not there. Our obstacles once came without ``polygon_point``, from which
    Prediction draws the semantic map of its pedestrian model, and every
    pedestrian near the ego was predicted as NaN; this oracle keeps such a gap
    from going unnoticed. It judges Prediction, not the ego's driving, so the
    search does not count it against the ego.
    """

    kind = "prediction_nan"
    channels = frozenset({PREDICTION})

    def __init__(self) -> None:
        super().__init__()
        self._reported: Set[int] = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        for predicted in message.prediction_obstacle:
            obstacle = predicted.perception_obstacle
            if obstacle.id in self._reported or not any(
                    isnan(point.path_point.x) or isnan(point.path_point.y)
                    for trajectory in predicted.trajectory for point in trajectory.trajectory_point):
                continue
            self._reported.add(obstacle.id)
            name = f"{obstacle.DESCRIPTOR.fields_by_name['type'].enum_type.values_by_number[obstacle.type].name} " \
                   f"{obstacle.id}"
            self.report(world, name, f"Prediction predicted {name} as a NaN trajectory; Planning ignores it")
