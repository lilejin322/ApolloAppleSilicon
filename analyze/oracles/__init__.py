"""
Oracles: each follows a record as it is replayed and reports the violations of its kind.

``Oracle`` is the base class; an oracle names the channels it reads, sees each
message with the ``World`` at that instant (``on_message``), may judge what
needs hindsight at the end (``finish``), and reports ``Violation`` values.
``Episodes`` groups a condition that lasts into one report.

=============================  ====================================================================
Oracle                         ``Violation.kind``
=============================  ====================================================================
``RssOracle``                  ``collision``, ``collision_preventable``, ``collision_by_other``
``RedLightOracle``             ``red_light``
``SpeedingOracle``             ``speeding``
``LaneMarkingOracle``          ``lane_marking``
``ComfortOracle``              ``comfort``
``EStopOracle``                ``estop``
``JunctionLaneChangeOracle``   ``junction_lane_change``
``PlanningFailureOracle``      ``planning_error``, ``planning_gap``
``PredictionNaNOracle``        ``prediction_nan``
``MispredictionOracle``        ``misprediction``
=============================  ====================================================================

``analyzer.default_oracles`` builds one of each.
"""

from .base import Episode, Episodes, Oracle, Violation
from .collision import RssOracle
from .comfort import ComfortOracle
from .lane_marking import LaneMarkingOracle
from .planning import EStopOracle, JunctionLaneChangeOracle, PlanningFailureOracle
from .prediction import MispredictionOracle, PredictionNaNOracle
from .red_light import RedLightOracle
from .speeding import SpeedingOracle

__all__ = ["ComfortOracle", "EStopOracle", "Episode", "Episodes", "JunctionLaneChangeOracle", "LaneMarkingOracle",
           "MispredictionOracle", "Oracle", "PlanningFailureOracle", "PredictionNaNOracle", "RedLightOracle",
           "RssOracle", "SpeedingOracle", "Violation"]
