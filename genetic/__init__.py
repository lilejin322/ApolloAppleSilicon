"""
Genetic operators for scenario chromosomes, one subpackage per agent type.

``common`` is shared by every agent type, ``route_operators`` by those on
lane routes (with ``LaneRouteConfig``), and ``profile_operators`` by every
replayed one; ``ad_agent`` holds the
``ADAgent`` operators, ``npc_agent`` the ``NPCVehicleAgent`` ones,
``bicycle`` the ``BicycleAgent`` ones and ``pedestrian`` the
``PedestrianAgent`` ones. Operators of different agent types
share names, so they are imported from their subpackage; only the entry
points are exported here. Scenario-level operators will wrap these.
"""

from typing import List

from . import ad_agent, bicycle, npc_agent, pedestrian, profile_operators, route_operators
from .ad_agent import ADOperatorConfig, cx_ad_agent, mut_ad_agent
from .bicycle import BicycleOperatorConfig, cx_bicycle, mut_bicycle
from .common import LaneRouteConfig, OperatorConfig, ReplayOperatorConfig
from .npc_agent import NPCOperatorConfig, cx_npc_agent, mut_npc_agent
from .pedestrian import PedestrianOperatorConfig, cx_pedestrian, mut_pedestrian

__all__: List[str] = [
    "ADOperatorConfig", "BicycleOperatorConfig", "LaneRouteConfig", "NPCOperatorConfig", "OperatorConfig",
    "ReplayOperatorConfig",
    "ad_agent", "bicycle", "cx_ad_agent", "cx_bicycle", "cx_npc_agent", "mut_ad_agent", "mut_bicycle",
    "mut_npc_agent", "npc_agent", "profile_operators", "route_operators",
    "PedestrianOperatorConfig", "cx_pedestrian", "mut_pedestrian", "pedestrian",
]
