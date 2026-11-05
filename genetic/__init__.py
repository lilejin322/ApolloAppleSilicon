"""
Genetic operators for scenario chromosomes, in three layers.

Bottom layer, one subpackage per agent type: ``ad_agent`` holds the ``ADAgent``
operators, ``npc_agent`` the ``NPCVehicleAgent`` ones, ``bicycle`` the
``BicycleAgent`` ones and ``pedestrian`` the ``PedestrianAgent`` ones.
``common`` is shared by every agent type, ``route_operators`` by those on lane
routes (with ``LaneRouteConfig``) and ``profile_operators`` by every replayed
one. ``traffic_signal`` varies the signal-plan gene.

Middle layer: ``section`` generates, crosses and mutates one section of agents
through a ``SectionKind`` per type, drawing new agents toward the ego
(``encounter``).

Top layer: ``scenario_chromosome`` does the same for whole scenarios (ego,
sections and signals); the search drives it through ``ScenarioKinds``,
``random_scenario``, ``cx_scenario``, ``mut_scenario`` and ``rebalance_scenarios``.

Operators of different agent types share names, so they are imported from
their subpackage; only the configs, the per-type entry points and the
scenario-level entry points are exported here.
"""

from typing import List

from . import (
    ad_agent, bicycle, common, encounter, npc_agent, pedestrian, profile_operators, route_operators,
    scenario_chromosome, section, traffic_signal,
)
from .ad_agent import ADOperatorConfig, cx_ad_agent, mut_ad_agent
from .bicycle import BicycleOperatorConfig, cx_bicycle, mut_bicycle
from .common import LaneRouteConfig, OperatorConfig, ReplayOperatorConfig
from .npc_agent import NPCOperatorConfig, cx_npc_agent, mut_npc_agent
from .pedestrian import PedestrianOperatorConfig, cx_pedestrian, mut_pedestrian
from .scenario_chromosome import (
    ScenarioKinds, ScenarioOperatorConfig, cx_scenario, mut_scenario, random_scenario, rebalance_scenarios,
)
from .section import SectionOperatorConfig
from .traffic_signal import SignalOperatorConfig

__all__: List[str] = [
    "ADOperatorConfig", "BicycleOperatorConfig", "LaneRouteConfig", "NPCOperatorConfig", "OperatorConfig",
    "PedestrianOperatorConfig", "ReplayOperatorConfig", "ScenarioKinds", "ScenarioOperatorConfig",
    "SectionOperatorConfig", "SignalOperatorConfig",
    "cx_ad_agent", "cx_bicycle", "cx_npc_agent", "cx_pedestrian", "cx_scenario",
    "mut_ad_agent", "mut_bicycle", "mut_npc_agent", "mut_pedestrian", "mut_scenario",
    "random_scenario", "rebalance_scenarios",
    "ad_agent", "bicycle", "common", "encounter", "npc_agent", "pedestrian", "profile_operators", "route_operators",
    "scenario_chromosome", "section", "traffic_signal",
]
