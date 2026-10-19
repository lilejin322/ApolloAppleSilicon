"""
Scenario chromosomes: plain data and map-only logic, independent of Apollo I/O.

Agents hold a ``LaneRoute`` and build what their managers need through
``routing``: ``ADAgent`` its routing request, ``NPCVehicleAgent`` and
``BicycleAgent`` their trajectories through ``replay``. Bicycles also ride
the virtual connectors of ``bike_connectors``.
"""

from .ad_agents import ADAgent
from .bicycle import RIDER_SPECS, BicycleAgent, RiderSpec, RiderType
from .bike_connectors import ConnectorKind
from .npc_agents import VEHICLE_SPECS, NPCVehicleAgent, VehicleSpec, VehicleType
from .replay import AgentState, ReplayTrajectory, SpeedPoint
from .routing import (
    InvalidRoute, LaneRoute, PathSection, RouteLeg, RoutePose, RoutePoses, RouteRules, RouteSpace,
    Transition, VirtualLane,
)

__all__ = [
    "ADAgent", "AgentState", "BicycleAgent", "ConnectorKind", "RIDER_SPECS", "RiderSpec", "RiderType",
    "NPCVehicleAgent", "ReplayTrajectory", "SpeedPoint", "VEHICLE_SPECS",
    "VehicleSpec", "VehicleType", "InvalidRoute", "LaneRoute", "RouteLeg", "RoutePose", "RoutePoses",
    "RouteRules", "RouteSpace", "Transition", "PathSection", "VirtualLane",
]
