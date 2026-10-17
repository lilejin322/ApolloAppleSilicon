"""
Scenario chromosomes: plain data and map-only logic, independent of Apollo I/O.

Agents hold a ``LaneRoute`` and build what their managers need through
``routing``: ``ADAgent`` its routing request, ``NPCVehicleAgent`` its
trajectory. Bicycles will follow.
"""

from .ad_agents import ADAgent
from .npc_agents import (
    VEHICLE_SPECS, NPCTrajectory, NPCVehicleAgent, SpeedPoint, VehicleSpec, VehicleState, VehicleType,
)
from .routing import (
    InvalidRoute, LaneRoute, PathSection, RouteLeg, RoutePose, RoutePoses, RouteRules, RouteSpace,
    Transition,
)

__all__ = [
    "ADAgent", "NPCVehicleAgent", "NPCTrajectory", "SpeedPoint", "VEHICLE_SPECS", "VehicleSpec",
    "VehicleState", "VehicleType", "InvalidRoute", "LaneRoute", "RouteLeg", "RoutePose", "RoutePoses",
    "RouteRules", "RouteSpace", "Transition", "PathSection",
]
