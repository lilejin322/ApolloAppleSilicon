"""
Scenario chromosomes: plain data and map-only logic, independent of Apollo I/O.

Agents hold a ``LaneRoute`` and build what their managers need through
``routing``: ``ADAgent`` its routing request, ``NPCVehicleAgent`` and
``BicycleAgent`` their trajectories through ``replay``. Bicycles also ride
the virtual connectors of ``bike_connectors``. ``PedestrianAgent`` holds a
``WalkRoute`` on the map's walk network (``walkways``) instead and replays its
trajectory through ``replay`` too.
"""

from .ad_agent import ADAgent
from .bicycle import RIDER_SPECS, BicycleAgent, RiderSpec, RiderType, random_rider_type
from .bike_connectors import ConnectorKind
from .npc_agent import VEHICLE_SPECS, NPCVehicleAgent, VehicleSpec, VehicleType, random_vehicle_type
from .pedestrian import WALKER_SPECS, PedestrianAgent, WalkerSpec, WalkerType, random_walker_type
from .replay import AgentState, ReplayTrajectory, SpeedPoint
from .routing import (
    InvalidRoute, LaneRoute, PathSection, RouteLeg, RoutePose, RoutePoses, RouteRules, RouteSpace,
    Transition, VirtualLane,
)
from .signal_program import SignalProgram
from .walkways import CurbPoint, Excursion, ExcursionKind, WalkNetwork, WalkRoute, walk_network

__all__ = [
    "ADAgent", "SignalProgram", "AgentState", "BicycleAgent", "ConnectorKind", "RIDER_SPECS", "RiderSpec", "RiderType",
    "NPCVehicleAgent", "ReplayTrajectory", "SpeedPoint", "VEHICLE_SPECS",
    "VehicleSpec", "VehicleType", "InvalidRoute", "LaneRoute", "RouteLeg", "RoutePose", "RoutePoses",
    "RouteRules", "RouteSpace", "Transition", "PathSection", "VirtualLane", "random_rider_type",
    "random_vehicle_type", "CurbPoint", "Excursion", "ExcursionKind", "PedestrianAgent", "WALKER_SPECS",
    "WalkNetwork", "WalkRoute", "WalkerSpec", "WalkerType", "random_walker_type", "walk_network",
]
