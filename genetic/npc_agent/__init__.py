"""
Crossover and mutation of ``NPCVehicleAgent`` chromosomes.

Every gene varies, each with its own chance per call: route, lane-change
positions, vehicle type, speed profile, and start time. Route changes reuse
``genetic.route_operators`` and repair lane-change positions with
``NPCVehicleAgent.with_route``; children are checked on their own vehicle
type's ``RouteSpace``.
"""

from .common import GENES, NPCOperatorConfig, Spaces, acceptable
from .crossover import (
    cx_bridge, cx_lane_change_s, cx_npc_agent, cx_route_swap, cx_shared_lane, cx_speed_profile, cx_start_time,
    cx_vehicle_type,
)
from .mutation import (
    mut_extend_or_truncate, mut_lane_change, mut_lane_change_s, mut_npc_agent, mut_reroute, mut_s, mut_speed,
    mut_speed_point, mut_start_t, mut_stop, mut_vehicle_type,
)

__all__ = [
    "GENES", "NPCOperatorConfig", "Spaces", "acceptable",
    "cx_bridge", "cx_lane_change_s", "cx_npc_agent", "cx_route_swap", "cx_shared_lane", "cx_speed_profile",
    "cx_start_time", "cx_vehicle_type",
    "mut_extend_or_truncate", "mut_lane_change", "mut_lane_change_s", "mut_npc_agent", "mut_reroute", "mut_s",
    "mut_speed", "mut_speed_point", "mut_start_t", "mut_stop", "mut_vehicle_type",
]
