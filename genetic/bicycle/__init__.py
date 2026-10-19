"""
Crossover and mutation of ``BicycleAgent`` chromosomes.

Every gene varies, each with its own chance per call: route, turn waits,
rider type, speed profile, and start time. Route changes reuse
``genetic.route_operators`` on the bicycles' ``RouteSpace``, whose virtual
connectors give bike routes their branches, and repair turn waits with
``BicycleAgent.with_route``. Speed-profile and start-time operators are
shared with NPC vehicles (``genetic.profile_operators``).
"""

from .common import GENES, BicycleOperatorConfig, acceptable, rebalance_rider_types
from .crossover import (
    cx_bicycle, cx_bridge, cx_rider_type, cx_route_swap, cx_shared_lane, cx_drive_style, cx_speed_profile, cx_start_time,
    cx_turn_dwell,
)
from .mutation import (
    mut_bicycle, mut_extend_or_truncate, mut_reroute, mut_rider_type, mut_drive_style, mut_s, mut_speed, mut_speed_point,
    mut_start_t, mut_stop, mut_turn_dwell,
)

__all__ = [
    "GENES", "BicycleOperatorConfig", "acceptable", "rebalance_rider_types",
    "cx_bicycle", "cx_bridge", "cx_rider_type", "cx_route_swap", "cx_shared_lane", "cx_drive_style", "cx_speed_profile",
    "cx_start_time", "cx_turn_dwell",
    "mut_bicycle", "mut_extend_or_truncate", "mut_reroute", "mut_rider_type", "mut_s", "mut_drive_style", "mut_speed",
    "mut_speed_point", "mut_start_t", "mut_stop", "mut_turn_dwell",
]
