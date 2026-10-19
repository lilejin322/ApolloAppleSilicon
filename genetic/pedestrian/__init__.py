"""
Crossover and mutation of ``PedestrianAgent`` chromosomes.

Every gene varies, each with its own chance per call: walk, walker type,
speed profile (with walk style), and start time. Walk operators work on the
map's walk network (``scenario.walkways``): they join walks at a shared
sidewalk, move points along sidewalks, redraw excursions or their kinds, add
or remove one, and set where the walker turns back. Speed-profile and
start-time operators are shared with NPC vehicles and bicycles
(``genetic.profile_operators``).
"""

from .common import GENES, PedestrianOperatorConfig, acceptable
from .crossover import cx_pedestrian, cx_route_swap, cx_shared_sidewalk, cx_walk_style, cx_walker_type
from .mutation import (
    COUNT_ACTIONS, STOP_ACTIONS, TURN_BACK_ACTIONS, mut_count, mut_kind, mut_pedestrian, mut_redraw, mut_shift, mut_turn_back,
    mut_walk_stop, mut_walk_style, mut_walker_type,
)

__all__ = [
    "COUNT_ACTIONS", "GENES", "STOP_ACTIONS", "TURN_BACK_ACTIONS", "PedestrianOperatorConfig", "acceptable",
    "cx_pedestrian", "cx_route_swap", "cx_shared_sidewalk", "cx_walk_style", "cx_walker_type",
    "mut_count", "mut_kind", "mut_pedestrian", "mut_redraw", "mut_shift", "mut_turn_back", "mut_walk_stop", "mut_walk_style",
    "mut_walker_type",
]
