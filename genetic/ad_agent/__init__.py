"""
Route-aware crossover and mutation of ``ADAgent`` chromosomes.

``cx_ad_agent`` and ``mut_ad_agent`` recombine and mutate routes through the
agent's ``RouteSpace``; every child they return is valid and within the
configured route-length and start-time bounds. Individual operators only
propose candidates and require validation by their caller.
"""

from .common import ADOperatorConfig
from .crossover import cx_ad_agent, cx_bridge, cx_shared_lane, cx_start_time
from .mutation import (
    mut_ad_agent, mut_extend_or_truncate, mut_lane_change, mut_reroute, mut_s, mut_start_t,
)

__all__ = [
    "ADOperatorConfig", "cx_ad_agent", "cx_bridge", "cx_shared_lane", "cx_start_time",
    "mut_ad_agent", "mut_extend_or_truncate", "mut_lane_change", "mut_reroute", "mut_s", "mut_start_t",
]
