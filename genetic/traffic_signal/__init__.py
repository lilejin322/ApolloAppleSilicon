"""
GA operators of the traffic-signal chromosome (``scenario.traffic_signal``).

Only demand (veh/h/lane per phase) and offset (fraction of the cycle) are free;
every timing is recomputed by ``SignalTiming.plan``, so offspring are
Webster-consistent and as realistic as random individuals. Generation draws
each gene uniformly within its bounds. Mutation takes reflected (demand) or
wrapped (offset) symmetric steps, which keep that uniform distribution
stationary, so populations do not drift. Crossover swaps whole plans per
junction and never splits a phase group.
"""

from .operators import (
    MUTATION_ACTIONS, SignalOperatorConfig, cx_signal_section, mut_demand, mut_offset, mut_signal_section,
    random_signal_plan, random_signal_section,
)

__all__ = [
    "MUTATION_ACTIONS", "SignalOperatorConfig", "cx_signal_section", "mut_demand", "mut_offset",
    "mut_signal_section", "random_signal_plan", "random_signal_section",
]
