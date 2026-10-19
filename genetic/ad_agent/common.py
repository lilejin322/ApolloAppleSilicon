"""
Shared pieces of route-aware ADAgent crossover and mutation.

Operators change a route's lanes, transitions, and positions through the
agent's ``RouteSpace``. ``on_route`` constructs proposals; ``acceptable``
checks route validity, driven-length bounds, and the start-time limit.
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar, Mapping, Tuple

from ...scenario import ADAgent, LaneRoute, RouteSpace
from ..common import OperatorConfig


@dataclass(frozen=True)
class ADOperatorConfig(OperatorConfig):
    """
    Bounds, weights, and step sizes for ADAgent crossover and mutation.

    Bounds and step sizes are inherited from ``OperatorConfig``. Weight keys
    must belong to ``CROSSOVERS`` or ``MUTATIONS``; omitted or zero-weight
    operators are disabled. Positive weights control selection order, with
    up to ``attempts`` tries per selected operator.
    """

    CROSSOVERS: ClassVar[Tuple[str, ...]] = ("shared_lane", "bridge", "start_time")
    """Supported crossover names used as keys in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = ("reroute", "extend_or_truncate", "lane_change", "s", "start_t")
    """Supported mutation names used as keys in ``mutation_weights``."""

    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "start_time": 1.0,
    })
    """Relative crossover weights; omitted or zero-weight operators are disabled."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "start_t": 1.0,
    })
    """Relative mutation weights; omitted or zero-weight operators are disabled."""


def on_route(agent: ADAgent, route: LaneRoute | None) -> ADAgent | None:
    """
    Return a copy of ``agent`` with ``route``, or None if no route was proposed.

    The start time is preserved; route validity and operator bounds are not
    checked here. Use ``acceptable`` to validate the candidate.

    :param ADAgent agent: Source agent whose start time is preserved.
    :param LaneRoute | None route: Proposed lane route, or None when the route operator produced no candidate.
    :returns: A copy with the proposed route, or None if ``route`` is None.
    :rtype: ADAgent | None
    """
    return None if route is None else replace(agent, route=route)


def acceptable(space: RouteSpace, agent: ADAgent, config: ADOperatorConfig) -> bool:
    """
    Check route validity, inclusive driven-length bounds, and the start-time limit.

    Lengths are in metres and ``max_start_t`` is in seconds. ``ADAgent``
    already enforces a finite, nonnegative start time at construction.

    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param ADAgent agent: Candidate whose route, driven length, and start time are checked.
    :param ADOperatorConfig config: Inclusive minimum/maximum driven length and maximum start time.
    :returns: True if all route, length, and time checks pass; otherwise False.
    :rtype: bool
    """
    return (agent.start_t <= config.max_start_t and space.is_valid(agent.route)
            and config.min_length <= space.length(agent.route) <= config.max_length)
