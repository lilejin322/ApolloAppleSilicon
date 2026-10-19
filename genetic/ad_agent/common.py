"""
Shared pieces of route-aware ADAgent crossover and mutation.

Operators change a route's lanes, transitions, and positions through the
agent's ``RouteSpace``. A child is accepted only when the space validates its
route and its driven length is within the configured bounds, so every child
can be sent to Apollo unambiguously.
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar, Mapping

from ...scenario import ADAgent, LaneRoute, RouteSpace
from ..common import OperatorConfig


@dataclass(frozen=True)
class ADOperatorConfig(OperatorConfig):
    """
    Bounds, weights, and step sizes for ADAgent crossover and mutation.

    Fields are those of ``OperatorConfig``; the weights default to the AD operators below.
    """

    CROSSOVERS: ClassVar[tuple[str, ...]] = ("shared_lane", "bridge", "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = ("reroute", "extend_or_truncate", "lane_change", "s", "start_t")

    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "start_t": 1.0,
    })


def on_route(agent: ADAgent, route: LaneRoute | None) -> ADAgent | None:
    """``agent`` moved to a proposed ``route``, or None when nothing was proposed."""
    return None if route is None else replace(agent, route=route)


def acceptable(space: RouteSpace, agent: ADAgent, config: ADOperatorConfig) -> bool:
    """Whether the agent's route is valid and within the length and time bounds."""
    return (agent.start_t <= config.max_start_t and space.is_valid(agent.route)
            and config.min_length <= space.length(agent.route) <= config.max_length)
