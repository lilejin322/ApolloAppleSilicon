"""
Route-aware crossover of ADAgent chromosomes.

Unlike META-V2V, which only exchanges start positions or times between
agents with identical routes, these operators recombine the routes
themselves. Each operator returns one child of ``first`` and ``second``, or
None when it does not apply; ``cx_ad_agent`` picks operators by weight and
returns two children.
Pure lane-route recombination is shared with NPCs through ``route_operators``.
"""

from dataclasses import replace
from random import Random

from ...scenario import ADAgent, RouteSpace
from .. import route_operators
from ..common import apply_weighted
from .common import ADOperatorConfig, acceptable, on_route


def cx_shared_lane(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random) -> ADAgent | None:
    """
    Join ``first``'s route up to a lane both routes use with ``second``'s route from it.

    The child starts like ``first`` (start s and start time) and ends like ``second``.

    :returns: The child, or None if the routes share no lane that joins without repeats.
    """
    return on_route(first, route_operators.cx_shared_lane(first.route, second.route, space, rng))


def cx_bridge(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random,
              config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    A lane of ``first`` and a lane of ``second`` reachable from it within
    ``config.max_length`` are chosen; one of the cheapest lane paths between
    them becomes the bridge. Routes need not meet.

    :returns: The child, or None if no lane pair is reachable or the bridge
        would repeat a lane.
    """
    return on_route(first, route_operators.cx_bridge(first.route, second.route, space, rng, config))


def cx_start_time(first: ADAgent, second: ADAgent) -> ADAgent | None:
    """
    Give ``first`` the start time of ``second``.

    :returns: The child, or None if the start times are equal.
    """
    if first.start_t == second.start_t:
        return None
    return replace(first, start_t=second.start_t)


def cx_ad_agent(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random,
                config: ADOperatorConfig = ADOperatorConfig()) -> tuple[ADAgent, ADAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each child comes from an operator drawn by ``config.crossover_weights``;
    inapplicable or failing operators are skipped. A child must be valid,
    within the length bounds, and differ from both parents; when no operator
    succeeds, the leading parent is returned unchanged.

    :rtype: tuple[ADAgent, ADAgent]
    """
    def child_of(lead: ADAgent, other: ADAgent) -> ADAgent:
        operators = {
            "shared_lane": lambda: cx_shared_lane(lead, other, space, rng),
            "bridge": lambda: cx_bridge(lead, other, space, rng, config),
            "start_time": lambda: cx_start_time(lead, other),
        }
        child = apply_weighted(
            operators, config.crossover_weights, rng, config.attempts,
            lambda candidate: candidate not in (first, second) and acceptable(space, candidate, config),
        )
        if child is None and not acceptable(space, lead, config):
            raise ValueError("no acceptable child: the unchanged ADS parent violates the operator constraints")
        return lead if child is None else child

    return child_of(first, second), child_of(second, first)
