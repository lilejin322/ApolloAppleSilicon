"""
Route-aware mutation of ADAgent chromosomes.

Unlike META-V2V, which redraws start positions and times on a fixed route
or replaces the agent, these operators also change the route: an inner
stretch, its ends, or its lane changes. Each operator returns a mutant, or
None when it does not apply; ``mut_ad_agent`` picks operators by weight.
Pure lane-route changes are shared with NPCs through ``route_operators``.
"""

from dataclasses import replace
from random import Random

from ...scenario import ADAgent, RouteSpace
from .. import route_operators
from ..common import apply_weighted, gauss_step
from .common import ADOperatorConfig, acceptable, on_route


def mut_reroute(agent: ADAgent, space: RouteSpace, rng: Random,
                config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :returns: The mutant, or None if the route has one lane or no alternative fits.
    """
    return on_route(agent, route_operators.mut_reroute(agent.route, space, rng, config))


def mut_extend_or_truncate(agent: ADAgent, space: RouteSpace, rng: Random,
                           config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent | None:
    """
    Extend or truncate the route at its start or end.

    Extensions add one to three lanes after the end, or one or two before
    the start, along any legal step; truncations drop lanes from either end.
    The new end lane gets a random s.

    :returns: The mutant, or None if no change applies.
    """
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config))


def mut_lane_change(agent: ADAgent, space: RouteSpace, rng: Random,
                    config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent | None:
    """
    Insert a lane change into the route, or remove one.

    An insertion moves to a neighbouring lane and rejoins the route later
    through a lane path, or ends on the neighbouring lane. A removal replaces
    a stretch containing a lane change with a path that has fewer, or ends
    the route before the lane change.

    :returns: The mutant, or None if the route can neither gain nor lose a lane change.
    """
    return on_route(agent, route_operators.mut_lane_change(agent.route, space, rng, config))


def mut_s(agent: ADAgent, space: RouteSpace, rng: Random,
          config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    :rtype: ADAgent
    """
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config))


def mut_start_t(agent: ADAgent, rng: Random, config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Move the start time by a normal step, within ``[0, config.max_start_t]``.

    :rtype: ADAgent
    """
    return replace(agent, start_t=gauss_step(rng, agent.start_t, config.sigma_t, config.max_start_t))


def mut_ad_agent(agent: ADAgent, space: RouteSpace, rng: Random,
                 config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Return a mutant from an operator drawn by ``config.mutation_weights``.

    Inapplicable or failing operators are skipped. The mutant must be valid,
    within the length bounds, and differ from ``agent``; when no operator
    succeeds, ``agent`` is returned unchanged.

    :rtype: ADAgent
    """
    operators = {
        "reroute": lambda: mut_reroute(agent, space, rng, config),
        "extend_or_truncate": lambda: mut_extend_or_truncate(agent, space, rng, config),
        "lane_change": lambda: mut_lane_change(agent, space, rng, config),
        "s": lambda: mut_s(agent, space, rng, config),
        "start_t": lambda: mut_start_t(agent, rng, config),
    }
    mutant = apply_weighted(
        operators, config.mutation_weights, rng, config.attempts,
        lambda candidate: candidate != agent and acceptable(space, candidate, config),
    )
    if mutant is None and not acceptable(space, agent, config):
        raise ValueError("no acceptable child: the unchanged ADS parent violates the operator constraints")
    return agent if mutant is None else mutant
