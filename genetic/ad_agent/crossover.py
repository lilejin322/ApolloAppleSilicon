"""
Route-aware crossover of ADAgent chromosomes.

Unlike DoppelTest / ScenoRITA which only exchanges start positions or times between
agents with identical routes, these operators recombine the routes
themselves. Each operator returns one child of ``first`` and ``second``, or
None when it does not apply; ``cx_ad_agent`` picks operators by weight and
returns two children.
Pure lane-route recombination is shared with NPCs through ``route_operators``.
"""

from dataclasses import replace
from random import Random
from typing import Tuple

from ...scenario import ADAgent, RouteSpace
from .. import route_operators
from ..common import apply_weighted
from .common import ADOperatorConfig, acceptable, on_route


def cx_shared_lane(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random) -> ADAgent | None:
    """
    Join the first parent's prefix to the second parent's suffix at a shared lane.

    Keep ``first.start_t``, ``first.route.start_s``, and ``second.route.end_s``.
    Prefer a join near a target length drawn between the parents' lengths.
    The caller must check the resulting agent against operator bounds.

    :param ADAgent first: Leading parent supplying the route prefix and initial start time.
    :param ADAgent second: Other parent supplying the route suffix or replacement start time.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :returns: A candidate, or None if no shared-lane join avoids repeated lanes.
    :rtype: ADAgent | None
    """
    return on_route(first, route_operators.cx_shared_lane(first.route, second.route, space, rng))


def cx_bridge(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random,
              config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent | None:
    """
    Join the first parent's prefix to the second parent's suffix through a lane path.

    Keep the first parent's start position and time and the second parent's
    end position. ``space`` supplies legal lane paths; ``config`` controls
    the search budget and optional parent-length restriction. Prefer a bridge
    near a target length drawn from the parents' length range with slack.
    The caller must validate the complete candidate and its bounds.

    :param ADAgent first: Leading parent supplying the route prefix and initial start time.
    :param ADAgent second: Other parent supplying the route suffix or replacement start time.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Bridge search limits and optional restriction to the parents' length range.
    :returns: A candidate, or None if no bridge satisfies the search constraints.
    :rtype: ADAgent | None
    """
    return on_route(first, route_operators.cx_bridge(first.route, second.route, space, rng, config))


def cx_start_time(first: ADAgent, second: ADAgent) -> ADAgent | None:
    """
    Give ``first`` the start time of ``second``.

    :param ADAgent first: Parent whose route is preserved.
    :param ADAgent second: Parent donating its start time, in scenario seconds.
    :returns: The child, or None if the start times are equal.
    :rtype: ADAgent | None
    """
    if first.start_t == second.start_t:
        return None
    return replace(first, start_t=second.start_t)


def cx_ad_agent(first: ADAgent, second: ADAgent, space: RouteSpace, rng: Random,
                config: ADOperatorConfig = ADOperatorConfig()) -> Tuple[ADAgent, ADAgent]:
    """
    Return two accepted children, led by ``first`` and ``second`` respectively.

    Draw positive-weight operators without replacement using ``rng``, with
    up to ``config.attempts`` tries each. Accept only candidates different
    from both parents whose routes are valid and whose driven lengths and
    start times satisfy ``config``. If all proposals fail, return the leading
    parent unchanged only if it satisfies those same bounds.

    :param ADAgent first: Leading parent supplying the route prefix and initial start time.
    :param ADAgent second: Other parent supplying the route suffix or replacement start time.
    :param RouteSpace space: Route space defining legal ego routes.
    :param Random rng: Random source shared by operator selection and proposals.
    :param ADOperatorConfig config: Operator weights, retry limits, and bounds.
    :raises ValueError: If no proposal is accepted and the leading parent is invalid.
    :returns: Children in the same leading-parent order as the inputs.
    :rtype: Tuple[ADAgent, ADAgent]
    """
    def child_of(lead: ADAgent, other: ADAgent) -> ADAgent:
        """
        Return an accepted child of ``lead``, falling back to ``lead`` only if acceptable.

        :param ADAgent lead: Parent supplying the base agent and the unchanged fallback.
        :param ADAgent other: Parent supplying the route suffix or replacement start time.
        :returns: An accepted child, or ``lead`` unchanged if it remains acceptable.
        :raises ValueError: If no candidate is accepted and ``lead`` is unacceptable.
        :rtype: ADAgent
        """
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
