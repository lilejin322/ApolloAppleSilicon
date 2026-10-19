"""
Route-aware mutation of ADAgent chromosomes.

Unlike DoppelTest / ScenoRITA, which redraws start positions and times on a fixed route
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
    Replace an inner route stretch while preserving endpoints and start time.

    Alternatives preserve the stretch's lane-change count and obey
    ``config.max_detour``. The caller must validate the complete candidate.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Candidate-path limit, search distance, and maximum relative detour.
    :returns: A candidate, or None if the route has one lane or no alternative fits.
    :rtype: ADAgent | None
    """
    return on_route(agent, route_operators.mut_reroute(agent.route, space, rng, config))


def mut_extend_or_truncate(agent: ADAgent, space: RouteSpace, rng: Random,
                           config: ADOperatorConfig = ADOperatorConfig(),
                           direction: str | None = None) -> ADAgent | None:
    """
    Extend or truncate either end of the route, preserving the start time.

    Normally add or remove one to three lanes at the end or one to two at
    the start. Endpoint restrictions may require reaching a farther lane.
    Sample an allowed position on the changed endpoint's lane; the caller
    must validate the resulting route and its length.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Shared route-operator configuration; currently unused by this proposal.
    :param str | None direction: ``"extend"`` or ``"truncate"``; None draws
        either with equal probability. Pass one draw to reuse it across retries.
    :raises ValueError: If ``direction`` is neither an allowed value nor None.
    :returns: A candidate, or None if the chosen direction cannot produce one.
    :rtype: ADAgent | None
    """
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config, direction=direction))


def mut_lane_change(agent: ADAgent, space: RouteSpace, rng: Random,
                    config: ADOperatorConfig = ADOperatorConfig(),
                    acceptance: float | None = None) -> ADAgent | None:
    """
    Propose adding or removing lane changes while preserving the start time.

    The shared route operator reconnects to the existing route or continues
    forward when reconnection is unavailable. The caller must validate the
    resulting route and its bounds.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Path-search limits and acceptance odds for additional lane changes.
    :param float | None acceptance: Uniform draw in [0, 1), reused across
        retries; None draws locally. A proposal adding ``k`` lane changes
        survives only if this draw is below ``config.lane_change_odds ** k``.
    :returns: A candidate, or None if no proposal survives construction and acceptance.
    :rtype: ADAgent | None
    """
    return on_route(agent, route_operators.mut_lane_change(agent.route, space, rng, config, acceptance=acceptance))


def mut_s(agent: ADAgent, space: RouteSpace, rng: Random,
          config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Move either or both endpoint positions by Gaussian steps, preserving start time.

    Steps use ``config.sigma_s`` in metres and are clamped to their lanes.
    With endpoint restrictions, snap to the nearest allowed position.
    The caller must check route validity and length; a proposal may be unchanged.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Endpoint-step standard deviation ``sigma_s``, in metres.
    :returns: A candidate, or None if a moved endpoint's lane has no allowed position.
    :rtype: ADAgent | None
    """
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config))


def mut_start_t(agent: ADAgent, rng: Random,
                config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Move the start time by a Gaussian step, preserving the route.

    Use ``config.sigma_t`` in seconds and clamp to ``[0, config.max_start_t]``.
    The result may equal the input; route validity and length are not checked.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param Random rng: Random source used to sample proposals; reuse a seeded instance for reproducibility.
    :param ADOperatorConfig config: Time-step standard deviation ``sigma_t`` and upper bound ``max_start_t``, in seconds.
    :returns: An agent with the proposed start time.
    :rtype: ADAgent
    """
    return replace(agent, start_t=gauss_step(rng, agent.start_t, config.sigma_t, config.max_start_t))


def mut_ad_agent(agent: ADAgent, space: RouteSpace, rng: Random,
                 config: ADOperatorConfig = ADOperatorConfig()) -> ADAgent:
    """
    Return an accepted mutant selected using ``config.mutation_weights``.

    Draw positive-weight operators without replacement, allowing up to
    ``config.attempts`` tries each. Accept only changed candidates with valid
    routes and driven lengths and start times within ``config`` bounds.
    Reuse one lane-change acceptance draw and one extend/truncate direction
    throughout the mutation event, including retries.

    :param ADAgent agent: Agent from which to construct the mutation candidate.
    :param RouteSpace space: Route space defining legal ego routes.
    :param Random rng: Random source shared by operator selection and proposals.
    :param ADOperatorConfig config: Operator weights, retry limits, and bounds.
    :raises ValueError: If all proposals fail and the original agent is unacceptable.
    :returns: An accepted mutant, or the unchanged original if it remains acceptable.
    :rtype: ADAgent
    """
    acceptance = rng.random()  # lane-change acceptance draw for this mutation event, kept over retries
    direction = rng.choice(("extend", "truncate"))  # extend/truncate direction for this event, kept over retries
    operators = {
        "reroute": lambda: mut_reroute(agent, space, rng, config),
        "extend_or_truncate": lambda: mut_extend_or_truncate(agent, space, rng, config, direction),
        "lane_change": lambda: mut_lane_change(agent, space, rng, config, acceptance),
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
