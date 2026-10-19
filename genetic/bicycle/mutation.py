"""
Mutation of BicycleAgent chromosomes.

Each operator returns a mutant that changes one gene, or None when it does
not apply. ``mut_bicycle`` mutates each gene with its rate. Bicycles never
change lanes, so there is no lane-change mutation.
"""

from dataclasses import replace
from random import Random

from ...scenario import BicycleAgent, RiderType, RouteSpace
from .. import route_operators
from ..common import gauss_step, vary
from ..profile_operators import mut_speed, mut_speed_point, mut_start_t, mut_stop
from .common import BicycleOperatorConfig, acceptable, on_route, retyped


def mut_reroute(agent: BicycleAgent, space: RouteSpace, rng: Random,
                config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :returns: The mutant, or None if no alternative fits.
    """
    return on_route(agent, route_operators.mut_reroute(agent.route, space, rng, config), space, rng, config)


def mut_extend_or_truncate(agent: BicycleAgent, space: RouteSpace, rng: Random,
                           config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Extend or truncate the route at its start or end.

    :returns: The mutant, or None if no change applies.
    """
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config),
                    space, rng, config)


def mut_s(agent: BicycleAgent, space: RouteSpace, rng: Random,
          config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    :returns: The mutant, or None if the new ends are invalid.
    """
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config), space, rng, config)


def mut_turn_dwell(agent: BicycleAgent, rng: Random,
                   config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Move one turn wait by a normal step (``config.sigma_turn_dwell``), within
    ``[0, config.max_turn_dwell]``.

    :returns: The mutant, or None if the route has no pivot.
    """
    if not agent.turn_dwell:
        return None
    index = rng.randrange(len(agent.turn_dwell))
    wait = gauss_step(rng, agent.turn_dwell[index], config.sigma_turn_dwell, config.max_turn_dwell)
    return replace(agent, turn_dwell=agent.turn_dwell[:index] + (wait,) + agent.turn_dwell[index + 1:])


def mut_rider_type(agent: BicycleAgent, rng: Random) -> BicycleAgent:
    """
    Change to another rider type, with speeds clamped to its maximum.

    :rtype: BicycleAgent
    """
    return retyped(agent, rng.choice([rider_type for rider_type in RiderType if rider_type is not agent.rider_type]))


def mut_bicycle(agent: BicycleAgent, space: RouteSpace, rng: Random,
                config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent:
    """
    Return a mutant: each gene mutates only with its own
    ``config.mutation_rates`` chance, through an operator drawn by
    ``config.mutation_weights``.

    Every step keeps the mutant acceptable. When no gene mutated, ``agent``
    is returned unchanged.

    :param RouteSpace space: Built with ``BicycleAgent.space``.
    :rtype: BicycleAgent
    """
    operators = {
        "reroute": lambda child: mut_reroute(child, space, rng, config),
        "extend_or_truncate": lambda child: mut_extend_or_truncate(child, space, rng, config),
        "s": lambda child: mut_s(child, space, rng, config),
        "turn_dwell": lambda child: mut_turn_dwell(child, rng, config),
        "rider_type": lambda child: mut_rider_type(child, rng),
        "speed": lambda child: mut_speed(child, rng, config),
        "speed_point": lambda child: mut_speed_point(child, rng, config, space),
        "stop": lambda child: mut_stop(child, rng, config),
        "start_t": lambda child: mut_start_t(child, rng, config),
    }
    return vary(agent, config.GENES, operators, config.MUTATION_GENES, config.mutation_weights,
                config.mutation_rates, rng, config.attempts, lambda candidate: acceptable(space, candidate, config))
