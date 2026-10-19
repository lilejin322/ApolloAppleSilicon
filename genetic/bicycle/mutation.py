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
from ..profile_operators import SPEED_POINT_ACTIONS, mut_drive_style, mut_speed, mut_speed_point, mut_start_t, mut_stop
from ...scenario.realism import draw_type
from .common import BicycleOperatorConfig, acceptable, keeps_start, on_route, retyped


def mut_reroute(agent: BicycleAgent, space: RouteSpace, rng: Random,
                config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :returns: The mutant, or None if no alternative fits.
    """
    return on_route(agent, route_operators.mut_reroute(agent.route, space, rng, config), space, rng, config)


def mut_extend_or_truncate(agent: BicycleAgent, space: RouteSpace, rng: Random,
                           config: BicycleOperatorConfig = BicycleOperatorConfig(),
                           direction: str | None = None) -> BicycleAgent | None:
    """
    Extend or truncate the route at its start or end (``direction`` as in
    ``route_operators.mut_extend_or_truncate``: the mutation event's draw).

    :returns: The mutant, or None if no change applies.
    """
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config, direction=direction),
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


def mut_rider_type(agent: BicycleAgent, rng: Random,
                   config: BicycleOperatorConfig = BicycleOperatorConfig(),
                   drawn: RiderType | None = None) -> BicycleAgent | None:
    """
    Resample the rider type from ``config.rider_type_weights``, with speeds
    moved to its prior (``retyped``); keeps a population's rider mix at those weights.
    A mutation event draws once (``drawn``); a draw of the current type is a
    no-op, since redrawing until it changes would flip types 50/50.

    :returns: The mutant, or None if the draw is the current type.
    """
    rider_type = drawn if drawn is not None else RiderType[draw_type(rng, config.rider_type_weights)]
    return None if rider_type is agent.rider_type else retyped(agent, rider_type)


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
    draws: list[RiderType] = []  # one type draw per mutation event, kept over retries

    def rider_type(child: BicycleAgent) -> BicycleAgent | None:
        if not draws:
            draws.append(RiderType[draw_type(rng, config.rider_type_weights)])
        return mut_rider_type(child, rng, config, draws[0])

    direction = rng.choice(("extend", "truncate"))  # extend/truncate direction for this event, kept over retries
    point_action = rng.choice(SPEED_POINT_ACTIONS)  # speed-point action, kept over retries
    operators = {
        "reroute": lambda child: mut_reroute(child, space, rng, config),
        "extend_or_truncate": lambda child: mut_extend_or_truncate(child, space, rng, config, direction),
        "s": lambda child: mut_s(child, space, rng, config),
        "turn_dwell": lambda child: mut_turn_dwell(child, rng, config),
        "rider_type": rider_type,
        "speed": lambda child: mut_speed(child, rng, config, space),
        "speed_point": lambda child: mut_speed_point(child, rng, config, space, point_action),
        "stop": lambda child: mut_stop(child, rng, config),
        "drive_style": lambda child: mut_drive_style(child, rng),
        "start_t": lambda child: mut_start_t(child, rng, config),
    }
    return vary(agent, config.GENES, operators, config.MUTATION_GENES, config.mutation_weights,
                config.mutation_rates, rng, config.attempts,
                lambda candidate: acceptable(space, candidate, config) and keeps_start(space, candidate, agent, config))
