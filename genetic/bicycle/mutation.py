"""
Mutation of BicycleAgent chromosomes.

Each operator returns a mutant that changes one gene, or None when it does
not apply. ``mut_bicycle`` mutates each gene with its rate. Bicycles never
change lanes, so there is no lane-change mutation.
"""

from dataclasses import replace
from random import Random
from typing import Callable, Dict, List

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

    :param BicycleAgent agent: Source agent supplying the route, rider type, start
        time, speed profile, and turn waits to preserve where pivots remain shared.
    :param RouteSpace space: Bicycle route space used to find alternative paths,
        rebase the speed profile, locate pivots, and validate the mutant.
    :param Random rng: Random source for stretch selection, alternative-path
        selection, and newly sampled turn waits.
    :param BicycleOperatorConfig config: Supplies path-search limits, the maximum
        relative detour, and maximum new turn wait. Operator acceptance bounds
        and ``keep_start_kind`` are checked by the caller.
    :returns: A mutant with repaired turn waits and a rebased speed profile, or None
        if no alternative fits or ``on_route`` rejects the resulting agent.
    :rtype: BicycleAgent | None
    """
    return on_route(agent, route_operators.mut_reroute(agent.route, space, rng, config), space, rng, config)


def mut_extend_or_truncate(agent: BicycleAgent, space: RouteSpace, rng: Random,
                           config: BicycleOperatorConfig = BicycleOperatorConfig(),
                           direction: str | None = None) -> BicycleAgent | None:
    """
    Extend or truncate the route at its start or end, repairing turn waits and the speed profile.

    :param BicycleAgent agent: Source agent supplying the route, rider type, start
        time, speed profile, and turn waits to preserve at retained pivots.
    :param RouteSpace space: Bicycle route space used to modify route endpoints,
        rebase the speed profile, locate pivots, and validate the mutant.
    :param Random rng: Random source for direction selection when omitted, endpoint
        changes, and newly sampled turn waits.
    :param BicycleOperatorConfig config: Supplies the maximum newly sampled turn
        wait. Operator acceptance bounds and ``keep_start_kind`` are checked
        by the caller.
    :param str | None direction: ``"extend"`` or ``"truncate"``; None draws either
        with equal probability. Pass the same draw across retries to preserve
        the mutation event's direction.
    :raises ValueError: If ``direction`` is neither an allowed string nor None.
    :returns: A mutant with repaired turn waits and a rebased speed profile, or None
        if the chosen direction produces no route or ``on_route`` rejects the agent.
    :rtype: BicycleAgent | None
    """
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config, direction=direction),
                    space, rng, config)


def mut_s(agent: BicycleAgent, space: RouteSpace, rng: Random,
          config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    Steps are clamped to their lanes and, where endpoint restrictions apply,
    snapped to the nearest allowed position.

    :param BicycleAgent agent: Source agent supplying the route, rider type, start
        time, speed profile, and turn waits to preserve at retained pivots.
    :param RouteSpace space: Bicycle route space supplying lane lengths and allowed
        endpoint positions, and used to rebase and validate the mutant.
    :param Random rng: Random source for endpoint selection, Gaussian position
        steps, and newly sampled turn waits.
    :param BicycleOperatorConfig config: Supplies ``sigma_s``, the position-step
        standard deviation in metres, and ``max_turn_dwell`` for new waits in
        seconds. Operator acceptance bounds are checked by the caller.
    :returns: A candidate with repaired turn waits and a rebased speed profile,
        possibly equal to the input; None if a moved endpoint has no allowed
        position or ``on_route`` rejects the resulting agent.
    :rtype: BicycleAgent | None
    """
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config), space, rng, config)


def mut_turn_dwell(agent: BicycleAgent, rng: Random,
                   config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Move one turn wait by a normal step (``config.sigma_turn_dwell``), within
    ``[0, config.max_turn_dwell]``.

    :param BicycleAgent agent: Source agent whose single selected turn wait is
        changed; other waits and all other genes are preserved.
    :param Random rng: Random source for uniform turn-wait selection and the
        Gaussian time step.
    :param BicycleOperatorConfig config: Supplies ``sigma_turn_dwell``, the step's
        standard deviation, and ``max_turn_dwell``, the clamping upper bound,
        both in seconds.
    :returns: A copy with the proposed turn wait, possibly equal to the input;
        None if ``agent.turn_dwell`` is empty.
    :rtype: BicycleAgent | None
    """
    if not agent.turn_dwell:
        return None
    index: int = rng.randrange(len(agent.turn_dwell))
    wait: float = gauss_step(rng, agent.turn_dwell[index], config.sigma_turn_dwell, config.max_turn_dwell)
    return replace(agent, turn_dwell=agent.turn_dwell[:index] + (wait,) + agent.turn_dwell[index + 1:])


def mut_rider_type(agent: BicycleAgent, rng: Random,
                   config: BicycleOperatorConfig = BicycleOperatorConfig(),
                   drawn: RiderType | None = None) -> BicycleAgent | None:
    """
    Resample the rider type from ``config.rider_type_weights``, with speeds
    moved to its prior (``retyped``); keeps a population's rider mix at those weights.
    A mutation event draws once (``drawn``); a draw of the current type is a
    no-op, since redrawing until it changes would flip types 50/50.

    :param BicycleAgent agent: Source agent whose rider type and positive target
        speeds are changed; stop points, route, turn waits, and start time are preserved.
    :param Random rng: Random source for weighted rider-type selection when
        ``drawn`` is None.
    :param BicycleOperatorConfig config: Supplies ``rider_type_weights``, mapping
        ``RiderType`` member names to sampling weights when ``drawn`` is None.
    :param RiderType | None drawn: Type already sampled for this mutation event;
        reuse it across retries. None samples a type on this call.
    :returns: The mutant, or None if the draw is the current type.
    :rtype: BicycleAgent | None
    """
    rider_type: RiderType = drawn if drawn is not None else RiderType[draw_type(rng, config.rider_type_weights)]
    return None if rider_type is agent.rider_type else retyped(agent, rider_type)


def mut_bicycle(agent: BicycleAgent, space: RouteSpace, rng: Random,
                config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent:
    """
    Return a mutant: each gene mutates only with its own
    ``config.mutation_rates`` chance, through an operator drawn by
    ``config.mutation_weights``.

    Every step keeps the mutant acceptable. When no gene mutated, ``agent``
    is returned unchanged if it passes acceptance. The rider-type draw,
    extend/truncate direction, and speed-point action are reused across retries.

    :param BicycleAgent agent: Base agent to mutate and reference for preserving
        the start-lane kind when ``config.keep_start_kind`` is enabled.
    :param RouteSpace space: Built with ``BicycleAgent.space``; used for route
        proposals, trajectory validation, and start-lane-kind checks.
    :param Random rng: Random source for gene selection, operator selection,
        and mutation proposals.
    :param BicycleOperatorConfig config: Supplies per-gene mutation rates, operator
        weights and retry limits, mutation step sizes, rider-type sampling weights,
        and acceptance bounds.
    :raises ValueError: If the final agent fails acceptance, including when no
        change succeeds and the original agent is unacceptable.
    :returns: An accepted mutant, or the original agent if no change is accepted
        and it already satisfies the acceptance checks.
    :rtype: BicycleAgent
    """
    draws: List[RiderType] = []  # one type draw per mutation event, kept over retries

    def rider_type(child: BicycleAgent) -> BicycleAgent | None:
        """
        Mutate the rider type using one weighted draw shared across this event's retries.

        :param BicycleAgent child: Current candidate whose rider type and positive
            target speeds are updated through ``mut_rider_type``.
        :returns: A candidate with the sampled rider type and mapped speeds, or
            None if the sampled type matches the child's current type.
        :rtype: BicycleAgent | None
        """
        if not draws:
            draws.append(RiderType[draw_type(rng, config.rider_type_weights)])
        return mut_rider_type(child, rng, config, draws[0])

    direction: str = rng.choice(("extend", "truncate"))  # extend/truncate direction for this event, kept over retries
    point_action: str = rng.choice(SPEED_POINT_ACTIONS)  # speed-point action, kept over retries
    operators: Dict[str, Callable[[BicycleAgent], BicycleAgent | None]] = {
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
