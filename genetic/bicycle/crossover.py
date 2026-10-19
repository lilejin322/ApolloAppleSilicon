"""
Crossover of BicycleAgent chromosomes.

Each operator returns one child of ``first`` and ``second`` that keeps
``first``'s other genes, or None when it does not apply. ``cx_bicycle``
crosses each gene with its own rate and returns two children.
"""

from dataclasses import replace
from random import Random

from ...scenario import BicycleAgent, RouteSpace
from .. import route_operators
from ..common import vary
from ..profile_operators import cx_drive_style, cx_speed_profile, cx_start_time
from .common import BicycleOperatorConfig, acceptable, keeps_start, on_route, retyped


def cx_shared_lane(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
                   config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Join ``first``'s route up to a lane both routes use with ``second``'s route from it.

    Turn waits are kept from either parent, ``first``'s first; others are drawn.

    :returns: The child, or None if the routes share no lane that joins without repeats.
    """
    route = route_operators.cx_shared_lane(first.route, second.route, space, rng)
    return on_route(first, route, space, rng, config,
                    keep={**second.turn_waits(space), **first.turn_waits(space)})


def cx_bridge(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
              config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    Turn waits are inherited as in ``cx_shared_lane``.

    :returns: The child, or None if no lane pair is reachable within ``config.max_length``.
    """
    route = route_operators.cx_bridge(first.route, second.route, space, rng, config)
    return on_route(first, route, space, rng, config,
                    keep={**second.turn_waits(space), **first.turn_waits(space)})


def cx_route_swap(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
                  config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Give ``first`` the route of ``second`` with its turn waits.

    ``cx_bicycle`` only uses it when route recombination fails.

    :returns: The child, or None if routes and waits are equal or the route
        does not fit ``first``'s rider type.
    """
    if (first.route, first.turn_dwell) == (second.route, second.turn_dwell):
        return None
    return on_route(first, second.route, space, rng, config, keep=second.turn_waits(space))


def cx_turn_dwell(first: BicycleAgent, second: BicycleAgent, space: RouteSpace) -> BicycleAgent | None:
    """
    Give ``first``'s pivots ``second``'s waits where both routes turn at the same pivot.

    :returns: The child, or None if no shared pivot has a different wait.
    """
    theirs = second.turn_waits(space)
    waits = tuple(theirs.get(key, wait) for key, wait in first.turn_waits(space).items())
    return None if waits == first.turn_dwell else replace(first, turn_dwell=waits)


def cx_rider_type(first: BicycleAgent, second: BicycleAgent) -> BicycleAgent | None:
    """
    Give ``first`` the rider type of ``second``, with speeds moved to that type's prior (``retyped``).

    :returns: The child, or None if the types are equal.
    """
    return None if first.rider_type is second.rider_type else retyped(first, second.rider_type)


def cx_bicycle(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
               config: BicycleOperatorConfig = BicycleOperatorConfig()) -> tuple[BicycleAgent, BicycleAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each gene of the lead is crossed with the other parent only with its own
    ``config.crossover_rates`` chance, through an operator drawn by
    ``config.crossover_weights``. Every step keeps the child acceptable. When
    the route gene is drawn and its recombining operators fail, the lead takes
    the other parent's whole route with probability ``config.route_swap_rate``
    (``cx_route_swap``). A child where no gene was crossed is its lead unchanged.

    :param RouteSpace space: Built with ``BicycleAgent.space``.
    :rtype: tuple[BicycleAgent, BicycleAgent]
    """
    def child_of(lead: BicycleAgent, other: BicycleAgent) -> BicycleAgent:
        operators = {
            "shared_lane": lambda child: cx_shared_lane(child, other, space, rng, config),
            "bridge": lambda child: cx_bridge(child, other, space, rng, config),
            "turn_dwell": lambda child: cx_turn_dwell(child, other, space),
            "rider_type": lambda child: cx_rider_type(child, other),
            "speed_profile": lambda child: cx_speed_profile(child, retyped(other, child.rider_type), rng, space, min_speed=config.min_speed),
            "drive_style": lambda child: cx_drive_style(child, other),
            "start_time": lambda child: cx_start_time(child, other),
        }
        return vary(lead, config.GENES, operators, config.CROSSOVER_GENES, config.crossover_weights,
                    config.crossover_rates, rng, config.attempts,
                    lambda candidate: acceptable(space, candidate, config)
                    and keeps_start(space, candidate, lead, config),
                    fallbacks={"route": (config.route_swap_rate,
                                         lambda child: cx_route_swap(child, other, space, rng, config))})

    return child_of(first, second), child_of(second, first)
