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

    :param BicycleAgent first: Leading parent supplying the route prefix, start
        position, rider type, start time, and speed profile to rebase.
    :param BicycleAgent second: Parent supplying the route suffix, end position,
        and turn waits for pivots absent from the first parent's wait mapping.
    :param RouteSpace space: Bicycle route space used to evaluate joins, locate
        pivots, and validate the child.
    :param Random rng: Random source for shared-lane join selection and new turn waits.
    :param BicycleOperatorConfig config: Supplies ``max_turn_dwell``, the upper bound
        in seconds for newly sampled waits. Operator acceptance bounds and
        ``keep_start_kind`` are checked by the caller.
    :returns: A child with inherited turn waits and a rebased speed profile, or None
        if no shared-lane join avoids repeated lanes or ``on_route`` rejects the child.
    :rtype: BicycleAgent | None
    """
    route = route_operators.cx_shared_lane(first.route, second.route, space, rng)
    return on_route(first, route, space, rng, config,
                    keep={**second.turn_waits(space), **first.turn_waits(space)})


def cx_bridge(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
              config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    Turn waits are inherited as in ``cx_shared_lane``.

    :param BicycleAgent first: Leading parent supplying the route prefix, start
        position, rider type, start time, and speed profile to rebase.
    :param BicycleAgent second: Parent supplying the route suffix, end position,
        and turn waits for pivots absent from the first parent's wait mapping.
    :param RouteSpace space: Bicycle route space used to find connecting lane paths,
        locate pivots, and validate the child.
    :param Random rng: Random source for bridge selection and newly sampled turn waits.
    :param BicycleOperatorConfig config: Supplies the bridge search distance and
        candidate-path limit, optional parent-length restriction, and maximum new
        turn wait. Operator acceptance bounds and ``keep_start_kind`` are checked
        by the caller.
    :returns: A bridged child with inherited turn waits and a rebased speed profile,
        or None if no bridge satisfies the search constraints or ``on_route``
        rejects the child.
    :rtype: BicycleAgent | None
    """
    route = route_operators.cx_bridge(first.route, second.route, space, rng, config)
    return on_route(first, route, space, rng, config,
                    keep={**second.turn_waits(space), **first.turn_waits(space)})


def cx_route_swap(first: BicycleAgent, second: BicycleAgent, space: RouteSpace, rng: Random,
                  config: BicycleOperatorConfig = BicycleOperatorConfig()) -> BicycleAgent | None:
    """
    Give ``first`` the route of ``second`` with its turn waits.

    ``cx_bicycle`` only uses it when route recombination fails.

    :param BicycleAgent first: Leading parent supplying the rider type, start time,
        and speed profile to rebase onto the replacement route.
    :param BicycleAgent second: Parent donating its entire route and turn waits.
    :param RouteSpace space: Bicycle route space used to locate pivots, rebase the
        speed profile, and validate the child.
    :param Random rng: Random source passed to ``on_route`` for any turn waits
        missing from the second parent's mapping.
    :param BicycleOperatorConfig config: Supplies the maximum newly sampled turn
        wait. Operator acceptance bounds and ``keep_start_kind`` are checked
        by the caller.
    :returns: A child with the second parent's route and waits, or None if both
        parents' routes and waits are equal or ``on_route`` rejects the child.
    :rtype: BicycleAgent | None
    """
    if (first.route, first.turn_dwell) == (second.route, second.turn_dwell):
        return None
    return on_route(first, second.route, space, rng, config, keep=second.turn_waits(space))


def cx_turn_dwell(first: BicycleAgent, second: BicycleAgent, space: RouteSpace) -> BicycleAgent | None:
    """
    Give ``first``'s pivots ``second``'s waits where both routes turn at the same pivot.

    :param BicycleAgent first: Leading parent whose turn waits are updated;
        waits at unshared pivots and all other genes are preserved.
    :param BicycleAgent second: Parent donating turn waits in seconds at shared pivots.
    :param RouteSpace space: Bicycle route space used to identify matching pivots
        by connector lane ID and pivot index.
    :returns: The child, or None if no shared pivot has a different wait.
    :rtype: BicycleAgent | None
    """
    theirs = second.turn_waits(space)
    waits = tuple(theirs.get(key, wait) for key, wait in first.turn_waits(space).items())
    return None if waits == first.turn_dwell else replace(first, turn_dwell=waits)


def cx_rider_type(first: BicycleAgent, second: BicycleAgent) -> BicycleAgent | None:
    """
    Give ``first`` the rider type of ``second``, with speeds moved to that type's prior (``retyped``).

    :param BicycleAgent first: Leading parent whose rider type and positive target
        speeds are changed; stop points, route, turn waits, and start time are preserved.
    :param BicycleAgent second: Parent donating the rider type used to map target
        speeds to the new type's speed prior and clamp them to its maximum.
    :returns: The child, or None if the types are equal.
    :rtype: BicycleAgent | None
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

    :param BicycleAgent first: Leading parent for the first child and gene donor
        for the second child.
    :param BicycleAgent second: Leading parent for the second child and gene donor
        for the first child.
    :param RouteSpace space: Built with ``BicycleAgent.space``; used for route
        recombination, trajectory validation, and start-lane-kind checks.
    :param Random rng: Random source for gene selection, operator selection,
        crossover proposals, and route-swap fallback draws.
    :param BicycleOperatorConfig config: Supplies per-gene crossover rates, operator
        weights and retry limits, route-swap probability, proposal settings, and
        acceptance bounds. ``keep_start_kind`` preserves each lead's start-lane kind.
    :raises ValueError: If variation cannot produce an acceptable result for either
        leading parent, including when its unchanged fallback fails acceptance.
    :returns: Two children in leading-parent order. A child with no accepted changes
        is its original leading parent, provided that parent passes acceptance.
    :rtype: tuple[BicycleAgent, BicycleAgent]
    """
    def child_of(lead: BicycleAgent, other: BicycleAgent) -> BicycleAgent:
        """
        Cross ``lead`` gene by gene with ``other`` using the enclosing crossover settings.

        :param BicycleAgent lead: Base agent and reference for start-lane-kind
            preservation when ``config.keep_start_kind`` is enabled.
        :param BicycleAgent other: Donor of genes and the fallback replacement route.
        :raises ValueError: If the final agent fails acceptance, including when
            no change succeeds and ``lead`` itself is unacceptable.
        :returns: An accepted child, or ``lead`` unchanged if no change is accepted
            and it already satisfies the acceptance checks.
        :rtype: BicycleAgent
        """
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
