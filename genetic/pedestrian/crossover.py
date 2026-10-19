"""
Crossover of PedestrianAgent chromosomes.

Each operator returns one child of ``first`` and ``second`` that keeps
``first``'s other genes, or None when it does not apply. ``cx_pedestrian``
crosses each gene with its own rate and returns two children.
"""

from dataclasses import replace
from random import Random

from ...scenario.pedestrian import PedestrianAgent
from ...scenario.walkways import MIN_WALK, WalkNetwork, WalkRoute
from ..common import vary
from ..profile_operators import cx_speed_profile, cx_start_time
from .common import PedestrianOperatorConfig, acceptable, on_route, retyped
from .mutation import _same_stops


def _shape(route: WalkRoute) -> tuple[int, bool]:
    """What crossover keeps of the lead's walk: its number of excursions, and whether it turns back."""
    return len(route.excursions), bool(route.excursions) and route.excursions[-1].turn_back is not None


def _sidewalks(route: WalkRoute) -> list[str]:
    """The sidewalk of each sidewalk element ``w<i>``, in walking order."""
    return [route.start.sidewalk] + [excursion.arrival.sidewalk for excursion in route.excursions]


def _links_within_generation(route: WalkRoute, network: WalkNetwork, config: PedestrianOperatorConfig) -> bool:
    """Whether every sidewalk stretch of the walk is as long as generation allows:
    start and end within ``config.max_approach`` of the first entry and last
    arrival (a walk without excursions ``MIN_WALK`` to twice that), consecutive
    excursions within ``config.max_link``."""
    excursions = route.excursions
    if not excursions:
        return MIN_WALK <= network.gap(route.start, route.end) <= max(MIN_WALK, 2 * config.max_approach)
    links = [network.gap(a.arrival, b.entry) for a, b in zip(excursions, excursions[1:])]
    return (network.gap(route.start, excursions[0].entry) <= config.max_approach
            and network.gap(route.end, excursions[-1].arrival) <= config.max_approach
            and all(link <= config.max_link for link in links))


def cx_shared_sidewalk(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork,
                       rng: Random, config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Join ``first``'s walk up to a sidewalk both walks use with ``second``'s
    walk from it: ``first``'s start and excursions before it, then along that
    sidewalk to ``second``'s next excursion, its later excursions and its end.
    The pedestrian analogue of ``route_operators.cx_shared_lane``.

    Every sidewalk stretch of the child stays within generation's bounds
    (``config.max_approach``, ``config.max_link``): unbounded joins walked
    along whole blocks, +20 m per join on borregas_ave (mean walk 58 -> 78 m
    in 30 generations).

    The child keeps ``first``'s number of excursions and whether it turns back
    (``_shape``): joins at any shared sidewalk spread counts apart (san_mateo,
    20 neutral generations: walks with one excursion fell from 67% to 30%), as
    unrestricted speed-profile splices once thinned point counts, and copied
    turn backs, whose shorter walks pass the length bound more often (4% -> 7%
    in 15 generations).

    :returns: The child, or None if the walks share no sidewalk where they join
        into a walk of ``first``'s shape.
    """
    mine, theirs = _sidewalks(first.route), _sidewalks(second.route)
    shape = _shape(first.route)
    options = []
    for i, sidewalk in enumerate(mine):
        for j, other in enumerate(theirs):
            if sidewalk != other:
                continue
            head, tail = first.route.excursions[:i], second.route.excursions[j:]
            if len(head) + len(tail) != shape[0] or (tail and any(e.turn_back is not None for e in head)):
                continue
            route = WalkRoute(first.route.start, head + tail, second.route.end)
            if not _links_within_generation(route, network, config):
                continue
            if _shape(route) == shape and route not in (first.route, second.route):
                options.append(route)
    if not options:
        return None
    return on_route(first, rng.choice(options), network)


def cx_route_swap(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork) -> PedestrianAgent | None:
    """
    Give ``first`` the walk of ``second`` if it has the same shape (as many
    excursions, turning back or not). ``cx_pedestrian`` only uses it when walk
    recombination fails, which it does more often for walks with none or two
    excursions: unrestricted swaps then turned them into the common
    one-excursion walks (san_mateo, 15 neutral generations: 33% -> 21%).

    :returns: The child, or None if the walks are equal, differ in shape, or
        the walk does not fit.
    """
    if first.route == second.route or _shape(first.route) != _shape(second.route):
        return None
    return on_route(first, second.route, network)


def cx_walker_type(first: PedestrianAgent, second: PedestrianAgent) -> PedestrianAgent | None:
    """
    Give ``first`` the walker type of ``second``, with speeds moved to that type's prior (``retyped``).

    :returns: The child, or None if the types are equal.
    """
    return None if first.walker_type is second.walker_type else retyped(first, second.walker_type)


def cx_walk_style(first: PedestrianAgent, second: PedestrianAgent) -> PedestrianAgent | None:
    """
    Give ``first`` the walk style of ``second``.

    :returns: The child, or None if the styles are equal.
    """
    return None if first.walk_style == second.walk_style else replace(first, walk_style=second.walk_style)


def cx_pedestrian(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  ) -> tuple[PedestrianAgent, PedestrianAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each gene of the lead is crossed with the other parent only with its own
    ``config.crossover_rates`` chance, through an operator drawn by
    ``config.crossover_weights``. Every step keeps the child acceptable. When
    the walk gene is drawn and recombination fails, the lead takes the other
    parent's whole walk with probability ``config.route_swap_rate``. A child
    where no gene was crossed is its lead unchanged.

    :rtype: tuple[PedestrianAgent, PedestrianAgent]
    """
    def child_of(lead: PedestrianAgent, other: PedestrianAgent) -> PedestrianAgent:
        operators = {
            "shared_sidewalk": lambda child: cx_shared_sidewalk(child, other, network, rng, config),
            "walker_type": lambda child: cx_walker_type(child, other),
            # The partner's speeds are first mapped onto the child's type (``retyped``):
            # spliced as they were, runners' 3 m/s points made adults run and adults'
            # points made runners walk (shalun: 35 of 400 children off their type's range).
            "speed_profile": lambda child: _same_stops(child, cx_speed_profile(child, retyped(other, child.walker_type),
                                                                                 rng, network,
                                                                                 min_speed=config.min_speed)),
            "walk_style": lambda child: cx_walk_style(child, other),
            "start_time": lambda child: cx_start_time(child, other),
        }
        return vary(lead, config.GENES, operators, config.CROSSOVER_GENES, config.crossover_weights,
                    config.crossover_rates, rng, config.attempts,
                    lambda candidate: acceptable(network, candidate, config),
                    fallbacks={"route": (config.route_swap_rate,
                                         lambda child: cx_route_swap(child, other, network))})

    return child_of(first, second), child_of(second, first)
