"""
Crossover of PedestrianAgent chromosomes.

Each operator returns one child of ``first`` and ``second`` that keeps
``first``'s other genes, or None when it does not apply. ``cx_pedestrian``
crosses each gene with its own rate and returns two children.
"""

from dataclasses import replace
from random import Random
from typing import Callable, Dict, List, Tuple

from ...scenario.pedestrian import PedestrianAgent
from ...scenario.walkways import MIN_WALK, WalkNetwork, WalkRoute, Excursion
from ..common import vary
from ..profile_operators import cx_speed_profile, cx_start_time
from .common import PedestrianOperatorConfig, acceptable, on_route, retyped
from .mutation import _same_stops


def _shape(route: WalkRoute) -> Tuple[int, bool]:
    """
    What crossover keeps of the lead's walk: its number of excursions, and whether it turns back.

    :param WalkRoute route: Walk whose shape is inspected.
    :returns: The excursion count and whether the last excursion has a turning point.
    :rtype: Tuple[int, bool]
    """
    return len(route.excursions), bool(route.excursions) and route.excursions[-1].turn_back is not None


def _sidewalks(route: WalkRoute) -> List[str]:
    """
    Return the sidewalk of each sidewalk element ``w<i>``, in walking order.

    :param WalkRoute route: Walk whose sidewalk sequence is extracted.
    :returns: The starting sidewalk followed by each excursion's arrival sidewalk.
    :rtype: List[str]
    """
    return [route.start.sidewalk] + [excursion.arrival.sidewalk for excursion in route.excursions]


def _links_within_generation(route: WalkRoute, network: WalkNetwork,
                             config: PedestrianOperatorConfig) -> bool:
    """
    Check whether every sidewalk stretch satisfies generation distance bounds.

    With excursions, the start and end must lie within ``config.max_approach``
    of the first entry and last arrival, and consecutive excursions must be
    within ``config.max_link``. Without excursions, the endpoint gap must lie
    between ``MIN_WALK`` and ``max(MIN_WALK, 2 * config.max_approach)``.

    :param WalkRoute route: Walk whose sidewalk stretches are checked.
    :param WalkNetwork network: Walk network used to measure gaps.
    :param PedestrianOperatorConfig config: Configuration defining approach and excursion-link bounds.
    :returns: ``True`` if every sidewalk stretch satisfies the generation bounds.
    :rtype: bool
    """
    excursions: Tuple[Excursion, ...] = route.excursions
    if not excursions:
        return MIN_WALK <= network.gap(route.start, route.end) <= max(MIN_WALK, 2 * config.max_approach)
    links: List[float] = [network.gap(a.arrival, b.entry) for a, b in zip(excursions, excursions[1:])]
    return (network.gap(route.start, excursions[0].entry) <= config.max_approach
            and network.gap(route.end, excursions[-1].arrival) <= config.max_approach
            and all(link <= config.max_link for link in links))


def cx_shared_sidewalk(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork,
                       rng: Random, config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Join ``first``'s walk prefix to ``second``'s walk suffix at a shared sidewalk.

    Keep ``first``'s start and excursions before the join, followed by
    ``second``'s remaining excursions and end. This is the pedestrian analogue
    of ``route_operators.cx_shared_lane``.

    Every sidewalk stretch stays within generation's bounds
    (``config.max_approach``, ``config.max_link``). Unbounded joins previously
    added 20 m per join on borregas_ave (mean walk 58 -> 78 m in 30 generations).

    The child preserves ``first``'s excursion count and final turn-back status
    (``_shape``). Unrestricted joins previously spread excursion counts apart
    (san_mateo, 20 neutral generations: one-excursion walks fell from 67% to 30%)
    and increased turn backs, whose shorter walks passed length bounds more
    often (4% -> 7% in 15 generations).

    :param PedestrianAgent first: Parent providing the walk prefix and preserved shape.
    :param PedestrianAgent second: Parent providing the walk suffix.
    :param WalkNetwork network: Walk network used to check sidewalk gaps and apply the child walk.
    :param Random rng: Random number generator used to select a candidate walk.
    :param PedestrianOperatorConfig config: Configuration defining approach and excursion-link bounds.
    :returns: The child, or ``None`` if no distinct walk preserves ``first``'s shape
        within the generation bounds, or the selected walk cannot be applied.
    :rtype: PedestrianAgent | None
    """
    mine: List[str]
    theirs: List[str]
    mine, theirs = _sidewalks(first.route), _sidewalks(second.route)
    shape: Tuple[int, bool] = _shape(first.route)
    options: List[WalkRoute] = []
    for i, sidewalk in enumerate(mine):
        for j, other in enumerate(theirs):
            if sidewalk != other:
                continue
            head: Tuple[Excursion, ...]
            tail: Tuple[Excursion, ...]
            head, tail = first.route.excursions[:i], second.route.excursions[j:]
            if len(head) + len(tail) != shape[0] or (tail and any(e.turn_back is not None for e in head)):
                continue
            route: WalkRoute = WalkRoute(first.route.start, head + tail, second.route.end)
            if not _links_within_generation(route, network, config):
                continue
            if _shape(route) == shape and route not in (first.route, second.route):
                options.append(route)
    if not options:
        return None
    return on_route(first, rng.choice(options), network)


def cx_route_swap(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork) -> PedestrianAgent | None:
    """
    Give ``first`` the walk of ``second`` if both walks have the same excursion
    count and final turn-back status.

    Used by ``cx_pedestrian`` when walk recombination fails, which happens more
    often for walks with zero or two excursions. Unrestricted swaps previously
    replaced these with common one-excursion walks (san_mateo, 15 neutral
    generations: their combined share fell from 33% to 21%).

    :param PedestrianAgent first: Parent receiving the other parent's walk.
    :param PedestrianAgent second: Parent providing the replacement walk.
    :param WalkNetwork network: Walk network used to apply the replacement walk.
    :returns: The child, or ``None`` if the walks are equal, differ in shape,
        or the replacement walk cannot be applied.
    :rtype: PedestrianAgent | None
    """
    if first.route == second.route or _shape(first.route) != _shape(second.route):
        return None
    return on_route(first, second.route, network)


def cx_walker_type(first: PedestrianAgent, second: PedestrianAgent) -> PedestrianAgent | None:
    """
    Give ``first`` the walker type of ``second``, mapping speeds to that type's prior through ``retyped``.

    :param PedestrianAgent first: Parent whose walker type and target speeds are updated.
    :param PedestrianAgent second: Parent providing the walker type.
    :returns: The child, or ``None`` if the walker types are equal.
    :rtype: PedestrianAgent | None
    """
    return None if first.walker_type is second.walker_type else retyped(first, second.walker_type)


def cx_walk_style(first: PedestrianAgent, second: PedestrianAgent) -> PedestrianAgent | None:
    """
    Give ``first`` the walk style of ``second``.

    :param PedestrianAgent first: Parent receiving the walk style.
    :param PedestrianAgent second: Parent providing the walk style.
    :returns: The child, or ``None`` if the walk styles are equal.
    :rtype: PedestrianAgent | None
    """
    return None if first.walk_style == second.walk_style else replace(first, walk_style=second.walk_style)


def cx_pedestrian(first: PedestrianAgent, second: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  ) -> Tuple[PedestrianAgent, PedestrianAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each gene is crossed independently with its own ``config.crossover_rates``
    probability, using an operator drawn by ``config.crossover_weights``.
    Every step keeps the child acceptable. When the walk gene is selected and
    recombination fails, a whole-walk swap is attempted with probability
    ``config.route_swap_rate``. A child with no successful crossover is its
    lead unchanged.

    :param PedestrianAgent first: Lead parent of the first child and donor for the second.
    :param PedestrianAgent second: Lead parent of the second child and donor for the first.
    :param WalkNetwork network: Walk network used to construct and validate the children.
    :param Random rng: Random number generator used for gene selection, operator selection, and crossover.
    :param PedestrianOperatorConfig config: Configuration controlling crossover rates, operator weights, and acceptance.
    :returns: Two acceptable children, led by ``first`` and ``second``, respectively.
    :rtype: Tuple[PedestrianAgent, PedestrianAgent]
    """
    def child_of(lead: PedestrianAgent, other: PedestrianAgent) -> PedestrianAgent:
        """
        Build an acceptable child by crossing selected genes of ``lead`` with ``other``.

        :param PedestrianAgent lead: Parent providing the child's initial genes.
        :param PedestrianAgent other: Parent providing genes selected for crossover.
        :returns: The resulting child, or ``lead`` unchanged if no crossover succeeds.
        :rtype: PedestrianAgent
        """
        operators: Dict[str, Callable[[PedestrianAgent], PedestrianAgent | None]] = {
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
