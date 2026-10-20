"""
Crossover of NPCVehicleAgent chromosomes.

Each operator returns one child of ``first`` and ``second`` that keeps
``first``'s other genes, or None when it does not apply. ``cx_npc_agent``
crosses each gene with its own rate and returns two children.
"""

from dataclasses import replace
from random import Random
from typing import Callable, Dict, Tuple

from ...scenario import NPCVehicleAgent, LaneRoute, RouteSpace
from .. import route_operators
from ..common import vary
from ..profile_operators import cx_drive_style, cx_speed_profile, cx_start_time
from .common import NPCOperatorConfig, Spaces, acceptable, on_route, positions, retyped


def cx_shared_lane(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces,
                   rng: Random) -> NPCVehicleAgent | None:
    """
    Join ``first``'s route up to a shared lane with ``second``'s route from that lane.

    Lane changes retain a valid position from either parent, preferring ``first``;
    otherwise, their positions are sampled again.

    :param NPCVehicleAgent first: Parent providing the route prefix.
    :param NPCVehicleAgent second: Parent providing the route suffix.
    :param Spaces spaces: Search spaces used to construct the child route.
    :param Random rng: Random number generator used for crossover and sampling.
    :returns: The child, or ``None`` if no shared lane allows joining without repeats.
    :rtype: NPCVehicleAgent | None
    """
    space: RouteSpace = spaces[first.vehicle_type]
    route: LaneRoute | None = route_operators.cx_shared_lane(first.route, second.route, space, rng)
    return on_route(first, route, space, rng, keep={**positions(second), **positions(first)})


def cx_bridge(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces, rng: Random,
              config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Join ``first``'s head to ``second``'s tail through a connecting lane path.

    Lane-change positions are inherited as in ``cx_shared_lane``.

    :param NPCVehicleAgent first: Parent providing the route head.
    :param NPCVehicleAgent second: Parent providing the route tail.
    :param Spaces spaces: Search spaces used to find a connecting lane path.
    :param Random rng: Random number generator used for crossover and sampling.
    :param NPCOperatorConfig config: Configuration controlling route construction and length limits.
    :returns: The child, or ``None`` if no lane pair is reachable within ``config.max_length``.
    :rtype: NPCVehicleAgent | None
    """
    space: RouteSpace = spaces[first.vehicle_type]
    route: LaneRoute | None = route_operators.cx_bridge(first.route, second.route, space, rng, config,
                                      lane_change_s=first.lane_change_s)
    return on_route(first, route, space, rng, keep={**positions(second), **positions(first)})


def cx_route_swap(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces,
                  rng: Random) -> NPCVehicleAgent | None:
    """
    Give ``first`` the route of ``second`` with its lane-change positions.

    Used by ``cx_npc_agent`` when route recombination fails. Positions that do
    not fit ``first``'s vehicle type are sampled again.

    :param NPCVehicleAgent first: Parent providing the child's vehicle type and other attributes.
    :param NPCVehicleAgent second: Parent providing the route and lane-change positions.
    :param Spaces spaces: Search spaces used to validate the route and sample positions.
    :param Random rng: Random number generator used to resample lane-change positions.
    :returns: The child, or ``None`` if routes and positions are equal or the route
        does not fit ``first``'s vehicle type.
    :rtype: NPCVehicleAgent | None
    """
    if (first.route, first.lane_change_s) == (second.route, second.lane_change_s):
        return None
    space: RouteSpace = spaces[first.vehicle_type]
    return on_route(first, second.route, space, rng, keep=positions(second))


def cx_lane_change_s(first: NPCVehicleAgent, second: NPCVehicleAgent) -> NPCVehicleAgent | None:
    """
    Give ``first``'s lane changes ``second``'s positions where both change between the same lanes.

    :param NPCVehicleAgent first: Parent providing the route and original lane-change positions.
    :param NPCVehicleAgent second: Parent providing positions for matching lane changes.
    :returns: The child, or ``None`` if no shared lane change has a different position.
    :rtype: NPCVehicleAgent | None
    """
    theirs: Dict[Tuple[str, str], float] = positions(second)
    changes: Tuple[float, ...] = tuple(theirs.get(pair, s) for pair, s in zip(first.route.lane_change_pairs, first.lane_change_s))
    return None if changes == first.lane_change_s else replace(first, lane_change_s=changes)


def cx_vehicle_type(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces) -> NPCVehicleAgent | None:
    """
    Give ``first`` the vehicle type of ``second``, with speeds clamped to that type's maximum.

    :param NPCVehicleAgent first: Parent providing the route, speeds, and other attributes.
    :param NPCVehicleAgent second: Parent providing the vehicle type.
    :param Spaces spaces: Search spaces defining the allowed vehicle types.
    :returns: The child, or ``None`` if the types are equal or ``second``'s type is not in ``spaces``.
    :rtype: NPCVehicleAgent | None
    """
    if first.vehicle_type is second.vehicle_type or second.vehicle_type not in spaces:
        return None
    return retyped(first, second.vehicle_type)


def cx_npc_agent(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces, rng: Random,
                 config: NPCOperatorConfig = NPCOperatorConfig()) -> Tuple[NPCVehicleAgent, NPCVehicleAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each gene is crossed independently with its own ``config.crossover_rates``
    probability, using an operator drawn by ``config.crossover_weights``.
    Every step keeps the child acceptable. If route recombination fails, the
    lead takes the other parent's whole route with probability
    ``config.route_swap_rate`` through ``cx_route_swap``. A child with no
    crossed genes is its lead unchanged.

    :param NPCVehicleAgent first: Lead parent of the first child and donor for the second.
    :param NPCVehicleAgent second: Lead parent of the second child and donor for the first.
    :param Spaces spaces: A route space per vehicle type the children may have.
    :param Random rng: Random number generator used for gene selection, operator selection, and crossover.
    :param NPCOperatorConfig config: Configuration controlling crossover probabilities, operator weights, and route swapping.
    :returns: Two children, led by ``first`` and ``second``, respectively.
    :rtype: Tuple[NPCVehicleAgent, NPCVehicleAgent]
    """
    def child_of(lead: NPCVehicleAgent, other: NPCVehicleAgent) -> NPCVehicleAgent:
        """
        Build an acceptable child by crossing selected genes of ``lead`` with ``other``.

        :param NPCVehicleAgent lead: Parent providing the child's initial genes.
        :param NPCVehicleAgent other: Parent providing genes selected for crossover.
        :returns: The resulting child, or ``lead`` unchanged if no crossover succeeds.
        :rtype: NPCVehicleAgent
        """
        # Routes recombine only between parents of one vehicle type (unless
        # ``config.cross_type_routes``): a car takes a van's road freely, but a
        # van rarely fits a car's, so routes flowed into the roads large vehicles
        # fit (shalun: car routes fitting a van 14% -> 51% in 30 generations).
        routes: bool = config.cross_type_routes or lead.vehicle_type is other.vehicle_type
        operators: Dict[str, Callable[[NPCVehicleAgent], NPCVehicleAgent | None]] = {
            "shared_lane": lambda child: cx_shared_lane(child, other, spaces, rng) if routes else None,
            "bridge": lambda child: cx_bridge(child, other, spaces, rng, config) if routes else None,
            "lane_change_s": lambda child: cx_lane_change_s(child, other),
            "vehicle_type": lambda child: cx_vehicle_type(child, other, spaces),
            "speed_profile": lambda child: cx_speed_profile(child, other, rng, spaces[child.vehicle_type],
                                                                spaces[other.vehicle_type], min_speed=config.min_speed),
            "drive_style": lambda child: cx_drive_style(child, other),
            "start_time": lambda child: cx_start_time(child, other),
        }
        return vary(lead, config.GENES, operators, config.CROSSOVER_GENES, config.crossover_weights,
                    config.crossover_rates, rng, config.attempts,
                    lambda candidate: acceptable(spaces, candidate, config),
                    fallbacks={"route": (config.route_swap_rate if routes else 0.0,
                                         lambda child: cx_route_swap(child, other, spaces, rng))})

    return child_of(first, second), child_of(second, first)
