"""
Crossover of NPCVehicleAgent chromosomes.

Each operator returns one child of ``first`` and ``second`` that keeps
``first``'s other genes, or None when it does not apply. ``cx_npc_agent``
crosses each gene with its own rate and returns two children.
"""

from dataclasses import replace
from random import Random

from ...scenario import NPCVehicleAgent
from .. import route_operators
from ..common import vary
from ..profile_operators import cx_drive_style, cx_speed_profile, cx_start_time
from .common import NPCOperatorConfig, Spaces, acceptable, on_route, positions, retyped


def cx_shared_lane(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces,
                   rng: Random) -> NPCVehicleAgent | None:
    """
    Join ``first``'s route up to a lane both routes use with ``second``'s route from it.

    Lane changes keep either parent's position where it still fits, ``first``'s
    first; others are drawn again.

    :returns: The child, or None if the routes share no lane that joins without repeats.
    """
    space = spaces[first.vehicle_type]
    route = route_operators.cx_shared_lane(first.route, second.route, space, rng)
    return on_route(first, route, space, rng, keep={**positions(second), **positions(first)})


def cx_bridge(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces, rng: Random,
              config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    Lane-change positions are inherited as in ``cx_shared_lane``.

    :returns: The child, or None if no lane pair is reachable within ``config.max_length``.
    """
    space = spaces[first.vehicle_type]
    route = route_operators.cx_bridge(first.route, second.route, space, rng, config,
                                      lane_change_s=first.lane_change_s)
    return on_route(first, route, space, rng, keep={**positions(second), **positions(first)})


def cx_route_swap(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces,
                  rng: Random) -> NPCVehicleAgent | None:
    """
    Give ``first`` the route of ``second`` with its lane-change positions.

    This exchanges routes rather than creating new ones, so ``cx_npc_agent``
    only uses it when route recombination fails. Positions that do not fit
    ``first``'s vehicle type are drawn again.

    :returns: The child, or None if routes and positions are equal or the route
        does not fit ``first``'s vehicle type.
    """
    if (first.route, first.lane_change_s) == (second.route, second.lane_change_s):
        return None
    space = spaces[first.vehicle_type]
    return on_route(first, second.route, space, rng, keep=positions(second))


def cx_lane_change_s(first: NPCVehicleAgent, second: NPCVehicleAgent) -> NPCVehicleAgent | None:
    """
    Give ``first``'s lane changes ``second``'s positions where both change between the same lanes.

    :returns: The child, or None if no shared lane change has a different position.
    """
    theirs = positions(second)
    changes = tuple(theirs.get(pair, s) for pair, s in zip(first.route.lane_change_pairs, first.lane_change_s))
    return None if changes == first.lane_change_s else replace(first, lane_change_s=changes)


def cx_vehicle_type(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces) -> NPCVehicleAgent | None:
    """
    Give ``first`` the vehicle type of ``second``, with speeds clamped to that type's maximum.

    :returns: The child, or None if the types are equal or ``second``'s is not in ``spaces``.
    """
    if first.vehicle_type is second.vehicle_type or second.vehicle_type not in spaces:
        return None
    return retyped(first, second.vehicle_type)


def cx_npc_agent(first: NPCVehicleAgent, second: NPCVehicleAgent, spaces: Spaces, rng: Random,
                 config: NPCOperatorConfig = NPCOperatorConfig()) -> tuple[NPCVehicleAgent, NPCVehicleAgent]:
    """
    Return two children: one led by ``first``, one led by ``second``.

    Each gene of the lead is crossed with the other parent only with its own
    ``config.crossover_rates`` chance, independently of the other genes,
    through an operator drawn by ``config.crossover_weights``. Every step
    keeps the child acceptable. When the route gene is drawn and its recombining
    operators fail, the lead takes the other parent's whole route with probability
    ``config.route_swap_rate`` (``cx_route_swap``). A child where no gene was
    crossed is its lead unchanged.

    :param Spaces spaces: A route space per vehicle type the children may have.
    :rtype: tuple[NPCVehicleAgent, NPCVehicleAgent]
    """
    def child_of(lead: NPCVehicleAgent, other: NPCVehicleAgent) -> NPCVehicleAgent:
        # Routes recombine only between parents of one vehicle type (unless
        # ``config.cross_type_routes``): a car takes a van's road freely, but a
        # van rarely fits a car's, so routes flowed into the roads large vehicles
        # fit (shalun: car routes fitting a van 14% -> 51% in 30 generations).
        routes = config.cross_type_routes or lead.vehicle_type is other.vehicle_type
        operators = {
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
