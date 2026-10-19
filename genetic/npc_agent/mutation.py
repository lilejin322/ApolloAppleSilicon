"""
Mutation of NPCVehicleAgent chromosomes.

Each operator returns a mutant that changes one gene, or None when it does
not apply. ``mut_npc_agent`` mutates each gene with its rate.
"""

from dataclasses import replace
from random import Random

from ...scenario import NPCVehicleAgent, VehicleType
from .. import profile_operators, route_operators
from ..common import vary
from ..profile_operators import mut_speed, mut_start_t, mut_stop
from .common import NPCOperatorConfig, Spaces, acceptable, on_route, retyped


def mut_reroute(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :returns: The mutant, or None if no alternative fits.
    """
    space = spaces[agent.vehicle_type]
    route = route_operators.mut_reroute(agent.route, space, rng, config, lane_change_s=agent.lane_change_s)
    return on_route(agent, route, space, rng)


def mut_extend_or_truncate(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                           config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Extend or truncate the route at its start or end.

    :returns: The mutant, or None if no change applies.
    """
    space = spaces[agent.vehicle_type]
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config), space, rng)


def mut_lane_change(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Insert a lane change into the route, or remove one.

    :returns: The mutant, or None if the route can neither gain nor lose a lane change.
    """
    space = spaces[agent.vehicle_type]
    route = route_operators.mut_lane_change(agent.route, space, rng, config, lane_change_s=agent.lane_change_s)
    return on_route(agent, route, space, rng)


def mut_s(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
          config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    :returns: The mutant, or None if no lane-change positions fit the new ends.
    """
    space = spaces[agent.vehicle_type]
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config), space, rng)


def mut_lane_change_s(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                      config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Move where one lane change starts by a normal step (``config.sigma_s``),
    within its window on the source lane.

    :returns: The mutant, or None if the route has no lane change.
    """
    if not agent.lane_change_s:
        return None
    space = spaces[agent.vehicle_type]
    index = rng.randrange(len(agent.lane_change_s))
    lane = agent.route.lane_change_indices[index]
    low = space.legs(agent.route, lane_change_s=agent.lane_change_s)[lane].start_s
    high = space.lane_length(agent.route.lanes[lane]) - space.rules.min_lane_change_distance
    change = min(max(agent.lane_change_s[index] + rng.gauss(0, config.sigma_s), low), high)
    changes = agent.lane_change_s[:index] + (change,) + agent.lane_change_s[index + 1:]
    return replace(agent, lane_change_s=changes)


def mut_vehicle_type(agent: NPCVehicleAgent, spaces: Spaces, rng: Random) -> NPCVehicleAgent | None:
    """
    Change to another vehicle type in ``spaces``, with speeds clamped to its maximum.

    :returns: The mutant, or None if ``spaces`` offers no other type.
    """
    types = [vehicle_type for vehicle_type in VehicleType
             if vehicle_type in spaces and vehicle_type is not agent.vehicle_type]
    return retyped(agent, rng.choice(types)) if types else None


def mut_speed_point(agent: NPCVehicleAgent, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig(),
                    spaces: Spaces | None = None) -> NPCVehicleAgent | None:
    """
    Add a speed point, remove one, or move one by a normal step of driven
    distance (``config.sigma_s`` metres) between its neighbours
    (``profile_operators.mut_speed_point`` on the agent's type's space, if
    ``spaces`` has it).

    :returns: The mutant, or None if the change would leave a malformed profile.
    """
    space = spaces.get(agent.vehicle_type) if spaces is not None else None
    return profile_operators.mut_speed_point(agent, rng, config, space)


def mut_npc_agent(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                  config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent:
    """
    Return a mutant: each gene mutates only with its own
    ``config.mutation_rates`` chance, independently of the other genes,
    through an operator drawn by ``config.mutation_weights``.

    Every step keeps the mutant acceptable. When no gene mutated, ``agent``
    is returned unchanged.

    :param Spaces spaces: A route space per vehicle type the mutant may have.
    :rtype: NPCVehicleAgent
    """
    operators = {
        "reroute": lambda child: mut_reroute(child, spaces, rng, config),
        "extend_or_truncate": lambda child: mut_extend_or_truncate(child, spaces, rng, config),
        "lane_change": lambda child: mut_lane_change(child, spaces, rng, config),
        "s": lambda child: mut_s(child, spaces, rng, config),
        "lane_change_s": lambda child: mut_lane_change_s(child, spaces, rng, config),
        "vehicle_type": lambda child: mut_vehicle_type(child, spaces, rng),
        "speed": lambda child: mut_speed(child, rng, config),
        "speed_point": lambda child: mut_speed_point(child, rng, config, spaces),
        "stop": lambda child: mut_stop(child, rng, config),
        "start_t": lambda child: mut_start_t(child, rng, config),
    }
    return vary(agent, config.GENES, operators, config.MUTATION_GENES, config.mutation_weights,
                config.mutation_rates, rng, config.attempts, lambda candidate: acceptable(spaces, candidate, config))
