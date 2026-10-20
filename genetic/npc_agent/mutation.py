"""
Mutation of NPCVehicleAgent chromosomes.

Each operator returns a mutant that changes one gene, or None when it does
not apply. ``mut_npc_agent`` mutates each gene with its rate.
"""

from dataclasses import replace
from random import Random
from typing import List

from ...scenario import NPCVehicleAgent, VehicleType
from .. import profile_operators, route_operators
from ..common import vary
from ..profile_operators import mut_drive_style, mut_speed, mut_start_t, mut_stop
from ...scenario.realism import draw_type
from .common import NPCOperatorConfig, Spaces, acceptable, on_route, retyped


def mut_reroute(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :param NPCVehicleAgent agent: Agent whose route is mutated.
    :param Spaces spaces: Search spaces used to find an alternative lane path.
    :param Random rng: Random number generator used to select the stretch and replacement path.
    :param NPCOperatorConfig config: Configuration controlling route mutation and acceptance.
    :returns: The mutant, or ``None`` if no alternative fits.
    :rtype: NPCVehicleAgent | None
    """
    space = spaces[agent.vehicle_type]
    route = route_operators.mut_reroute(agent.route, space, rng, config, lane_change_s=agent.lane_change_s)
    return on_route(agent, route, space, rng)


def mut_extend_or_truncate(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                           config: NPCOperatorConfig = NPCOperatorConfig(),
                           direction: str | None = None) -> NPCVehicleAgent | None:
    """
    Extend or truncate the route at its start or end.

    :param NPCVehicleAgent agent: Agent whose route is mutated.
    :param Spaces spaces: Search spaces used to mutate the route and update lane-change positions.
    :param Random rng: Random number generator used for route mutation and position sampling.
    :param NPCOperatorConfig config: Configuration controlling route mutation and acceptance.
    :param str | None direction: Mutation direction drawn for the event, as defined by
        ``route_operators.mut_extend_or_truncate``.
    :returns: The mutant, or ``None`` if no change applies.
    :rtype: NPCVehicleAgent | None
    """
    space = spaces[agent.vehicle_type]
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config, direction=direction), space, rng)


def mut_lane_change(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig(),
                    acceptance: float | None = None) -> NPCVehicleAgent | None:
    """
    Insert a lane change into the route, or remove one.

    :param NPCVehicleAgent agent: Agent whose route is mutated.
    :param Spaces spaces: Search spaces used to mutate the route and update lane-change positions.
    :param Random rng: Random number generator used for route mutation and position sampling.
    :param NPCOperatorConfig config: Configuration controlling lane-change mutation and acceptance.
    :param float | None acceptance: Mutation event's uniform draw, passed to
        ``route_operators.mut_lane_change``.
    :returns: The mutant, or ``None`` if the route can neither gain nor lose a lane change.
    :rtype: NPCVehicleAgent | None
    """
    space = spaces[agent.vehicle_type]
    route = route_operators.mut_lane_change(agent.route, space, rng, config, lane_change_s=agent.lane_change_s,
                                            acceptance=acceptance)
    return on_route(agent, route, space, rng)


def mut_s(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
          config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    :param NPCVehicleAgent agent: Agent whose route endpoints are mutated.
    :param Spaces spaces: Search spaces used to constrain endpoints and update lane-change positions.
    :param Random rng: Random number generator used for endpoint mutation and position sampling.
    :param NPCOperatorConfig config: Configuration controlling endpoint mutation and acceptance.
    :returns: The mutant, or ``None`` if no lane-change positions fit the new ends.
    :rtype: NPCVehicleAgent | None
    """
    space = spaces[agent.vehicle_type]
    return on_route(agent, route_operators.mut_s(agent.route, space, rng, config), space, rng)


def mut_lane_change_s(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                      config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent | None:
    """
    Move one lane change's start position by a normal step (``config.sigma_s``),
    within its window on the source lane.

    :param NPCVehicleAgent agent: Agent whose lane-change start position is mutated.
    :param Spaces spaces: Search spaces defining valid lane-change windows.
    :param Random rng: Random number generator used to select a lane change and sample its step.
    :param NPCOperatorConfig config: Configuration controlling the mutation step size.
    :returns: The mutant, or ``None`` if the route has no lane change.
    :rtype: NPCVehicleAgent | None
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


def mut_vehicle_type(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                     config: NPCOperatorConfig = NPCOperatorConfig(),
                     drawn: VehicleType | None = None) -> NPCVehicleAgent | None:
    """
    Resample the vehicle type from ``config.vehicle_type_weights`` using only
    types in ``spaces``, with speeds clamped to the selected type's maximum.

    Sampling from the configured composition preserves the intended population
    mix. The caller's acceptance check rejects types incompatible with the
    route's footprint or turning-radius constraints.

    Draw once per mutation event using ``draw_vehicle_type`` and pass the result
    as ``drawn``. Drawing the current type is a no-op; redrawing until the type
    changes would pull the population mix towards uniform.

    :param NPCVehicleAgent agent: Agent whose vehicle type is mutated.
    :param Spaces spaces: Search spaces defining the vehicle types available for sampling.
    :param Random rng: Random number generator used when ``drawn`` is not provided.
    :param NPCOperatorConfig config: Configuration defining vehicle-type sampling weights.
    :param VehicleType | None drawn: Type already drawn for this mutation event, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if the drawn type is unchanged or no type is available.
    :rtype: NPCVehicleAgent | None
    """
    vehicle_type = drawn if drawn is not None else draw_vehicle_type(spaces, rng, config)
    if vehicle_type is None or vehicle_type is agent.vehicle_type:
        return None
    return retyped(agent, vehicle_type)


def draw_vehicle_type(spaces: Spaces, rng: Random,
                      config: NPCOperatorConfig = NPCOperatorConfig()) -> VehicleType | None:
    """
    Draw a vehicle type from ``config.vehicle_type_weights`` over the types in ``spaces``.

    :param Spaces spaces: Search spaces defining the available vehicle types.
    :param Random rng: Random number generator used for weighted sampling.
    :param NPCOperatorConfig config: Configuration defining vehicle-type sampling weights.
    :returns: The drawn type, or ``None`` if no available type has positive weight.
    :rtype: VehicleType | None
    """
    weights = {name: weight for name, weight in config.vehicle_type_weights.items()
               if VehicleType[name] in spaces}
    if not any(weight > 0 for weight in weights.values()):
        return None
    return VehicleType[draw_type(rng, weights)]


def mut_speed_point(agent: NPCVehicleAgent, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig(),
                    spaces: Spaces | None = None, action: str | None = None) -> NPCVehicleAgent | None:
    """
    Add or remove a speed point, or move one along driven distance by a normal
    step (``config.sigma_s`` metres), staying between its neighbours.

    Delegates to ``profile_operators.mut_speed_point``, using the agent's
    vehicle-type space when available.

    :param NPCVehicleAgent agent: Agent whose speed profile is mutated.
    :param Random rng: Random number generator used to select and apply the mutation.
    :param NPCOperatorConfig config: Configuration controlling speed-point mutation.
    :param Spaces | None spaces: Optional search spaces providing the agent's vehicle-type space.
    :param str | None action: Mutation action passed to ``profile_operators.mut_speed_point``.
    :returns: The mutant, or ``None`` if the change would leave a malformed profile.
    :rtype: NPCVehicleAgent | None
    """
    space = spaces.get(agent.vehicle_type) if spaces is not None else None
    return profile_operators.mut_speed_point(agent, rng, config, space, action)


def mut_npc_agent(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                  config: NPCOperatorConfig = NPCOperatorConfig()) -> NPCVehicleAgent:
    """
    Mutate each gene independently with its own ``config.mutation_rates``
    probability, using an operator drawn by ``config.mutation_weights``.

    Every step keeps the mutant acceptable. If no gene mutates, return
    ``agent`` unchanged.

    :param NPCVehicleAgent agent: Agent whose genes are mutated.
    :param Spaces spaces: A route space per vehicle type the mutant may have.
    :param Random rng: Random number generator used for gene selection, operator selection, and mutation.
    :param NPCOperatorConfig config: Configuration controlling mutation probabilities, operator weights, and acceptance.
    :returns: An acceptable mutant, or ``agent`` unchanged if no mutation succeeds.
    :rtype: NPCVehicleAgent
    """
    draws: List[VehicleType | None] = []  # one type draw per mutation event, kept over retries

    def vehicle_type(child: NPCVehicleAgent) -> NPCVehicleAgent | None:
        """
        Mutate the vehicle type using one cached draw for the mutation event.

        :param NPCVehicleAgent child: Agent whose vehicle type is mutated.
        :returns: The mutant, or ``None`` if the drawn type is unchanged or unavailable.
        :rtype: NPCVehicleAgent | None
        """
        if not draws:
            draws.append(draw_vehicle_type(spaces, rng, config))
        return mut_vehicle_type(child, spaces, rng, config, draws[0])

    acceptance = rng.random()  # lane-change acceptance draw for this mutation event, kept over retries
    direction = rng.choice(("extend", "truncate"))  # extend/truncate direction for this event, kept over retries
    point_action = rng.choice(profile_operators.SPEED_POINT_ACTIONS)  # speed-point action, kept over retries

    operators = {
        "reroute": lambda child: mut_reroute(child, spaces, rng, config),
        "extend_or_truncate": lambda child: mut_extend_or_truncate(child, spaces, rng, config, direction),
        "lane_change": lambda child: mut_lane_change(child, spaces, rng, config, acceptance),
        "s": lambda child: mut_s(child, spaces, rng, config),
        "lane_change_s": lambda child: mut_lane_change_s(child, spaces, rng, config),
        "vehicle_type": vehicle_type,
        "speed": lambda child: mut_speed(child, rng, config, spaces.get(child.vehicle_type)),
        "speed_point": lambda child: mut_speed_point(child, rng, config, spaces, point_action),
        "stop": lambda child: mut_stop(child, rng, config),
        "drive_style": lambda child: mut_drive_style(child, rng),
        "start_t": lambda child: mut_start_t(child, rng, config),
    }
    return vary(agent, config.GENES, operators, config.MUTATION_GENES, config.mutation_weights,
                config.mutation_rates, rng, config.attempts, lambda candidate: acceptable(spaces, candidate, config))
