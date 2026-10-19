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
from ..profile_operators import mut_drive_style, mut_speed, mut_start_t, mut_stop
from ...scenario.realism import draw_type
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
                           config: NPCOperatorConfig = NPCOperatorConfig(),
                           direction: str | None = None) -> NPCVehicleAgent | None:
    """
    Extend or truncate the route at its start or end (``direction`` as in
    ``route_operators.mut_extend_or_truncate``: the mutation event's draw).

    :returns: The mutant, or None if no change applies.
    """
    space = spaces[agent.vehicle_type]
    return on_route(agent, route_operators.mut_extend_or_truncate(agent.route, space, rng, config, direction=direction), space, rng)


def mut_lane_change(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig(),
                    acceptance: float | None = None) -> NPCVehicleAgent | None:
    """
    Insert a lane change into the route, or remove one (``route_operators.mut_lane_change``;
    ``acceptance`` is the mutation event's uniform draw).

    :returns: The mutant, or None if the route can neither gain nor lose a lane change.
    """
    space = spaces[agent.vehicle_type]
    route = route_operators.mut_lane_change(agent.route, space, rng, config, lane_change_s=agent.lane_change_s,
                                            acceptance=acceptance)
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


def mut_vehicle_type(agent: NPCVehicleAgent, spaces: Spaces, rng: Random,
                     config: NPCOperatorConfig = NPCOperatorConfig(),
                     drawn: VehicleType | None = None) -> NPCVehicleAgent | None:
    """
    Resample the vehicle type from ``config.vehicle_type_weights`` (types in
    ``spaces`` only), with speeds clamped to its maximum. Resampling from the
    composition, rather than picking another type uniformly, keeps a
    population's mix at those weights over generations. The caller's
    acceptance check rejects a type the route does not fit (footprint, turning radius).

    A mutation event must draw once (``drawn``, see ``draw_vehicle_type``): a
    draw of the current type is a no-op. Redrawing on each retry until the
    type changes would force a change and pull the mix towards uniform.

    :returns: The mutant, or None if the draw is the current type or ``spaces`` offers no type.
    """
    vehicle_type = drawn if drawn is not None else draw_vehicle_type(spaces, rng, config)
    if vehicle_type is None or vehicle_type is agent.vehicle_type:
        return None
    return retyped(agent, vehicle_type)


def draw_vehicle_type(spaces: Spaces, rng: Random,
                      config: NPCOperatorConfig = NPCOperatorConfig()) -> VehicleType | None:
    """One draw from ``config.vehicle_type_weights`` over the types in ``spaces``, or None if none has weight."""
    weights = {name: weight for name, weight in config.vehicle_type_weights.items()
               if VehicleType[name] in spaces}
    if not any(weight > 0 for weight in weights.values()):
        return None
    return VehicleType[draw_type(rng, weights)]


def mut_speed_point(agent: NPCVehicleAgent, rng: Random,
                    config: NPCOperatorConfig = NPCOperatorConfig(),
                    spaces: Spaces | None = None, action: str | None = None) -> NPCVehicleAgent | None:
    """
    Add a speed point, remove one, or move one by a normal step of driven
    distance (``config.sigma_s`` metres) between its neighbours
    (``profile_operators.mut_speed_point`` on the agent's type's space, if
    ``spaces`` has it).

    :returns: The mutant, or None if the change would leave a malformed profile.
    """
    space = spaces.get(agent.vehicle_type) if spaces is not None else None
    return profile_operators.mut_speed_point(agent, rng, config, space, action)


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
    draws: list[VehicleType | None] = []  # one type draw per mutation event, kept over retries

    def vehicle_type(child: NPCVehicleAgent) -> NPCVehicleAgent | None:
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
