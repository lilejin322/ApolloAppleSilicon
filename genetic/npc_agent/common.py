"""
Shared pieces of NPCVehicleAgent crossover and mutation.

Every gene of an NPC varies: its route, lane-change positions, vehicle type,
speed profile, and start time. Each gene has its own crossover chance
(``crossover_rates``) and mutation chance (``mutation_rates``), drawn
independently per call, so a child may change several genes at once or
none; a gene with several operators draws one by weight. Every step
must keep the child acceptable: valid for its vehicle type's ``RouteSpace``,
with a replay path within the length bounds and a start time within range.
"""

from dataclasses import dataclass, field, replace
from random import Random
from typing import ClassVar, Mapping, Sequence

from ...scenario import InvalidRoute, LaneRoute, NPCVehicleAgent, RouteSpace, VehicleType
from ...scenario.npc_agents import VEHICLE_SPECS
from ...scenario.realism import VEHICLE_MIX
from ...utils import require_nonnegative
from ..common import ReplayOperatorConfig
from ..profile_operators import clamp_speeds

GENES = ("route", "lane_change_s", "vehicle_type", "speed_profile", "start_t")
"""NPC genes, in the order each call varies them: the route first, since the
other genes are checked on it, and the speed profile after the vehicle type
whose limits it must respect."""

Spaces = Mapping[VehicleType, RouteSpace]
"""A route space per vehicle type, built with ``NPCVehicleAgent.rules``. Only
the types present are offered to vehicle-type crossover and mutation."""


@dataclass(frozen=True)
class NPCOperatorConfig(ReplayOperatorConfig):
    """
    Rates, weights, bounds, and step sizes for NPCVehicleAgent crossover and mutation.

    ``min_length`` and ``max_length`` bound the replay path, as in
    ``NPCVehicleAgent.random``. The other fields are as in ``ReplayOperatorConfig``.

    :param Mapping[str, float] vehicle_type_weights: Traffic composition (type name ->
        weight) that type mutation resamples from, so a population keeps this mix;
        default urban US vehicle-miles (``realism.VEHICLE_MIX``).
    :param bool cross_type_routes: Let route crossover combine parents of different
        vehicle types. Off by default: a small type takes a large type's roads freely but
        not the reverse, so routes drift towards the roads large vehicles fit.
    :param float sigma_fraction: Unused since speed points are map positions (format
        version 2); speed point moves step by ``sigma_s`` metres of driven
        distance. Kept so existing configurations still construct.
    """

    GENES: ClassVar[tuple[str, ...]] = GENES
    CROSSOVERS: ClassVar[tuple[str, ...]] = (
        "shared_lane", "bridge", "lane_change_s", "vehicle_type", "speed_profile", "drive_style", "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "lane_change", "s", "lane_change_s", "vehicle_type",
        "speed", "speed_point", "stop", "drive_style", "start_t")
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_lane": "route", "bridge": "route", "lane_change_s": "lane_change_s",
        "vehicle_type": "vehicle_type", "speed_profile": "speed_profile", "drive_style": "speed_profile",
        "start_time": "start_t",
    }
    """The gene each crossover changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "reroute": "route", "extend_or_truncate": "route", "lane_change": "route", "s": "route",
        "lane_change_s": "lane_change_s", "vehicle_type": "vehicle_type",
        "speed": "speed_profile", "speed_point": "speed_profile", "stop": "speed_profile",
        "drive_style": "speed_profile", "start_t": "start_t",
    }
    """The gene each mutation changes."""

    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "lane_change_s": 1.0, "vehicle_type": 1.0,
        "speed_profile": 1.0, "drive_style": 1.0, "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "lane_change_s": 1.0,
        "vehicle_type": 1.0, "speed": 2.0, "speed_point": 1.0, "stop": 0.0, "drive_style": 1.0, "start_t": 1.0,
    })
    vehicle_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(VEHICLE_MIX))
    cross_type_routes: bool = False
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    min_length: float = 150.0
    sigma_fraction: float = 0.1
    min_speed: float = 4.5

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative(sigma_fraction=self.sigma_fraction)


def acceptable(spaces: Spaces, agent: NPCVehicleAgent, config: NPCOperatorConfig) -> bool:
    """Whether the agent is valid for its type's space, with its replay path and start time in bounds."""
    if agent.start_t > config.max_start_t or agent.vehicle_type not in spaces:
        return False
    try:
        trajectory = agent.trajectory(spaces[agent.vehicle_type])
    except InvalidRoute:
        return False
    return (config.min_length <= trajectory.path.length <= config.max_length
            and (not config.min_arrival_t or agent.start_t < config.min_arrival_t <= trajectory.arrival_t))


def positions(agent: NPCVehicleAgent) -> dict[tuple[str, str], float]:
    """The agent's lane-change positions by (source lane, target lane)."""
    return dict(zip(agent.route.lane_change_pairs, agent.lane_change_s))


def on_route(agent: NPCVehicleAgent, route: LaneRoute | None, space: RouteSpace, rng: Random, *,
             keep: Mapping[tuple[str, str], float] | None = None) -> NPCVehicleAgent | None:
    """``agent`` on a proposed ``route`` with repaired lane-change positions, or None."""
    if route is None:
        return None
    try:
        return agent.with_route(route, space, rng, keep=keep)
    except InvalidRoute:
        return None


def retyped(agent: NPCVehicleAgent, vehicle_type: VehicleType) -> NPCVehicleAgent:
    """``agent`` as another vehicle type, its speeds clamped to that type's maximum."""
    profile = clamp_speeds(agent.speed_profile, VEHICLE_SPECS[vehicle_type].max_speed)
    return replace(agent, vehicle_type=vehicle_type, speed_profile=profile)


def rebalance_vehicle_types(population: Sequence[NPCVehicleAgent], spaces: Spaces, rng: Random,
                            config: NPCOperatorConfig = NPCOperatorConfig(), *,
                            redraw: bool = True) -> list[NPCVehicleAgent]:
    """
    Restore a population's vehicle mix to ``config.vehicle_type_weights``
    (types in ``spaces``), by retyping agents of over-represented types whose
    routes fit an under-represented type (``acceptable``); routes and other
    genes are kept. Call it once per generation, after variation.

    Per-agent operators cannot keep the mix on their own. A type the route
    does not fit is rejected, so changes towards large vehicles fail more
    often than changes back, and large vehicles fade over generations. A
    per-type acceptance correction is only exact while the population's routes
    keep their generation-0 distribution, which crossover does not preserve.
    When no surplus agent's route fits a short type, a surplus agent is replaced by
    a fresh random agent of that type (``NPCVehicleAgent.random``, as in generation
    0) if ``redraw``; otherwise that type stays short. Without the redraw, types
    whose roads few routes reach (trucks on shalun) faded once route crossover
    stopped mixing types: TRUCK 22 -> 5 of 300 in 30 generations.

    :returns: A new list; agents that changed type are new objects.
    """
    offered = {name: weight for name, weight in config.vehicle_type_weights.items()
               if weight > 0 and VehicleType[name] in spaces}
    population = list(population)
    if not offered or not population:
        return population
    total = sum(offered.values())
    exact = {name: weight / total * len(population) for name, weight in offered.items()}
    target = {name: int(share) for name, share in exact.items()}
    for name in sorted(exact, key=lambda n: exact[n] - target[n], reverse=True)[:len(population) - sum(target.values())]:
        target[name] += 1
    counts = {name: sum(agent.vehicle_type.name == name for agent in population) for name in offered}
    unreachable: set[str] = set()
    while True:
        short = [name for name in offered if counts[name] < target[name] and name not in unreachable]
        if not short:
            return population
        wanted = max(short, key=lambda name: target[name] - counts[name])
        donors = [index for index, agent in enumerate(population)
                  if agent.vehicle_type.name not in offered or counts[agent.vehicle_type.name] > target[agent.vehicle_type.name]]
        rng.shuffle(donors)
        for index in donors:
            child = retyped(population[index], VehicleType[wanted])
            if acceptable(spaces, child, config):
                old = population[index].vehicle_type.name
                if old in counts:
                    counts[old] -= 1
                counts[wanted] += 1
                population[index] = child
                break
        else:
            fresh = _fresh(spaces, VehicleType[wanted], rng, config) if redraw and donors else None
            if fresh is None:
                unreachable.add(wanted)
                continue
            index = donors[0]
            old = population[index].vehicle_type.name
            if old in counts:
                counts[old] -= 1
            counts[wanted] += 1
            population[index] = fresh


def _fresh(spaces: Spaces, vehicle_type: VehicleType, rng: Random, config: NPCOperatorConfig,
           attempts: int = 20) -> NPCVehicleAgent | None:
    """A random acceptable agent of ``vehicle_type``, or ``None``."""
    for _ in range(attempts):
        try:
            agent = NPCVehicleAgent.random(spaces[vehicle_type], rng, vehicle_type=vehicle_type,
                                           max_start_t=config.max_start_t, min_speed=config.min_speed)
        except InvalidRoute:
            continue
        if acceptable(spaces, agent, config):
            return agent
    return None
