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
from typing import ClassVar, Mapping

from ...scenario import InvalidRoute, LaneRoute, NPCVehicleAgent, RouteSpace, VehicleType
from ...scenario.npc_agents import VEHICLE_SPECS
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

    :param float sigma_fraction: Unused since speed points are map positions (format
        version 2); speed point moves step by ``sigma_s`` metres of driven
        distance. Kept so existing configurations still construct.
    """

    GENES: ClassVar[tuple[str, ...]] = GENES
    CROSSOVERS: ClassVar[tuple[str, ...]] = (
        "shared_lane", "bridge", "lane_change_s", "vehicle_type", "speed_profile", "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "lane_change", "s", "lane_change_s", "vehicle_type",
        "speed", "speed_point", "stop", "start_t")
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_lane": "route", "bridge": "route", "lane_change_s": "lane_change_s",
        "vehicle_type": "vehicle_type", "speed_profile": "speed_profile", "start_time": "start_t",
    }
    """The gene each crossover changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "reroute": "route", "extend_or_truncate": "route", "lane_change": "route", "s": "route",
        "lane_change_s": "lane_change_s", "vehicle_type": "vehicle_type",
        "speed": "speed_profile", "speed_point": "speed_profile", "stop": "speed_profile", "start_t": "start_t",
    }
    """The gene each mutation changes."""

    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "lane_change_s": 1.0, "vehicle_type": 1.0,
        "speed_profile": 1.0, "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "lane_change_s": 1.0,
        "vehicle_type": 1.0, "speed": 2.0, "speed_point": 1.0, "stop": 0.0, "start_t": 1.0,
    })
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
