"""
Shared pieces of BicycleAgent crossover and mutation.

Every gene of a bicycle varies: its route, its turn waits, its rider type,
its speed profile, and its start time. As for NPC vehicles, each gene has
its own crossover and mutation chance per call, and every step must keep the
child acceptable: valid in the bicycles' ``RouteSpace``, with a replay path
within the length bounds and a start time within range. All rider types
share one route space.
"""

from dataclasses import dataclass, field, replace
from random import Random
from typing import ClassVar, Mapping

from ...scenario import RIDER_SPECS, BicycleAgent, InvalidRoute, LaneRoute, RiderType, RouteSpace
from ...scenario.bicycle import PivotKey
from ...utils import require_nonnegative
from ..common import ReplayOperatorConfig
from ..profile_operators import clamp_speeds

GENES = ("route", "turn_dwell", "rider_type", "speed_profile", "start_t")
"""Bicycle genes, in the order each call varies them: the route first, since
the turn waits are counted on it, and the speed profile after the rider type
whose limits it must respect."""


@dataclass(frozen=True)
class BicycleOperatorConfig(ReplayOperatorConfig):
    """
    Rates, weights, bounds, and step sizes for BicycleAgent crossover and mutation.

    ``min_length`` and ``max_length`` bound the replay path, as in
    ``BicycleAgent.random``. The other inherited fields are as in ``ReplayOperatorConfig``.

    :param float sigma_turn_dwell: Standard deviation of turn wait steps, in seconds.
    :param float max_turn_dwell: Longest turn wait drawn or stepped to, in seconds.
    """

    GENES: ClassVar[tuple[str, ...]] = GENES
    CROSSOVERS: ClassVar[tuple[str, ...]] = (
        "shared_lane", "bridge", "turn_dwell", "rider_type", "speed_profile", "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "s", "turn_dwell", "rider_type", "speed", "speed_point", "stop",
        "start_t")
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_lane": "route", "bridge": "route", "turn_dwell": "turn_dwell", "rider_type": "rider_type",
        "speed_profile": "speed_profile", "start_time": "start_t",
    }
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "reroute": "route", "extend_or_truncate": "route", "s": "route", "turn_dwell": "turn_dwell",
        "rider_type": "rider_type", "speed": "speed_profile", "speed_point": "speed_profile",
        "stop": "speed_profile", "start_t": "start_t",
    }

    min_length: float = 30.0
    max_length: float = 300.0
    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed_profile": 1.0,
        "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "s": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed": 2.0,
        "speed_point": 1.0, "stop": 0.0, "start_t": 1.0,
    })
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    min_speed: float = 1.0
    sigma_turn_dwell: float = 3.0
    max_turn_dwell: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative(sigma_turn_dwell=self.sigma_turn_dwell, max_turn_dwell=self.max_turn_dwell)


def acceptable(space: RouteSpace, agent: BicycleAgent, config: BicycleOperatorConfig) -> bool:
    """Whether the agent is valid in the space, with its replay path and start time in bounds."""
    if agent.start_t > config.max_start_t:
        return False
    try:
        trajectory = agent.trajectory(space)
    except InvalidRoute:
        return False
    return (config.min_length <= trajectory.path.length <= config.max_length
            and (not config.min_arrival_t or agent.start_t < config.min_arrival_t <= trajectory.arrival_t))


def on_route(agent: BicycleAgent, route: LaneRoute | None, space: RouteSpace, rng: Random,
             config: BicycleOperatorConfig, *, keep: Mapping[PivotKey, float] | None = None,
             ) -> BicycleAgent | None:
    """``agent`` on a proposed ``route`` with repaired turn waits, or None."""
    if route is None:
        return None
    try:
        return agent.with_route(route, space, rng, keep=keep, max_turn_dwell=config.max_turn_dwell)
    except InvalidRoute:
        return None


def retyped(agent: BicycleAgent, rider_type: RiderType) -> BicycleAgent:
    """``agent`` as another rider type, its speeds clamped to that type's maximum."""
    profile = clamp_speeds(agent.speed_profile, RIDER_SPECS[rider_type].max_speed)
    return replace(agent, rider_type=rider_type, speed_profile=profile)
