"""
Shared pieces of PedestrianAgent crossover and mutation.

Every gene of a pedestrian varies: its walk, its walker type, its speed
profile (with its walk style), and its start time. As for NPC vehicles and
bicycles, each gene has its own crossover and mutation chance per call, and
every step must keep the child acceptable: valid on the map's walk network,
with a replay path within the length bounds and a start time within range.
"""

from dataclasses import dataclass, field, replace
from random import Random
from typing import ClassVar, Mapping

from ...scenario.pedestrian import WALKER_SPECS, PedestrianAgent, WalkerType
from ...scenario.realism import EXCURSION_COUNT_MIX, EXCURSION_MIX, WALKER_MIX, mapped_speed
from ...scenario.routing import InvalidRoute
from ...scenario.walkways import CurbPoint, WalkNetwork, WalkRoute
from ...utils import require_nonnegative
from ..common import ReplayOperatorConfig

GENES = ("route", "walker_type", "speed_profile", "start_t")
"""Pedestrian genes, in the order each call varies them: the walk first, and the
speed profile after the walker type whose limits it must respect."""


@dataclass(frozen=True)
class PedestrianOperatorConfig(ReplayOperatorConfig):
    """
    Rates, weights, bounds, and step sizes for PedestrianAgent crossover and mutation.

    ``min_length`` and ``max_length`` bound the replay path, as in
    ``PedestrianAgent.random``; ``sigma_s`` is the step of a walk's start and
    end along the sidewalk. The other inherited fields are as in ``ReplayOperatorConfig``.

    :param Mapping[str, float] walker_type_weights: Walker composition that type
        mutation resamples from (``realism.WALKER_MIX`` by default).
    :param Mapping[str, float] kind_weights: Excursion kinds that kind mutation and
        added excursions draw from (``realism.EXCURSION_MIX`` by default).
    :param Mapping[int, float] count_weights: The share of walks by excursion count
        that count mutation keeps (``realism.EXCURSION_COUNT_MIX`` by default).
    :param float sigma_curb: Step of an excursion's entry or exit along the sidewalk, in metres.
    :param float sigma_turn_back: Step of a turning point, in metres.
    :param float turn_back_probability: Chance that turn-back mutation draws a turning
        point rather than none, as at generation, so the share of walkers who turn
        back stays put.
    :param float redraw_reach: Metres along the sidewalk from the old entry within which a
        redrawn excursion enters.
    :param float max_link: Metres along the sidewalk within which an added excursion enters.
    :param float max_approach: Metres along the sidewalk from an entry or arrival within
        which a moved start or end is drawn.
    """

    GENES: ClassVar[tuple[str, ...]] = GENES
    CROSSOVERS: ClassVar[tuple[str, ...]] = ("shared_sidewalk", "walker_type", "speed_profile", "walk_style",
                                             "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = (
        "shift", "redraw", "kind", "count", "turn_back", "walker_type", "speed", "speed_point", "stop", "walk_style",
        "start_t")
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_sidewalk": "route", "walker_type": "walker_type", "speed_profile": "speed_profile",
        "walk_style": "speed_profile", "start_time": "start_t",
    }
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "shift": "route", "redraw": "route", "kind": "route", "count": "route", "turn_back": "route",
        "walker_type": "walker_type", "speed": "speed_profile", "speed_point": "speed_profile",
        "stop": "speed_profile", "walk_style": "speed_profile", "start_t": "start_t",
    }

    min_length: float = 10.0
    max_length: float = 200.0
    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_sidewalk": 1.0, "walker_type": 1.0, "speed_profile": 1.0, "walk_style": 1.0, "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shift": 3.0, "redraw": 2.0, "kind": 1.0, "count": 1.0, "turn_back": 1.0, "walker_type": 1.0, "speed": 2.0,
        "speed_point": 1.0, "stop": 1.0, "walk_style": 1.0, "start_t": 1.0,
    })
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    walker_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(WALKER_MIX))
    kind_weights: Mapping[str, float] = field(default_factory=lambda: dict(EXCURSION_MIX))
    count_weights: Mapping[int, float] = field(default_factory=lambda: dict(EXCURSION_COUNT_MIX))
    sigma_s: float = 2.0
    sigma_speed: float = 0.2
    min_speed: float = 0.5
    max_dwell: float = 20.0
    sigma_curb: float = 0.5
    sigma_turn_back: float = 1.0
    turn_back_probability: float = 0.05
    redraw_reach: float = 10.0
    max_link: float = 40.0
    max_approach: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative(sigma_curb=self.sigma_curb, sigma_turn_back=self.sigma_turn_back,
                            redraw_reach=self.redraw_reach, max_link=self.max_link, max_approach=self.max_approach)
        if not 0 <= self.turn_back_probability <= 1:
            raise ValueError("turn_back_probability must be within [0, 1]")


def acceptable(network: WalkNetwork, agent: PedestrianAgent, config: PedestrianOperatorConfig) -> bool:
    """Whether the agent is valid on the network, with its replay path and start time in bounds."""
    if agent.start_t > config.max_start_t:
        return False
    try:
        trajectory = agent.trajectory(network)
    except InvalidRoute:
        return False
    return (config.min_length <= trajectory.path.length <= config.max_length
            and (not config.min_arrival_t or agent.start_t < config.min_arrival_t <= trajectory.arrival_t))


def on_route(agent: PedestrianAgent, route: WalkRoute | None, network: WalkNetwork) -> PedestrianAgent | None:
    """``agent`` on a proposed walk with its speed points moved, or None."""
    if route is None:
        return None
    try:
        return agent.with_route(route, network)
    except (InvalidRoute, ValueError):
        return None


def retyped(agent: PedestrianAgent, walker_type: WalkerType) -> PedestrianAgent:
    """
    ``agent`` as another walker type: each target speed moves to the same
    quantile of the new type's speed prior (``realism.mapped_speed``), within
    the type's maximum. Only clamping kept speeds, and type changes soon
    produced runners at walking pace and elderly walkers at 2 m/s (borregas,
    30 generations: 87 of 97 runners and 50 of 123 elderly outside their ranges).
    """
    if walker_type is agent.walker_type:
        return agent
    old, new, top = agent.walker_type.name, walker_type.name, WALKER_SPECS[walker_type].max_speed
    profile = tuple(replace(point, speed=min(mapped_speed(old, new, point.speed), top)) if point.speed > 0 else point
                    for point in agent.speed_profile)
    return replace(agent, walker_type=walker_type, speed_profile=profile)


def rehomed(route: WalkRoute, network: WalkNetwork, rng: Random, config: PedestrianOperatorConfig,
            *, start: bool = False) -> WalkRoute:
    """
    ``route`` after an excursion changed, with its end, and its start if
    ``start``, redrawn where generation draws them when they no longer lie
    there: within ``config.max_approach`` along the sidewalk of the last
    arrival and of the first entry. Points that still lie there stay.
    """
    if not route.excursions:
        return route
    arrival, first = route.excursions[-1].arrival, route.excursions[0].entry
    end, begin = route.end, route.start
    if network.gap(end, arrival) > config.max_approach:
        end = _around(network, arrival, rng, config.max_approach)
    if start and network.gap(begin, first) > config.max_approach:
        begin = _around(network, first, rng, config.max_approach)
    return replace(route, start=begin, end=end)


def _around(network: WalkNetwork, point: CurbPoint, rng: Random, reach: float) -> CurbPoint:
    return network.wrapped(point.sidewalk, point.s + rng.uniform(-reach, reach))
