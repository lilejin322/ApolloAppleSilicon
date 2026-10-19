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
from typing import ClassVar, Mapping, Sequence

from ...scenario import RIDER_SPECS, BicycleAgent, InvalidRoute, LaneRoute, RiderType, RouteSpace
from ...scenario.bicycle import PivotKey, starts_on_bike_lane
from ...scenario.realism import RIDER_MIX, mapped_speed
from ...utils import require_nonnegative
from ..common import ReplayOperatorConfig

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

    :param Mapping[str, float] rider_type_weights: Rider composition that rider-type
        mutation resamples from (``realism.RIDER_MIX`` by default).
    :param float sigma_turn_dwell: Standard deviation of turn wait steps, in seconds.
    :param float max_turn_dwell: Longest turn wait drawn or stepped to, in seconds.
    :param bool keep_start_kind: Children start on the same kind of lane as their
        (lead) parent: a bike lane or the road (``scenario.bicycle.starts_on_bike_lane``).
        Extending or truncating a route's start, rerouting it, or recombining it
        can otherwise move the start between the two, and whichever move succeeds
        more often would shift the population's share of bike-lane starts away
        from ``BicycleAgent.random``'s ``bike_lane_start_share``.
    """

    GENES: ClassVar[tuple[str, ...]] = GENES
    CROSSOVERS: ClassVar[tuple[str, ...]] = (
        "shared_lane", "bridge", "turn_dwell", "rider_type", "speed_profile", "drive_style", "start_time")
    MUTATIONS: ClassVar[tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "s", "turn_dwell", "rider_type", "speed", "speed_point", "stop",
        "drive_style", "start_t")
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_lane": "route", "bridge": "route", "turn_dwell": "turn_dwell", "rider_type": "rider_type",
        "speed_profile": "speed_profile", "drive_style": "speed_profile", "start_time": "start_t",
    }
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "reroute": "route", "extend_or_truncate": "route", "s": "route", "turn_dwell": "turn_dwell",
        "rider_type": "rider_type", "speed": "speed_profile", "speed_point": "speed_profile",
        "stop": "speed_profile", "drive_style": "speed_profile", "start_t": "start_t",
    }

    min_length: float = 30.0
    max_length: float = 300.0
    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed_profile": 1.0,
        "drive_style": 1.0, "start_time": 1.0,
    })
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "s": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed": 2.0,
        "speed_point": 1.0, "stop": 0.0, "drive_style": 1.0, "start_t": 1.0,
    })
    rider_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(RIDER_MIX))
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    min_speed: float = 1.0
    sigma_turn_dwell: float = 3.0
    max_turn_dwell: float = 30.0
    keep_start_kind: bool = True
    bridge_within_parents: bool = True

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


def keeps_start(space: RouteSpace, candidate: BicycleAgent, parent: BicycleAgent,
                config: BicycleOperatorConfig) -> bool:
    """Whether ``candidate`` starts on the same kind of lane as ``parent`` (``keep_start_kind``)."""
    return (not config.keep_start_kind
            or starts_on_bike_lane(space, candidate.route) == starts_on_bike_lane(space, parent.route))


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
    """
    ``agent`` as another rider type: each target speed moves to the same
    quantile of the new type's speed prior (``realism.mapped_speed``), within
    the type's maximum. Only clamping, and bicycles retyped to e-bikes kept
    bicycle pace while e-bikes retyped to bicycles all rode at the bicycle maximum.
    """
    if rider_type is agent.rider_type:
        return agent
    old, new, top = agent.rider_type.name, rider_type.name, RIDER_SPECS[rider_type].max_speed
    profile = tuple(replace(point, speed=min(mapped_speed(old, new, point.speed), top)) if point.speed > 0 else point
                    for point in agent.speed_profile)
    return replace(agent, rider_type=rider_type, speed_profile=profile)


def rebalance_rider_types(population: Sequence[BicycleAgent], space: RouteSpace, rng: Random,
                          config: BicycleOperatorConfig = BicycleOperatorConfig(), *,
                          redraw: bool = True) -> list[BicycleAgent]:
    """
    Restore a population's rider mix to ``config.rider_type_weights``, as
    ``npc_agent.rebalance_vehicle_types`` does for vehicles. Call it once per
    generation, after variation.

    Rider types differ in turning radius, so a change to the type with the
    larger radius (e-bike) fails on routes with tight turns while the change
    back never does, and e-bikes fade (shalun, riding curb lanes: 18% -> 9% in
    30 neutral generations). Agents of an over-represented type are retyped
    where their route fits the short type; if none fits, a surplus agent is
    replaced by a fresh random agent of the short type that starts on the same
    kind of lane (bike lane or road), so the bike-lane start share is kept.

    :returns: A new list; agents that changed are new objects.
    """
    offered = {name: weight for name, weight in config.rider_type_weights.items() if weight > 0}
    population = list(population)
    if not offered or not population:
        return population
    total = sum(offered.values())
    exact = {name: weight / total * len(population) for name, weight in offered.items()}
    target = {name: int(share) for name, share in exact.items()}
    for name in sorted(exact, key=lambda n: exact[n] - target[n], reverse=True)[:len(population) - sum(target.values())]:
        target[name] += 1
    counts = {name: sum(agent.rider_type.name == name for agent in population) for name in offered}
    unreachable: set[str] = set()
    while True:
        short = [name for name in offered if counts[name] < target[name] and name not in unreachable]
        if not short:
            return population
        wanted = max(short, key=lambda name: target[name] - counts[name])
        donors = [index for index, agent in enumerate(population)
                  if agent.rider_type.name not in offered or counts[agent.rider_type.name] > target[agent.rider_type.name]]
        rng.shuffle(donors)
        child = None
        for index in donors:
            child = retyped(population[index], RiderType[wanted])
            if acceptable(space, child, config):
                break
            child = None
        if child is None and redraw and donors:
            index = donors[0]
            child = _fresh(space, RiderType[wanted], rng, config, starts_on_bike_lane(space, population[index].route))
        if child is None:
            unreachable.add(wanted)
            continue
        old = population[index].rider_type.name
        if old in counts:
            counts[old] -= 1
        counts[wanted] += 1
        population[index] = child


def _fresh(space: RouteSpace, rider_type: RiderType, rng: Random, config: BicycleOperatorConfig,
           on_bike_lane: bool, attempts: int = 20) -> BicycleAgent | None:
    """A random acceptable agent of ``rider_type`` starting on a bike lane or the road, or None."""
    for _ in range(attempts):
        try:
            agent = BicycleAgent.random(space, rng, rider_type=rider_type, max_start_t=config.max_start_t,
                                        min_length=config.min_length, max_length=config.max_length,
                                        min_speed=config.min_speed, bike_lane_start_share=float(on_bike_lane))
        except RuntimeError:
            return None
        if acceptable(space, agent, config) and starts_on_bike_lane(space, agent.route) == on_bike_lane:
            return agent
    return None
