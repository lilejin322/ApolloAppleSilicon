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
from typing import ClassVar, Mapping, Sequence, Tuple, Dict, List, Set

from ...scenario import InvalidRoute, LaneRoute, NPCVehicleAgent, RouteSpace, VehicleType
from ...scenario.npc_agent import VEHICLE_SPECS
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

    Length bounds apply to the NPC's replay path. Parameters below include all
    fields inherited from ``ReplayOperatorConfig`` and ``OperatorConfig``;
    NPC-specific defaults are noted where those fields are overridden.

    :param float min_length: Minimum accepted replay-path length in metres; defaults to 150.
    :param float max_length: Maximum accepted replay-path length and lane-path search
        distance in metres; defaults to 500.
    :param float max_start_t: Latest accepted start time, in seconds.
    :param int attempts: Tries per operator before the next operator is drawn.
    :param Mapping[str, float] crossover_weights: Relative operator weights within
        each selected gene. Defaults: ``shared_lane`` 3, ``bridge`` 2, and
        ``lane_change_s``, ``vehicle_type``, ``speed_profile``, ``drive_style``,
        ``start_time`` 1. Missing or zero-weight operators are disabled.
    :param Mapping[str, float] mutation_weights: Relative operator weights within
        each selected gene. Defaults: ``reroute`` 3; ``extend_or_truncate``,
        ``lane_change``, ``s``, ``speed`` 2; ``lane_change_s``, ``vehicle_type``,
        ``speed_point``, ``drive_style``, ``start_t`` 1; ``stop`` 0. Missing or
        zero-weight operators are disabled.
    :param float sigma_s: Standard deviation of route position steps, in metres.
    :param float sigma_t: Standard deviation of start time steps, in seconds.
    :param int bridge_candidates: Cheapest lane paths considered when bridging or rerouting.
    :param float max_detour: A rerouted stretch is at most this share longer than the
        shortest of it and its alternatives (``inf`` disables). Real routes are
        near-shortest; unbounded alternatives made routes loop and grow over generations.
    :param float lane_change_odds: Odds ``p / (1 - p)`` of a lane change where one is
        possible, as in generation (``lane_change_probability`` 0.2 -> 0.25). Lane-change
        mutation accepts a child with ``k`` more lane changes with probability
        ``lane_change_odds ** k``, so lane changes do not pile up over generations.
    :param bool bridge_within_parents: Bridge crossover keeps only children whose
        length lies in the parents' range widened by ``BRIDGE_BOUND_SLACK`` times
        the longer parent's length. Where
        parents rarely meet, the only bridges are long detours, and bridged children
        grew routes by a step (shalun bicycles on curb lanes: +27 m per child, median
        150 -> 190 m in five generations). Off by default: vehicle routes stay stable
        without it.
    :param Mapping[str, float] crossover_rates: Per-gene crossover probabilities for
        ``route``, ``lane_change_s``, ``vehicle_type``, ``speed_profile``, and
        ``start_t``; each defaults to 0.5. Missing genes are disabled.
    :param Mapping[str, float] mutation_rates: Per-gene mutation probabilities for
        the same genes as ``crossover_rates``; each defaults to 0.3. Missing genes
        are disabled.
    :param float sigma_speed: Standard deviation of target speed steps, in m/s.
    :param float min_speed: Lowest nonzero target speed drawn or stepped to,
        in m/s; defaults to 4.5.
    :param float max_dwell: Longest stop drawn or stepped to, in seconds.
    :param float min_arrival_t: Earliest route-end time in scenario seconds; when positive,
        the agent must also start before it. Zero disables the horizon check.
    :param float route_swap_rate: Chance that a route crossover whose operators
        all failed takes the other parent's whole route instead.
    :param Mapping[str, float] vehicle_type_weights: Traffic composition (type name ->
        weight) that type mutation resamples from, so a population keeps this mix;
        default urban US vehicle-miles (``realism.VEHICLE_MIX``).
    :param bool cross_type_routes: Let route crossover combine parents of different
        vehicle types. Off by default: a small type takes a large type's roads freely but
        not the reverse, so routes drift towards the roads large vehicles fit.
    :param float sigma_fraction: Unused compatibility field, defaulting to 0.1;
        must be finite and nonnegative. Speed-point moves use ``sigma_s`` metres
        of driven distance since map-position speed profiles (format version 2).
    :raises ValueError: If inherited bounds, weights, gene names, or probabilities
        are invalid, or speed, dwell, arrival-time, or ``sigma_fraction`` values
        are nonfinite or negative.
    """

    GENES: ClassVar[Tuple[str, ...]] = GENES
    """NPC gene names, in the order each variation call processes them."""
    CROSSOVERS: ClassVar[Tuple[str, ...]] = (
        "shared_lane", "bridge", "lane_change_s", "vehicle_type", "speed_profile", "drive_style", "start_time")
    """Supported crossover names used as keys in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "lane_change", "s", "lane_change_s", "vehicle_type",
        "speed", "speed_point", "stop", "drive_style", "start_t")
    """Supported mutation names used as keys in ``mutation_weights``."""
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
    """Relative crossover operator weights within each selected gene; missing or zero weights disable operators."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "lane_change_s": 1.0,
        "vehicle_type": 1.0, "speed": 2.0, "speed_point": 1.0, "stop": 0.0, "drive_style": 1.0, "start_t": 1.0,
    })
    """Relative mutation operator weights within each selected gene; missing or zero weights disable operators."""
    vehicle_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(VEHICLE_MIX))
    """Vehicle-type sampling and population rebalancing weights, keyed by ``VehicleType`` member names."""
    cross_type_routes: bool = False
    """Whether route crossover may combine parents of different vehicle types."""
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    """Per-gene crossover probabilities; missing genes are disabled."""
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    """Per-gene mutation probabilities; missing genes are disabled."""
    min_length: float = 150.0
    """Minimum accepted replay-path length, in metres."""
    sigma_fraction: float = 0.1
    """Unused compatibility field; speed-point position steps use ``sigma_s`` metres instead."""
    min_speed: float = 4.5
    """Lowest nonzero target speed drawn or stepped to, in metres per second."""

    def __post_init__(self) -> None:
        """
        Validate inherited settings and require a finite, nonnegative ``sigma_fraction``.
        """
        super().__post_init__()
        require_nonnegative(sigma_fraction=self.sigma_fraction)


def acceptable(spaces: Spaces, agent: NPCVehicleAgent, config: NPCOperatorConfig) -> bool:
    """
    Whether the agent is valid for its type's space, with its replay path and start time in bounds.

    :param Spaces spaces: Route spaces keyed by vehicle type; the agent's type
        must be present to construct and validate its trajectory.
    :param NPCVehicleAgent agent: Candidate whose trajectory and timing are checked.
    :param NPCOperatorConfig config: Inclusive replay-path length bounds in metres
        and maximum start time in seconds. A positive ``min_arrival_t`` also requires
        the agent to start before that time and arrive at or after it; zero disables this check.
    :returns: True if all checks pass; False if the vehicle type has no route space,
        trajectory construction raises ``InvalidRoute``, or any length or timing
        bound is violated.
    :rtype: bool
    """
    if agent.start_t > config.max_start_t or agent.vehicle_type not in spaces:
        return False
    try:
        trajectory = agent.trajectory(spaces[agent.vehicle_type])
    except InvalidRoute:
        return False
    return (config.min_length <= trajectory.path.length <= config.max_length
            and (not config.min_arrival_t or agent.start_t < config.min_arrival_t <= trajectory.arrival_t))


def positions(agent: NPCVehicleAgent) -> Dict[Tuple[str, str], float]:
    """
    The agent's lane-change positions by (source lane, target lane).

    :param NPCVehicleAgent agent: Agent supplying the ordered lane-change pairs
        and their corresponding start positions.
    :returns: A dictionary mapping each source/target lane-ID pair to the
        lane-change start position on the source lane, in metres.
    :rtype: Dict[Tuple[str, str], float]
    """
    return dict(zip(agent.route.lane_change_pairs, agent.lane_change_s))


def on_route(agent: NPCVehicleAgent, route: LaneRoute | None, space: RouteSpace, rng: Random, *,
             keep: Mapping[Tuple[str, str], float] | None = None) -> NPCVehicleAgent | None:
    """
    ``agent`` on a proposed ``route`` with repaired lane-change positions, or None.

    :param NPCVehicleAgent agent: Source agent supplying the vehicle type, start
        time, speed profile, and default lane-change positions.
    :param LaneRoute | None route: Proposed lane route, or None if no route was produced.
    :param RouteSpace space: Route space for the agent's vehicle type, used to repair
        lane-change positions, rebase the speed profile, and validate the candidate.
    :param Random rng: Random source for sampling replacement lane-change positions.
    :param Mapping[Tuple[str, str], float] | None keep: Positions to reuse where
        feasible, keyed by source/target lane-ID pairs and measured in metres on
        the source lane. None uses the agent's own positions; an empty mapping
        samples all positions. Inherited positions may be discarded during repair.
    :returns: An agent with the proposed route, repaired lane-change positions,
        and rebased speed profile; None if ``route`` is None or ``with_route``
        raises ``InvalidRoute``. Operator length and timing bounds are not checked here.
    :rtype: NPCVehicleAgent | None
    """
    if route is None:
        return None
    try:
        return agent.with_route(route, space, rng, keep=keep)
    except InvalidRoute:
        return None


def retyped(agent: NPCVehicleAgent, vehicle_type: VehicleType) -> NPCVehicleAgent:
    """
    ``agent`` as another vehicle type, its speeds clamped to that type's maximum.

    :param NPCVehicleAgent agent: Source agent whose vehicle type and target speeds
        are updated; route, lane-change positions, and start time are preserved.
    :param VehicleType vehicle_type: Target vehicle type supplying the maximum
        speed used to clamp the speed profile.
    :returns: A new agent with the target vehicle type and clamped speed profile,
        even if the type is unchanged. Route and trajectory validity for the new
        vehicle type are not checked here.
    :rtype: NPCVehicleAgent
    """
    profile = clamp_speeds(agent.speed_profile, VEHICLE_SPECS[vehicle_type].max_speed)
    return replace(agent, vehicle_type=vehicle_type, speed_profile=profile)


def rebalance_vehicle_types(population: Sequence[NPCVehicleAgent], spaces: Spaces, rng: Random,
                            config: NPCOperatorConfig = NPCOperatorConfig(), *,
                            redraw: bool = True) -> List[NPCVehicleAgent]:
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

    :param Sequence[NPCVehicleAgent] population: Agents to rebalance without changing
        the input sequence or its size.
    :param Spaces spaces: Route spaces keyed by vehicle type; only types present
        here with positive configured weights contribute to the target mix.
    :param Random rng: Random source for donor ordering and replacement generation.
    :param NPCOperatorConfig config: Supplies vehicle-type weights, replacement
        generation settings, and acceptance bounds. Weight keys name ``VehicleType``
        members; eligible weights are normalized into integer target counts, with
        leftover slots allocated by largest fractional remainder.
    :param bool redraw: Allow fresh-agent generation when no donor can be retyped
        acceptably; False only permits retyping existing agents.
    :returns: A new list of the same size; retyped or replaced agents are new objects
        and unchanged agents are reused. Empty populations or no eligible types
        return a shallow copy. Types that cannot be filled are skipped, so the
        resulting mix may still differ from the target.
    :rtype: List[NPCVehicleAgent]
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
    unreachable: Set[str] = set()
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
    """
    Generate a random acceptable NPC vehicle agent, or ``None`` if all attempts fail.

    :param Spaces spaces: Search spaces used to sample the agent.
    :param VehicleType vehicle_type: Vehicle type of the generated agent.
    :param Random rng: Random number generator used for sampling.
    :param NPCOperatorConfig config: Configuration controlling agent generation and acceptance.
    :param int attempts: Maximum number of generation attempts.
    :returns: An acceptable agent, or ``None`` if no attempt succeeds.
    :rtype: NPCVehicleAgent | None
    """
    for _ in range(attempts):
        try:
            agent = NPCVehicleAgent.random(spaces[vehicle_type], rng, vehicle_type=vehicle_type,
                                           max_start_t=config.max_start_t, min_speed=config.min_speed)
        except InvalidRoute:
            continue
        if acceptable(spaces, agent, config):
            return agent
    return None
