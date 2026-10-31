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
from typing import ClassVar, Dict, List, Mapping, Sequence, Set, Tuple

from ...scenario import RIDER_SPECS, BicycleAgent, InvalidRoute, LaneRoute, RiderType, RouteSpace
from ...scenario.bicycle import PivotKey, starts_on_bike_lane
from ...scenario.replay import SpeedPoint
from ...scenario.realism import RIDER_MIX, mapped_speed
from ...utils import require_nonnegative
from ..common import LaneRouteConfig, ReplayOperatorConfig, replay_acceptable

GENES: Tuple[str, ...] = ("route", "turn_dwell", "rider_type", "speed_profile", "start_t")
"""Bicycle genes, in the order each call varies them: the route first, since
the turn waits are counted on it, and the speed profile after the rider type
whose limits it must respect."""


@dataclass(frozen=True)
class BicycleOperatorConfig(LaneRouteConfig, ReplayOperatorConfig):
    """
    Rates, weights, bounds, and step sizes for BicycleAgent crossover and mutation.

    Length bounds apply to the bicycle's replay path. Parameters below include
    all fields inherited from ``LaneRouteConfig``, ``ReplayOperatorConfig``, and ``OperatorConfig``;
    bicycle-specific defaults are noted where those fields are overridden.

    :param float min_length: Minimum accepted replay-path length in metres; defaults to 30.
    :param float max_length: Maximum accepted replay-path length and lane-path search
        distance in metres; defaults to 300.
    :param float max_start_t: Latest accepted start time, in seconds.
    :param int attempts: Tries per operator before the next operator is drawn.
    :param Mapping[str, float] crossover_weights: Relative operator-selection weights
        within each selected gene. Defaults: ``shared_lane`` 3, ``bridge`` 2, and
        ``turn_dwell``, ``rider_type``, ``speed_profile``, ``drive_style``, ``start_time`` 1.
        Missing or zero-weight operators are disabled.
    :param Mapping[str, float] mutation_weights: Relative operator-selection weights
        within each selected gene. Defaults: ``reroute`` 3; ``extend_or_truncate``,
        ``s``, ``speed`` 2; ``turn_dwell``, ``rider_type``, ``speed_point``,
        ``drive_style``, ``start_t`` 1; ``stop`` 0. Missing or zero-weight operators
        are disabled.
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
    :param bool bridge_within_parents: Restrict bridged route lengths to the parents'
        length range widened by ``route_operators.BRIDGE_BOUND_SLACK`` times the
        longer parent's length; defaults to True.
    :param Mapping[str, float] crossover_rates: Per-gene crossover probabilities
        for ``route``, ``turn_dwell``, ``rider_type``, ``speed_profile``, and ``start_t``;
        each defaults to 0.5. Missing genes are disabled.
    :param Mapping[str, float] mutation_rates: Per-gene mutation probabilities
        for the same genes as ``crossover_rates``; each defaults to 0.3.
        Missing genes are disabled.
    :param float sigma_speed: Standard deviation of target speed steps, in m/s.
    :param float min_speed: Lowest nonzero target speed drawn or stepped to,
        in m/s; defaults to 1.
    :param float max_dwell: Longest stop drawn or stepped to, in seconds.
    :param float route_swap_rate: Chance that a route crossover whose operators
        all failed takes the other parent's whole route instead.
    :param float min_arrival_t: Earliest route-end time in scenario seconds; when positive,
        the agent must also start before it. Zero disables the horizon check.
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
    :raises ValueError: If inherited bounds, weights, gene names, or probabilities
        are invalid, or speed, dwell, or arrival-time parameters are nonfinite
        or negative. Turn-dwell parameters must also be finite and nonnegative.
    """

    GENES: ClassVar[Tuple[str, ...]] = GENES
    """Bicycle gene names, in the order each variation call processes them."""
    CROSSOVERS: ClassVar[Tuple[str, ...]] = (
        "shared_lane", "bridge", "turn_dwell", "rider_type", "speed_profile", "drive_style", "start_time")
    """Supported crossover names used as keys in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = (
        "reroute", "extend_or_truncate", "s", "turn_dwell", "rider_type", "speed", "speed_point", "stop",
        "drive_style", "start_t")
    """Supported mutation names used as keys in ``mutation_weights``."""
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_lane": "route", "bridge": "route", "turn_dwell": "turn_dwell", "rider_type": "rider_type",
        "speed_profile": "speed_profile", "drive_style": "speed_profile", "start_time": "start_t",
    }
    """Map each crossover operator to the bicycle gene it changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "reroute": "route", "extend_or_truncate": "route", "s": "route", "turn_dwell": "turn_dwell",
        "rider_type": "rider_type", "speed": "speed_profile", "speed_point": "speed_profile",
        "stop": "speed_profile", "drive_style": "speed_profile", "start_t": "start_t",
    }
    """Map each mutation operator to the bicycle gene it changes."""

    min_length: float = 30.0
    """Shortest accepted replay path length, in metres."""
    max_length: float = 300.0
    """Longest accepted replay path length and lane-path search distance, in metres."""
    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed_profile": 1.0,
        "drive_style": 1.0, "start_time": 1.0,
    })
    """Relative selection weights of crossover operators within each selected gene."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "s": 2.0, "turn_dwell": 1.0, "rider_type": 1.0, "speed": 2.0,
        "speed_point": 1.0, "stop": 0.0, "drive_style": 1.0, "start_t": 1.0,
    })
    """Relative selection weights of mutation operators; zero disables an operator."""
    rider_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(RIDER_MIX))
    """Rider-type composition used for weighted type sampling."""
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    """Independent crossover probability per gene."""
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    """Independent mutation probability per gene."""
    min_speed: float = 1.0
    """Lowest nonzero target speed drawn or stepped to, in m/s."""
    sigma_turn_dwell: float = 3.0
    """Standard deviation of turn wait steps, in seconds."""
    max_turn_dwell: float = 30.0
    """Longest turn wait drawn or stepped to, in seconds."""
    keep_start_kind: bool = True
    """Whether route variation preserves the parent's start-lane kind: bike lane or road."""
    bridge_within_parents: bool = True
    """Whether bridge crossover constrains child length to the widened parental range."""

    def __post_init__(self) -> None:
        """
        Validate inherited settings and require finite, nonnegative turn-dwell parameters.

        :raises ValueError: If inherited validation fails, or a turn-dwell parameter is nonfinite or negative.
        """
        super().__post_init__()
        require_nonnegative(sigma_turn_dwell=self.sigma_turn_dwell, max_turn_dwell=self.max_turn_dwell)


def acceptable(space: RouteSpace, agent: BicycleAgent, config: BicycleOperatorConfig) -> bool:
    """
    Check trajectory validity, replay-path length, start time, and the optional arrival horizon.

    :param RouteSpace space: Bicycle route space used to construct and validate the trajectory.
    :param BicycleAgent agent: Candidate whose trajectory and timing are checked.
    :param BicycleOperatorConfig config: Inclusive replay-path length bounds in metres
        and maximum start time in seconds. A positive ``min_arrival_t`` also requires
        the agent to start before that time and arrive at or after it; zero disables this check.
    :returns: True if all checks pass; False if trajectory construction raises
        ``InvalidRoute`` or any length or timing bound is violated.
    :rtype: bool
    """
    return replay_acceptable(agent, lambda: agent.trajectory(space), config)


def keeps_start(space: RouteSpace, candidate: BicycleAgent, parent: BicycleAgent,
                config: BicycleOperatorConfig) -> bool:
    """
    Whether ``candidate`` starts on the same kind of lane as ``parent`` (``keep_start_kind``).

    :param RouteSpace space: Route space used to classify each route's start lane.
    :param BicycleAgent candidate: Proposed child whose start-lane kind is checked.
    :param BicycleAgent parent: Leading parent supplying the reference start-lane kind.
    :param BicycleOperatorConfig config: Controls whether start-lane kind must be
        preserved through ``keep_start_kind``.
    :returns: True if the check is disabled or both agents start on the same kind
        of lane (bike lane or road); otherwise False.
    :rtype: bool
    """
    return (not config.keep_start_kind
            or starts_on_bike_lane(space, candidate.route) == starts_on_bike_lane(space, parent.route))


def on_route(agent: BicycleAgent, route: LaneRoute | None, space: RouteSpace, rng: Random,
             config: BicycleOperatorConfig, *, keep: Mapping[PivotKey, float] | None = None,
             ) -> BicycleAgent | None:
    """
    ``agent`` on a proposed ``route`` with repaired turn waits, or None.

    Speed-profile positions are rebased onto the proposed route. The rider type
    and start time are preserved; operator length and timing bounds are not checked.

    :param BicycleAgent agent: Source agent supplying the rider type, speed profile,
        start time, and default turn waits.
    :param LaneRoute | None route: Proposed lane route, or None if no route was produced.
    :param RouteSpace space: Bicycle route space used to locate pivots, rebase the
        speed profile, and validate the resulting agent.
    :param Random rng: Random source for new turn waits, drawn uniformly from
        ``[0, config.max_turn_dwell]``.
    :param BicycleOperatorConfig config: Supplies ``max_turn_dwell``, the upper bound
        in seconds for newly sampled turn waits.
    :param Mapping[PivotKey, float] | None keep: Turn waits in seconds to preserve,
        keyed by connector lane ID and pivot index. None uses the source agent's
        waits; an empty mapping resamples every wait on the proposed route.
    :returns: An agent with the proposed route, repaired turn waits, and rebased
        speed profile; None if ``route`` is None or ``with_route`` raises ``InvalidRoute``.
    :rtype: BicycleAgent | None
    """
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

    :param BicycleAgent agent: Source agent whose positive target speeds are mapped;
        stop points, route, turn waits, and start time are preserved.
    :param RiderType rider_type: Target rider type supplying the speed prior and
        maximum speed used to map and clamp positive target speeds.
    :returns: A copy with the target rider type and mapped speed profile, or the
        original agent if its rider type already matches. Route and trajectory
        validity for the new rider type are not checked here.
    :rtype: BicycleAgent
    """
    if rider_type is agent.rider_type:
        return agent
    old: str
    new: str
    top: float
    old, new, top = agent.rider_type.name, rider_type.name, RIDER_SPECS[rider_type].max_speed
    profile: Tuple[SpeedPoint, ...] = tuple(replace(point, speed=min(mapped_speed(old, new, point.speed), top)) if point.speed > 0 else point
                    for point in agent.speed_profile)
    return replace(agent, rider_type=rider_type, speed_profile=profile)


def rebalance_rider_types(population: Sequence[BicycleAgent], space: RouteSpace, rng: Random,
                          config: BicycleOperatorConfig = BicycleOperatorConfig(), *,
                          redraw: bool = True) -> List[BicycleAgent]:
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

    :param Sequence[BicycleAgent] population: Agents to rebalance without changing
        the input sequence or its size.
    :param RouteSpace space: Bicycle route space used to validate retyped agents
        and generate replacement routes.
    :param Random rng: Random source for donor ordering and replacement generation.
    :param BicycleOperatorConfig config: Supplies the target rider-type weights,
        replacement-generation settings, and acceptance bounds. Positive weights
        are normalized into integer target counts, allocating leftover slots by
        largest fractional remainder; keys must name members of ``RiderType``.
    :param bool redraw: Allow fresh-agent generation when no donor can be retyped
        acceptably, preserving the donor's start-lane kind; False only permits retyping.
    :returns: A new list of the same size; changed agents are new objects and
        unchanged agents are reused. Empty populations or weights with no positive
        entries return a shallow copy. Types that cannot be filled are skipped,
        so the resulting mix may still differ from the target.
    :rtype: List[BicycleAgent]
    """
    offered: Dict[str, float] = {name: weight for name, weight in config.rider_type_weights.items() if weight > 0}
    population: List[BicycleAgent] = list(population)
    if not offered or not population:
        return population
    total: float = sum(offered.values())
    exact: Dict[str, float] = {name: weight / total * len(population) for name, weight in offered.items()}
    target: Dict[str, int] = {name: int(share) for name, share in exact.items()}
    for name in sorted(exact, key=lambda n: exact[n] - target[n], reverse=True)[:len(population) - sum(target.values())]:
        target[name] += 1
    counts: Dict[str, int] = {name: sum(agent.rider_type.name == name for agent in population) for name in offered}
    unreachable: Set[str] = set()
    while True:
        short: List[str] = [name for name in offered if counts[name] < target[name] and name not in unreachable]
        if not short:
            return population
        wanted: str = max(short, key=lambda name: target[name] - counts[name])
        donors: List[int] = [index for index, agent in enumerate(population)
                  if agent.rider_type.name not in offered or counts[agent.rider_type.name] > target[agent.rider_type.name]]
        rng.shuffle(donors)
        child: BicycleAgent | None = None
        for index in donors:
            child = retyped(population[index], RiderType[wanted])
            if acceptable(space, child, config):
                break
            child = None
        if child is None and redraw and donors:
            index: int = donors[0]
            child = fresh_agent(space, RiderType[wanted], rng, config, starts_on_bike_lane(space, population[index].route))
            if child is not None:
                child = replace(child, lineage_id=population[index].lineage_id)   # Same slot, same lineage.
        if child is None:
            unreachable.add(wanted)
            continue
        old: str = population[index].rider_type.name
        if old in counts:
            counts[old] -= 1
        counts[wanted] += 1
        population[index] = child


def fresh_agent(space: RouteSpace, rider_type: RiderType, rng: Random, config: BicycleOperatorConfig,
                on_bike_lane: bool | None = None, attempts: int = 20) -> BicycleAgent | None:
    """
    A random bicycle of ``rider_type`` drawn within ``config``'s bounds and accepted by ``acceptable``.

    Every draw of a new bicycle (generation, a section's fill and count mutation, rebalancing)
    goes through here, so a new agent always passes the checks the operators apply.

    :param RouteSpace space: Bicycle route space used to generate and validate agents.
    :param RiderType rider_type: Required rider type for the new agent.
    :param Random rng: Random source used by ``BicycleAgent.random``.
    :param BicycleOperatorConfig config: Length, speed, dwell, turn-wait, start and arrival bounds.
    :param bool | None on_bike_lane: Require a bike-lane start (True) or a road start (False);
        None keeps ``BicycleAgent.random``'s default share.
    :param int attempts: Draws before giving up; one draw already retries routes internally.
    :returns: The first acceptable agent with the required start kind, or None if the draws are
        exhausted or generation raises ``RuntimeError`` (it already retried internally).
    :rtype: BicycleAgent | None
    """
    share = {} if on_bike_lane is None else {"bike_lane_start_share": float(on_bike_lane)}
    for _ in range(attempts):
        try:
            agent: BicycleAgent = BicycleAgent.random(
                space, rng, rider_type=rider_type, max_start_t=config.max_start_t, min_length=config.min_length,
                max_length=config.max_length, max_dwell=config.max_dwell, max_turn_dwell=config.max_turn_dwell,
                min_speed=config.min_speed, min_arrival_t=config.min_arrival_t, **share)
        except (InvalidRoute, ValueError):
            continue
        except RuntimeError:
            return None
        if acceptable(space, agent, config) and (on_bike_lane is None
                                                 or starts_on_bike_lane(space, agent.route) == on_bike_lane):
            return agent
    return None
