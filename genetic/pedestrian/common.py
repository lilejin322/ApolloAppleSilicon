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
from typing import ClassVar, Mapping, Tuple

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

    The parameters below include all fields inherited from ``ReplayOperatorConfig``
    and ``OperatorConfig``. Length bounds apply to the replay path, as in
    ``PedestrianAgent.random``; route position steps apply along the sidewalk.

    :param float min_length: Shortest accepted replay path length, in metres (default: 10.0).
    :param float max_length: Longest accepted replay path length, in metres (default: 200.0).
    :param float max_start_t: Latest accepted start time, in seconds (default: 10.0).
    :param int attempts: Tries per operator before the next operator is drawn (default: 5).
    :param Mapping[str, float] crossover_weights: Relative crossover operator weights;
        defaults to 1.0 each for ``shared_sidewalk``, ``walker_type``,
        ``speed_profile``, ``walk_style``, and ``start_time``.
    :param Mapping[str, float] mutation_weights: Relative mutation operator weights;
        defaults to 3.0 for ``shift``, 2.0 each for ``redraw`` and ``speed``, and
        1.0 each for ``kind``, ``count``, ``turn_back``, ``walker_type``,
        ``speed_point``, ``stop``, ``walk_style``, and ``start_t``.
    :param float sigma_s: Standard deviation of walk endpoint and speed-point
        position steps, in metres (default: 2.0).
    :param float sigma_t: Standard deviation of start time steps, in seconds (default: 1.0).
    :param int bridge_candidates: Number of candidate lane paths considered by shared
        lane-route operators (default: 5); unused by pedestrian operators.
    :param float max_detour: Maximum relative detour allowed by shared lane-route
        rerouting (default: 0.3); unused by pedestrian operators.
    :param float lane_change_odds: Lane-change acceptance odds for shared lane-route
        mutation (default: 0.25); unused by pedestrian operators.
    :param bool bridge_within_parents: Whether shared lane-route bridge crossover
        constrains child length to the widened parental range (default: False);
        unused by pedestrian operators.
    :param Mapping[str, float] crossover_rates: Independent crossover probability per
        gene; defaults to 0.5 each for ``route``, ``walker_type``, ``speed_profile``,
        and ``start_t``. Missing genes never cross.
    :param Mapping[str, float] mutation_rates: Independent mutation probability per
        gene; defaults to 0.3 each for ``route``, ``walker_type``, ``speed_profile``,
        and ``start_t``. Missing genes never mutate.
    :param float sigma_speed: Standard deviation of target speed steps, in m/s (default: 0.2).
    :param float min_speed: Lowest nonzero target speed drawn or stepped to, in m/s (default: 0.5).
    :param float max_dwell: Longest stop drawn or stepped to, in seconds (default: 20.0).
    :param float min_arrival_t: Earliest route-end time in scenario seconds; when
        positive, the agent must also start before it (default: 0.0, disabling the check).
    :param float route_swap_rate: Probability of taking the other parent's whole
        route when all route crossover operators fail (default: 0.5).
    :param Mapping[str, float] walker_type_weights: Walker composition that type
        mutation resamples from (default: ``realism.WALKER_MIX``).
    :param Mapping[str, float] kind_weights: Excursion kinds that kind mutation and
        added excursions draw from (default: ``realism.EXCURSION_MIX``).
    :param Mapping[int, float] count_weights: Target distribution of excursion counts
        maintained by count mutation (default: ``realism.EXCURSION_COUNT_MIX``).
    :param float sigma_curb: Standard deviation of excursion entry or exit steps
        along the sidewalk, in metres (default: 0.5).
    :param float sigma_turn_back: Standard deviation of turning-point steps,
        in metres (default: 1.0).
    :param float turn_back_probability: Probability of drawing a turning point
        instead of none during turn-back mutation, matching generation to preserve
        the share of walkers who turn back (default: 0.05).
    :param float redraw_reach: Maximum distance along the sidewalk from the old
        entry to a redrawn excursion's entry, in metres (default: 10.0).
    :param float max_link: Maximum distance along the sidewalk within which an
        added excursion enters, in metres (default: 40.0).
    :param float max_approach: Maximum distance along the sidewalk from an entry
        or arrival within which a moved start or end is drawn, in metres (default: 30.0).
    :raises ValueError: If inherited bounds, operator weights, gene names, rates,
        or step sizes fail validation; pedestrian step sizes or distance bounds
        are nonfinite or negative; or ``turn_back_probability`` is outside [0, 1].
    """

    GENES: ClassVar[Tuple[str, ...]] = GENES
    """Pedestrian gene names, in the order each variation call processes them."""
    CROSSOVERS: ClassVar[Tuple[str, ...]] = ("shared_sidewalk", "walker_type", "speed_profile", "walk_style",
                                             "start_time")
    """Supported crossover names used as keys in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = (
        "shift", "redraw", "kind", "count", "turn_back", "walker_type", "speed", "speed_point", "stop", "walk_style",
        "start_t")
    """Supported mutation names used as keys in ``mutation_weights``."""
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {
        "shared_sidewalk": "route", "walker_type": "walker_type", "speed_profile": "speed_profile",
        "walk_style": "speed_profile", "start_time": "start_t",
    }
    """Map each crossover operator to the pedestrian gene it changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {
        "shift": "route", "redraw": "route", "kind": "route", "count": "route", "turn_back": "route",
        "walker_type": "walker_type", "speed": "speed_profile", "speed_point": "speed_profile",
        "stop": "speed_profile", "walk_style": "speed_profile", "start_t": "start_t",
    }
    """Map each mutation operator to the pedestrian gene it changes."""

    min_length: float = 10.0
    """Shortest accepted replay path length, in metres."""
    max_length: float = 200.0
    """Longest accepted replay path length, in metres."""
    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_sidewalk": 1.0, "walker_type": 1.0, "speed_profile": 1.0, "walk_style": 1.0, "start_time": 1.0,
    })
    """Relative selection weights of crossover operators within each selected gene."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shift": 3.0, "redraw": 2.0, "kind": 1.0, "count": 1.0, "turn_back": 1.0, "walker_type": 1.0, "speed": 2.0,
        "speed_point": 1.0, "stop": 1.0, "walk_style": 1.0, "start_t": 1.0,
    })
    """Relative selection weights of mutation operators; zero disables an operator."""
    crossover_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.5))
    """Independent crossover probability per gene; missing genes are disabled."""
    mutation_rates: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(GENES, 0.3))
    """Independent mutation probability per gene; missing genes are disabled."""
    walker_type_weights: Mapping[str, float] = field(default_factory=lambda: dict(WALKER_MIX))
    """Walker-type composition used for weighted type sampling."""
    kind_weights: Mapping[str, float] = field(default_factory=lambda: dict(EXCURSION_MIX))
    """Excursion-kind weights used for kind mutation and added excursions."""
    count_weights: Mapping[int, float] = field(default_factory=lambda: dict(EXCURSION_COUNT_MIX))
    """Target excursion-count distribution maintained by count mutation."""
    sigma_s: float = 2.0
    """Standard deviation of walk endpoint and speed-point position steps, in metres."""
    sigma_speed: float = 0.2
    """Standard deviation of target speed steps, in m/s."""
    min_speed: float = 0.5
    """Lowest nonzero target speed drawn or stepped to, in m/s."""
    max_dwell: float = 20.0
    """Longest stop drawn or stepped to, in seconds."""
    sigma_curb: float = 0.5
    """Standard deviation of excursion entry or exit steps along the sidewalk, in metres."""
    sigma_turn_back: float = 1.0
    """Standard deviation of turning-point steps, in metres."""
    turn_back_probability: float = 0.05
    """Probability that turn-back redraw produces a turning point rather than none."""
    redraw_reach: float = 10.0
    """Maximum distance along the sidewalk from the old entry to a redrawn excursion entry, in metres."""
    max_link: float = 40.0
    """Maximum distance along the sidewalk from the walk end to an added excursion entry, in metres."""
    max_approach: float = 30.0
    """Maximum distance along the sidewalk from an entry or arrival to a sampled endpoint, in metres."""

    def __post_init__(self) -> None:
        """
        Validate inherited configuration, pedestrian step sizes, distance bounds, and turn-back probability.

        :raises ValueError: If inherited validation fails, pedestrian step sizes or distance
            bounds are nonfinite or negative, or ``turn_back_probability`` is outside [0, 1].
        """
        super().__post_init__()
        require_nonnegative(sigma_curb=self.sigma_curb, sigma_turn_back=self.sigma_turn_back,
                            redraw_reach=self.redraw_reach, max_link=self.max_link, max_approach=self.max_approach)
        if not 0 <= self.turn_back_probability <= 1:
            raise ValueError("turn_back_probability must be within [0, 1]")


def acceptable(network: WalkNetwork, agent: PedestrianAgent, config: PedestrianOperatorConfig) -> bool:
    """
    Whether the agent is valid on the network, with its replay path and start time in bounds.

    :param WalkNetwork network: Walk network used to construct the agent's trajectory.
    :param PedestrianAgent agent: Agent to validate.
    :param PedestrianOperatorConfig config: Configuration defining path length, start time, and arrival time bounds.
    :returns: ``True`` if the trajectory is valid and all configured bounds are satisfied.
    :rtype: bool
    """
    if agent.start_t > config.max_start_t:
        return False
    try:
        trajectory = agent.trajectory(network)
    except InvalidRoute:
        return False
    return (config.min_length <= trajectory.path.length <= config.max_length
            and (not config.min_arrival_t or agent.start_t < config.min_arrival_t <= trajectory.arrival_t))


def on_route(agent: PedestrianAgent, route: WalkRoute | None, network: WalkNetwork) -> PedestrianAgent | None:
    """
    Put ``agent`` on the proposed walk, adjusting its speed-point positions.

    :param PedestrianAgent agent: Agent to place on the proposed walk.
    :param WalkRoute | None route: Proposed walk, or ``None`` if no walk is available.
    :param WalkNetwork network: Walk network used to apply the proposed walk.
    :returns: The updated agent, or ``None`` if the walk is absent or cannot be applied.
    :rtype: PedestrianAgent | None
    """
    if route is None:
        return None
    try:
        return agent.with_route(route, network)
    except (InvalidRoute, ValueError):
        return None


def retyped(agent: PedestrianAgent, walker_type: WalkerType) -> PedestrianAgent:
    """
    Change the walker type, mapping positive target speeds to the same quantiles
    of the new type's speed prior through ``realism.mapped_speed``.

    Mapped speeds are capped at the new type's maximum; stop points stay unchanged.
    Clamping alone previously produced runners at walking pace and elderly walkers
    at 2 m/s (borregas, 30 generations: 87 of 97 runners and 50 of 123 elderly
    walkers outside their ranges).

    :param PedestrianAgent agent: Agent whose walker type and target speeds are updated.
    :param WalkerType walker_type: Walker type to assign.
    :returns: The updated agent, or ``agent`` unchanged if its type already matches.
    :rtype: PedestrianAgent
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
    Redraw endpoints that exceed ``config.max_approach`` after an excursion changes.

    The end is redrawn around the last excursion's arrival; when ``start`` is
    true, the start is also redrawn around the first excursion's entry.
    Sampling follows generation, within ``config.max_approach`` metres along
    the sidewalk. Endpoints already within range stay unchanged.

    :param WalkRoute route: Walk whose endpoints may need adjustment.
    :param WalkNetwork network: Walk network used to measure gaps and sample endpoints.
    :param Random rng: Random number generator used to redraw endpoints.
    :param PedestrianOperatorConfig config: Configuration defining the maximum approach distance.
    :param bool start: Whether to check and adjust the start as well as the end.
    :returns: The adjusted walk, or ``route`` unchanged if it has no excursions.
    :rtype: WalkRoute
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
    """
    Sample a point within ``reach`` metres along the same sidewalk, wrapping its position.

    :param WalkNetwork network: Walk network used to wrap the sampled position.
    :param CurbPoint point: Centre of the sampling interval.
    :param Random rng: Random number generator used to sample a uniform offset.
    :param float reach: Maximum absolute offset along the sidewalk, in metres.
    :returns: The sampled point with its position wrapped along the sidewalk.
    :rtype: CurbPoint
    """
    return network.wrapped(point.sidewalk, point.s + rng.uniform(-reach, reach))
