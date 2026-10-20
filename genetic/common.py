"""
Pieces shared by the crossover and mutation of every agent type.

``OperatorConfig`` holds the bounds and steps of shared lane-route proposals
(``route_operators``); each agent type subclasses it with its operator names
and weights. ``ReplayOperatorConfig`` adds per-gene rates and speed-profile
steps for agents that replay a trajectory, which ``vary`` applies gene by
gene. ``apply_weighted`` draws operators by weight until one yields an
accepted child.
"""

import logging
from dataclasses import dataclass, field
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Callable, ClassVar, Dict, List, Mapping, Sequence, Set, Tuple, TypeVar

from ..utils import require_nonnegative

logger: logging.Logger = logging.getLogger(__name__)

Agent = TypeVar("Agent")


@dataclass(frozen=True)
class OperatorConfig:
    """
    Bounds and step sizes shared by every agent type's crossover and mutation.

    Subclasses name their operators in ``CROSSOVERS`` and ``MUTATIONS`` and
    give default weights; the base declares none, so it only configures
    ``route_operators``.

    :param float min_length: Shortest accepted length of a child route, in metres.
    :param float max_length: Longest accepted length of a child route, in metres.
    :param float max_start_t: Latest accepted start time, in seconds.
    :param int attempts: Tries per operator before the next operator is drawn.
    :param Mapping[str, float] crossover_weights: Relative chance of each crossover.
    :param Mapping[str, float] mutation_weights: Relative chance of each mutation.
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
    """

    CROSSOVERS: ClassVar[Tuple[str, ...]] = ()
    """Crossover operator names, as used in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = ()
    """Mutation operator names, as used in ``mutation_weights``."""

    min_length: float = 50.0
    """Shortest accepted child route length, in metres."""
    max_length: float = 500.0
    """Longest accepted child route length, in metres."""
    max_start_t: float = 10.0
    """Latest accepted start time, in seconds."""
    attempts: int = 5
    """Maximum tries per operator before the next operator is drawn."""
    crossover_weights: Mapping[str, float] = field(default_factory=dict)
    """Relative crossover operator weights; missing or zero weights disable operators."""
    mutation_weights: Mapping[str, float] = field(default_factory=dict)
    """Relative mutation operator weights; missing or zero weights disable operators."""
    sigma_s: float = 10.0
    """Standard deviation of route position steps, in metres."""
    sigma_t: float = 1.0
    """Standard deviation of start time steps, in seconds."""
    bridge_candidates: int = 5
    """Number of cheapest lane paths considered when bridging or rerouting."""
    max_detour: float = 0.3
    """Maximum relative rerouting detour over the shortest candidate stretch; infinity disables the bound."""
    lane_change_odds: float = 0.25
    """Lane-change odds used to accept mutations that add lane changes."""
    bridge_within_parents: bool = False
    """Whether bridge crossover constrains child length to the widened parental range."""

    def __post_init__(self) -> None:
        """
        Validate route bounds, step sizes, operator limits, and selection weights.

        :raises ValueError: If length bounds are invalid, time or step parameters
            are nonfinite or negative, ``lane_change_odds`` is outside [0, 1],
            ``max_detour`` is negative or NaN, ``attempts`` or ``bridge_candidates``
            is less than 1, or operator weights contain unknown names, nonfinite
            values, or negative values.
        """
        if not (isfinite(self.max_length) and 0 <= self.min_length <= self.max_length):
            raise ValueError("min_length must be nonnegative and at most a finite max_length")
        require_nonnegative(max_start_t=self.max_start_t, sigma_s=self.sigma_s, sigma_t=self.sigma_t)
        if not 0 <= self.lane_change_odds <= 1:
            raise ValueError("lane_change_odds must be within [0, 1]")
        if not self.max_detour >= 0:
            raise ValueError("max_detour must be nonnegative")
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.bridge_candidates < 1:
            raise ValueError("bridge_candidates must be at least 1")
        for weights, names in ((self.crossover_weights, self.CROSSOVERS), (self.mutation_weights, self.MUTATIONS)):
            unknown: Set[str] = set(weights) - set(names)
            if unknown:
                raise ValueError(f"unknown operators {sorted(unknown)}; expected some of {names}")
            if not all(isfinite(weight) and weight >= 0 for weight in weights.values()):
                raise ValueError("operator weights must be finite and not negative")


def gauss_step(rng: Random, value: float, sigma: float, high: float) -> float:
    """
    Move ``value`` by a normal step of standard deviation ``sigma``, clamped to ``[0, high]``.

    :param Random rng: Random number generator used to sample the step.
    :param float value: Initial value to move.
    :param float sigma: Standard deviation of the normal step.
    :param float high: Upper bound of the clamping interval; must be nonnegative.
    :returns: The stepped value clamped to ``[0, high]``.
    :rtype: float
    """
    return min(max(value + rng.gauss(0, sigma), 0.0), high)


def apply_weighted(
    operators: Mapping[str, Callable[[], Agent | None]], weights: Mapping[str, float],
    rng: Random, attempts: int, accept: Callable[[Agent], bool],
) -> Agent | None:
    """
    Draw operators by weight, without replacement, until one yields an accepted child.

    Only operators with positive weights and matching names are considered.
    Each drawn operator gets up to ``attempts`` tries before it is removed from
    consideration. A ``None`` result or a rejected child consumes one try.

    :param Mapping[str, Callable[[], Agent | None]] operators: Named operators
        returning a proposed child, or ``None`` if inapplicable or unsuccessful.
    :param Mapping[str, float] weights: Relative selection weights keyed by
        operator name; missing, nonpositive, or unmatched entries are ignored.
    :param Random rng: Random number generator used for weighted operator selection.
    :param int attempts: Maximum number of calls to each selected operator.
    :param Callable[[Agent], bool] accept: Predicate deciding whether a proposed child is accepted.
    :returns: The first accepted child, or ``None`` if no eligible operator succeeds.
    :rtype: Agent | None
    """
    remaining: Dict[str, float] = {name: weight for name, weight in weights.items() if weight > 0 and name in operators}
    while remaining:
        names: List[str] = sorted(remaining)
        name: str = rng.choices(names, weights=[remaining[n] for n in names])[0]
        for _ in range(attempts):
            child: Agent | None = operators[name]()
            if child is not None and accept(child):
                logger.debug("%s produced %s", name, "->".join(child.route.lanes))
                return child
        del remaining[name]
    return None


@dataclass(frozen=True)
class ReplayOperatorConfig(OperatorConfig):
    """
    Gene rates and speed-profile steps for agents that replay a trajectory.

    Subclasses declare genes in ``GENES`` and map operators to them through
    ``CROSSOVER_GENES`` and ``MUTATION_GENES``. ``vary`` first selects each gene
    by its rate, then selects its operators by weight. The parameters below
    include all fields inherited from ``OperatorConfig``.

    :param float min_length: Shortest accepted length of a child route, in metres.
    :param float max_length: Longest accepted length of a child route, in metres.
    :param float max_start_t: Latest accepted start time, in seconds.
    :param int attempts: Tries per operator before the next operator is drawn.
    :param Mapping[str, float] crossover_weights: Relative chance of each crossover.
    :param Mapping[str, float] mutation_weights: Relative chance of each mutation.
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
    :param Mapping[str, float] crossover_rates: Chance, per gene in ``GENES``,
        that a child takes it from the other parent; missing genes never cross.
    :param Mapping[str, float] mutation_rates: Chance, per gene, that it mutates.
    :param float sigma_speed: Standard deviation of target speed steps, in m/s.
    :param float min_speed: Lowest nonzero target speed drawn or stepped to, in m/s.
    :param float max_dwell: Longest stop drawn or stepped to, in seconds.
    :param float min_arrival_t: Earliest route-end time in scenario seconds; when positive,
        the agent must also start before it. Zero disables the horizon check.
    :param float route_swap_rate: Chance that a route crossover whose operators
        all failed takes the other parent's whole route instead.
    :raises ValueError: If inherited bounds, weights, gene names, or probabilities
        are invalid, or speed, dwell, or arrival-time parameters are nonfinite
        or negative.
    """

    GENES: ClassVar[Tuple[str, ...]] = ()
    """Gene names, in the order ``vary`` changes them."""
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {}
    """The gene each crossover changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {}
    """The gene each mutation changes."""

    crossover_rates: Mapping[str, float] = field(default_factory=dict)
    """Independent crossover probability per gene; missing genes never cross."""
    mutation_rates: Mapping[str, float] = field(default_factory=dict)
    """Independent mutation probability per gene; missing genes never mutate."""
    sigma_speed: float = 2.0
    """Standard deviation of target speed steps, in m/s."""
    min_speed: float = 2.0
    """Lowest nonzero target speed drawn or stepped to, in m/s."""
    max_dwell: float = 5.0
    """Longest stop drawn or stepped to, in seconds."""
    route_swap_rate: float = 0.5
    """Probability of attempting a whole-route swap after all route crossover operators fail."""
    min_arrival_t: float = 0.0
    """Earliest accepted route-end time in scenario seconds; positive values also require an earlier start, and zero disables the check."""

    def __post_init__(self) -> None:
        """
        Validate inherited settings, gene rates, speed and timing bounds, and route-swap probability.

        :raises ValueError: If inherited validation fails, rate mappings contain
            unknown genes or probabilities that are nonfinite or outside [0, 1], or speed, dwell,
            or arrival-time parameters are nonfinite or negative, or
            ``route_swap_rate`` is nonfinite or outside [0, 1].
        """
        super().__post_init__()
        for rates in (self.crossover_rates, self.mutation_rates):
            unknown: Set[str] = set(rates) - set(self.GENES)
            if unknown:
                raise ValueError(f"unknown genes {sorted(unknown)}; expected some of {self.GENES}")
            if not all(isfinite(rate) and 0 <= rate <= 1 for rate in rates.values()):
                raise ValueError("gene rates must be within [0, 1]")
        require_nonnegative(sigma_speed=self.sigma_speed, min_speed=self.min_speed, max_dwell=self.max_dwell,
                            min_arrival_t=self.min_arrival_t)
        if not (isfinite(self.route_swap_rate) and 0 <= self.route_swap_rate <= 1):
            raise ValueError("route_swap_rate must be within [0, 1]")


def vary(lead: Agent, order: Sequence[str], operators: Mapping[str, Callable[[Agent], Agent | None]],
         genes: Mapping[str, str], weights: Mapping[str, float], rates: Mapping[str, float],
         rng: Random, attempts: int, accept: Callable[[Agent], bool],
         fallbacks: Mapping[str, Tuple[float, Callable[[Agent], Agent | None]]] = MappingProxyType({}),
         ) -> Agent:
    """
    Vary ``lead`` gene by gene, selecting each independently with its own rate.

    Process genes in ``order`` and apply selected operators to the child so far.
    Operators are drawn by weight without replacement, each receiving up to
    ``attempts`` tries. A proposal must differ from the current child and pass
    ``accept``. If all operators fail, an available fallback is selected with
    its configured probability and receives up to ``attempts`` tries.

    Genes with no successful proposal stay unchanged. If no gene changes,
    return ``lead`` unchanged, provided it passes the final acceptance check.

    :param Agent lead: Agent providing the initial genes.
    :param Sequence[str] order: Gene names in the order they are processed.
    :param Mapping[str, Callable[[Agent], Agent | None]] operators: Named operators
        taking the current child and returning a proposal, or ``None`` on failure.
    :param Mapping[str, str] genes: Mapping from every operator name to the gene it changes.
    :param Mapping[str, float] weights: Relative operator selection weights;
        missing or nonpositive weights disable an operator.
    :param Mapping[str, float] rates: Independent selection probability per gene;
        missing genes are never selected.
    :param Random rng: Random number generator used for gene, operator, and fallback selection.
    :param int attempts: Maximum number of tries per selected operator or fallback.
    :param Callable[[Agent], bool] accept: Predicate applied to changed proposals and the final child.
    :param Mapping[str, Tuple[float, Callable[[Agent], Agent | None]]] fallbacks:
        Optional per-gene fallback probability and operator, used after regular operators fail.
    :returns: The acceptable child, or ``lead`` unchanged if no gene changes.
    :rtype: Agent
    :raises ValueError: If the final child fails the acceptance check.
    :raises KeyError: If an operator name is missing from ``genes`` when operators are grouped.
    """
    child: Agent = lead
    for gene in order:
        if rng.random() >= rates.get(gene, 0.0):
            continue
        current: Agent = child

        def changes(candidate: Agent) -> bool:
            return candidate != current and accept(candidate)

        step: Agent | None = apply_weighted(
            {name: (lambda op=op: op(current)) for name, op in operators.items() if genes[name] == gene},
            weights, rng, attempts, changes)
        if step is None and gene in fallbacks:
            chance: float
            fallback: Callable[[Agent], Agent | None]
            chance, fallback = fallbacks[gene]
            if rng.random() < chance:
                name: str = f"{gene} fallback"
                step = apply_weighted({name: lambda: fallback(current)}, {name: 1.0}, rng, attempts, changes)
        if step is not None:
            child = step
    if not accept(child):
        raise ValueError("no acceptable child: the unchanged parent also violates the operator constraints")
    return child
