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
from typing import Callable, ClassVar, Mapping, Sequence, TypeVar

from ..utils import require_nonnegative

logger = logging.getLogger(__name__)

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
        length lies in the parents' range widened by ``BRIDGE_LENGTH_SLACK``. Where
        parents rarely meet, the only bridges are long detours, and bridged children
        grew routes by a step (shalun bicycles on curb lanes: +27 m per child, median
        150 -> 190 m in five generations). Off by default: vehicle routes stay stable
        without it.
    """

    CROSSOVERS: ClassVar[tuple[str, ...]] = ()
    """Crossover operator names, as used in ``crossover_weights``."""
    MUTATIONS: ClassVar[tuple[str, ...]] = ()
    """Mutation operator names, as used in ``mutation_weights``."""

    min_length: float = 50.0
    max_length: float = 500.0
    max_start_t: float = 10.0
    attempts: int = 5
    crossover_weights: Mapping[str, float] = field(default_factory=dict)
    mutation_weights: Mapping[str, float] = field(default_factory=dict)
    sigma_s: float = 10.0
    sigma_t: float = 1.0
    bridge_candidates: int = 5
    max_detour: float = 0.3
    lane_change_odds: float = 0.25
    bridge_within_parents: bool = False

    def __post_init__(self) -> None:
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
            unknown = set(weights) - set(names)
            if unknown:
                raise ValueError(f"unknown operators {sorted(unknown)}; expected some of {names}")
            if not all(isfinite(weight) and weight >= 0 for weight in weights.values()):
                raise ValueError("operator weights must be finite and not negative")


def gauss_step(rng: Random, value: float, sigma: float, high: float) -> float:
    """Move ``value`` by a normal step of standard deviation ``sigma``, clamped to ``[0, high]``."""
    return min(max(value + rng.gauss(0, sigma), 0.0), high)


def apply_weighted(
    operators: Mapping[str, Callable[[], Agent | None]], weights: Mapping[str, float],
    rng: Random, attempts: int, accept: Callable[[Agent], bool],
) -> Agent | None:
    """
    Draw operators by weight, without replacement, until one yields an accepted child.

    Each drawn operator gets ``attempts`` tries; an operator returning None is
    not applicable or failed. Returns None when every operator failed.
    """
    remaining = {name: weight for name, weight in weights.items() if weight > 0 and name in operators}
    while remaining:
        names = sorted(remaining)
        name = rng.choices(names, weights=[remaining[n] for n in names])[0]
        for _ in range(attempts):
            child = operators[name]()
            if child is not None and accept(child):
                logger.debug("%s produced %s", name, "->".join(child.route.lanes))
                return child
        del remaining[name]
    return None


@dataclass(frozen=True)
class ReplayOperatorConfig(OperatorConfig):
    """
    Gene rates and speed-profile steps for agents that replay a trajectory.

    Subclasses name their genes in ``GENES``, in the order ``vary`` changes
    them, and map each operator to the gene it changes in
    ``CROSSOVER_GENES`` and ``MUTATION_GENES``. The inherited fields are as
    in ``OperatorConfig``.

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
    """

    GENES: ClassVar[tuple[str, ...]] = ()
    """Gene names, in the order ``vary`` changes them."""
    CROSSOVER_GENES: ClassVar[Mapping[str, str]] = {}
    """The gene each crossover changes."""
    MUTATION_GENES: ClassVar[Mapping[str, str]] = {}
    """The gene each mutation changes."""

    crossover_rates: Mapping[str, float] = field(default_factory=dict)
    mutation_rates: Mapping[str, float] = field(default_factory=dict)
    sigma_speed: float = 2.0
    min_speed: float = 2.0
    max_dwell: float = 5.0
    route_swap_rate: float = 0.5
    min_arrival_t: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for rates in (self.crossover_rates, self.mutation_rates):
            unknown = set(rates) - set(self.GENES)
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
         fallbacks: Mapping[str, tuple[float, Callable[[Agent], Agent | None]]] = MappingProxyType({}),
         ) -> Agent:
    """
    Vary ``lead`` gene by gene, each gene only with its own rate.

    In ``order``, each gene is drawn with its rate; a drawn gene gets one of
    its operators by weight, applied to the child so far, and the result
    must change and stay acceptable. If they all fail and the gene has a
    fallback ``(chance, operator)``, that operator is tried with its chance.
    A gene whose operators and fallback all fail stays as it is. When no gene
    changes, ``lead`` is returned unchanged.
    """
    child = lead
    for gene in order:
        if rng.random() >= rates.get(gene, 0.0):
            continue
        current = child

        def changes(candidate: Agent) -> bool:
            return candidate != current and accept(candidate)

        step = apply_weighted(
            {name: (lambda op=op: op(current)) for name, op in operators.items() if genes[name] == gene},
            weights, rng, attempts, changes)
        if step is None and gene in fallbacks:
            chance, fallback = fallbacks[gene]
            if rng.random() < chance:
                name = f"{gene} fallback"
                step = apply_weighted({name: lambda: fallback(current)}, {name: 1.0}, rng, attempts, changes)
        if step is not None:
            child = step
    if not accept(child):
        raise ValueError("no acceptable child: the unchanged parent also violates the operator constraints")
    return child
