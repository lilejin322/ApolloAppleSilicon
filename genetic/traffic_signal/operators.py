"""
Random generation, crossover and mutation of ``SignalSection`` chromosomes.

Only demand and offset are free genes. Every timing is recomputed by
``SignalTiming.plan``, so each offspring stays Webster-consistent.
"""

from dataclasses import dataclass, field
from random import Random
from typing import List, Mapping, Tuple

from ...scenario.traffic_signal import SignalPlan, SignalSection, SignalTiming
from ..common import reflect

MUTATION_ACTIONS: Tuple[str, ...] = ("demand", "offset")
"""``mut_signal_section`` actions: step one phase's demand, or shift the junction offset."""


@dataclass(frozen=True)
class SignalOperatorConfig:
    """
    Step sizes and weights for traffic-signal mutation and crossover.

    :param Mapping[str, float] mutation_weights: Chance weights of ``"demand"`` and ``"offset"`` per mutation event.
    :param float demand_step: Standard deviation of a demand step, as a fraction of the phase's demand range.
    :param float offset_step: Standard deviation of an offset step, as a fraction of the cycle.
    :param float swap_rate: Chance per junction that crossover swaps its plans.
    """
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {"demand": 2.0, "offset": 1.0})
    """Chance weights of ``"demand"`` and ``"offset"`` per mutation event."""
    demand_step: float = 0.15
    """Standard deviation of a demand step, as a fraction of the phase's demand range."""
    offset_step: float = 0.2
    """Standard deviation of an offset step, as a fraction of the cycle."""
    swap_rate: float = 0.5
    """Chance per junction that crossover swaps its plans."""


def _phase(plan: SignalPlan) -> float:
    """
    The plan's offset as a fraction of its cycle.

    :param SignalPlan plan: Plan whose offset is read.
    :returns: ``offset / cycle``, in ``[0, 1)``.
    :rtype: float
    """
    return plan.offset / plan.cycle


def random_signal_plan(timing: SignalTiming, rng: Random) -> SignalPlan:
    """
    A plan with each phase's demand uniform in its road class's range and a uniform offset.

    :param SignalTiming timing: Junction timing that turns the genes into a Webster plan.
    :param Random rng: Random source for the demands and the offset.
    :returns: The Webster plan of those genes.
    :rtype: SignalPlan
    """
    demand: Tuple[float, ...] = tuple(round(rng.uniform(*timing.demand_bounds(i)), 1) for i in range(len(timing.phases)))
    return timing.plan(demand, rng.random())


def random_signal_section(timings: Mapping[str, SignalTiming], rng: Random) -> SignalSection:
    """
    A random plan for every control unit, in unit order.

    :param Mapping[str, SignalTiming] timings: One ``SignalTiming`` per control unit, in unit order.
    :param Random rng: Random source passed to ``random_signal_plan``.
    :returns: A section covering those units.
    :rtype: SignalSection
    """
    return SignalSection(tuple(random_signal_plan(t, rng) for t in timings.values()))


def mut_demand(plan: SignalPlan, timing: SignalTiming, rng: Random,
               config: SignalOperatorConfig = SignalOperatorConfig()) -> SignalPlan:
    """
    Step one phase's demand, reflected into its range, and recompute the Webster plan.

    :param SignalPlan plan: Plan whose demand is stepped. The offset fraction is kept.
    :param SignalTiming timing: Junction timing that recomputes the Webster plan.
    :param Random rng: Random source for the phase and the demand step.
    :param SignalOperatorConfig config: Supplies ``demand_step``. Defaults to ``SignalOperatorConfig()``.
    :returns: The recomputed plan.
    :rtype: SignalPlan
    """
    index: int = rng.randrange(len(plan.phases))
    low: float
    high: float
    low, high = timing.demand_bounds(index)
    demand: List[float] = list(plan.demand)
    demand[index] = round(reflect(demand[index] + rng.gauss(0, config.demand_step * (high - low)), low, high), 1)
    return timing.plan(tuple(demand), _phase(plan))


def mut_offset(plan: SignalPlan, timing: SignalTiming, rng: Random,
               config: SignalOperatorConfig = SignalOperatorConfig()) -> SignalPlan:
    """
    Shift the offset by a normal step, wrapped around the cycle.

    :param SignalPlan plan: Plan whose offset is shifted. Its demand is kept.
    :param SignalTiming timing: Junction timing that recomputes the Webster plan.
    :param Random rng: Random source for the offset step.
    :param SignalOperatorConfig config: Supplies ``offset_step``. Defaults to ``SignalOperatorConfig()``.
    :returns: The recomputed plan.
    :rtype: SignalPlan
    """
    return timing.plan(plan.demand, (_phase(plan) + rng.gauss(0, config.offset_step)) % 1.0)


def mut_signal_section(section: SignalSection, timings: Mapping[str, SignalTiming], rng: Random,
                       config: SignalOperatorConfig = SignalOperatorConfig()) -> SignalSection:
    """
    One mutation event: one parameter of one junction, drawn by ``config.mutation_weights``.

    :param SignalSection section: Section to mutate. An empty section is returned unchanged.
    :param Mapping[str, SignalTiming] timings: Junction timings keyed by the plans' junction IDs.
    :param Random rng: Random source for the action, the junction and the step.
    :param SignalOperatorConfig config: Action weights and step sizes. Defaults to ``SignalOperatorConfig()``.
    :returns: The section with one plan replaced.
    :rtype: SignalSection
    :raises IndexError: If every mutation weight is missing or not positive.
    :raises KeyError: If the chosen plan's junction is absent from ``timings``.
    """
    if not section.plans:
        return section
    names: List[str] = [n for n in MUTATION_ACTIONS if config.mutation_weights.get(n, 0) > 0]
    action: str = rng.choices(names, weights=[config.mutation_weights[n] for n in names])[0]
    index: int = rng.randrange(len(section.plans))
    plan: SignalPlan = section.plans[index]
    timing: SignalTiming = timings[plan.junction_id]
    mutant: SignalPlan = (mut_demand if action == "demand" else mut_offset)(plan, timing, rng, config)
    return SignalSection(section.plans[:index] + (mutant,) + section.plans[index + 1:])


def cx_signal_section(first: SignalSection, second: SignalSection, rng: Random,
                      config: SignalOperatorConfig = SignalOperatorConfig()) -> Tuple[SignalSection, SignalSection]:
    """
    Uniform crossover of whole plans: each junction's plans swap with ``config.swap_rate``.

    :param SignalSection first: First parent. Its junction order must match ``second``.
    :param SignalSection second: Second parent.
    :param Random rng: Random source for which junctions swap.
    :param SignalOperatorConfig config: Supplies ``swap_rate``. Defaults to ``SignalOperatorConfig()``.
    :returns: ``(first child, second child)``.
    :rtype: Tuple[SignalSection, SignalSection]
    :raises ValueError: If the parents cover different junctions.
    """
    if [p.junction_id for p in first.plans] != [p.junction_id for p in second.plans]:
        raise ValueError("sections cover different junctions")
    a: List[SignalPlan]
    b: List[SignalPlan]
    a, b = list(first.plans), list(second.plans)
    for i in range(len(a)):
        if rng.random() < config.swap_rate:
            a[i], b[i] = b[i], a[i]
    return SignalSection(tuple(a)), SignalSection(tuple(b))
