"""
Population-wide crossing-kind mix of pedestrians.
"""

from dataclasses import replace
from random import Random
from typing import Dict, List, Sequence, Set, Tuple

from ...scenario import PedestrianAgent, WalkNetwork
from ...scenario.realism import draw_type
from ...scenario.walkways import CurbPoint, Excursion, ExcursionKind
from .common import PedestrianOperatorConfig, acceptable
from .mutation import _with_excursion

CROSSWALK: ExcursionKind = ExcursionKind.CROSSWALK
"""The on-crosswalk excursion kind. Every other kind is the off-crosswalk side of the split."""
ATTEMPTS: int = 10
"""Redraws tried per surplus excursion."""
REBALANCE_REACH: float = 150.0
"""Metres along the sidewalk within which a surplus excursion is redrawn: crosswalks are
fewer than places to cross off them, so ``redraw_reach`` (10 m) seldom finds one (borregas_ave: 40 m left 31% of 25%).
The new walk must still be ``acceptable`` (approach and link limits)."""


def crosswalk_share(population: Sequence[PedestrianAgent]) -> float | None:
    """
    Share of the population's excursions that use a crosswalk; None without excursions.

    :param Sequence[PedestrianAgent] population: Pedestrians whose excursions are counted.
    :returns: The fraction whose kind is ``CROSSWALK``, or ``None`` when the population has no excursions.
    :rtype: float | None
    """
    kinds: List[ExcursionKind] = [e.kind for agent in population for e in agent.route.excursions]
    return sum(k is CROSSWALK for k in kinds) / len(kinds) if kinds else None


def rebalance_crossing_kinds(population: Sequence[PedestrianAgent], network: WalkNetwork, rng: Random,
                             config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> List[PedestrianAgent]:
    """
    Restore the population's crosswalk / off-crosswalk split of excursions to
    ``config.kind_weights`` (kinds the map offers), as ``rebalance_vehicle_types``
    does for vehicle types. Call it once per generation, after variation.

    Per-agent operators cannot keep the split: in a section, walkers sharing a
    crosswalk overlap more often, so crosswalk crossings are dropped and redrawn
    more often and the off-crosswalk share creeps up (borregas_ave, sections of
    pedestrians: 28.5% -> 38.3% in 30 generations against 25%). Surplus excursions
    of one side are redrawn as the other near their old entry (``network.propose``
    within ``REBALANCE_REACH``, up to ``ATTEMPTS`` tries), off-crosswalk kinds drawn from their weights;
    a redraw that is not ``acceptable`` is skipped. Other excursions and genes are kept.

    :param Sequence[PedestrianAgent] population: Pedestrians to rebalance. The input sequence is not modified.
    :param WalkNetwork network: Walk network that proposes the replacement excursions.
    :param Random rng: Random source for the donor order and the off-crosswalk kinds.
    :param PedestrianOperatorConfig config: Kind weights and acceptance limits. Defaults to ``PedestrianOperatorConfig()``.
    :returns: A new list of the same length. Agents that changed are new objects; the others are reused.
    :rtype: List[PedestrianAgent]
    """
    population: List[PedestrianAgent] = list(population)
    available: Set[str] = {kind.name for kind in network.kinds()}
    weights: Dict[str, float] = {n: w for n, w in config.kind_weights.items() if n in available and w > 0}
    off: Dict[str, float] = {n: w for n, w in weights.items() if n != CROSSWALK.name}
    slots: List[Tuple[int, int]] = [(i, j) for i, agent in enumerate(population)
                                    for j in range(len(agent.route.excursions))]
    if CROSSWALK.name not in weights or not off or not slots:
        return population
    target: int = round(weights[CROSSWALK.name] / sum(weights.values()) * len(slots))
    have: int = sum(population[i].route.excursions[j].kind is CROSSWALK for i, j in slots)
    to_crosswalk: bool = have < target
    need: int = abs(target - have)
    donors: List[Tuple[int, int]] = [(i, j) for i, j in slots
                                     if (population[i].route.excursions[j].kind is CROSSWALK) != to_crosswalk]
    rng.shuffle(donors)
    for i, j in donors:
        if need <= 0:
            break
        agent: PedestrianAgent = population[i]
        excursion: Excursion = agent.route.excursions[j]
        if (excursion.kind is CROSSWALK) == to_crosswalk:      # Changed earlier through the same agent.
            continue
        for _ in range(ATTEMPTS):
            kind: ExcursionKind = CROSSWALK if to_crosswalk else ExcursionKind[draw_type(rng, off)]
            pair: Tuple[CurbPoint, CurbPoint] | None = network.propose(
                kind, rng, excursion.entry, max(config.redraw_reach, REBALANCE_REACH))
            if pair is None:
                continue
            child: PedestrianAgent | None = _with_excursion(
                agent, j, replace(excursion, kind=kind, entry=pair[0], exit=pair[1]), network, rng, config)
            if child is not None and len(child.route.excursions) == len(agent.route.excursions) \
                    and acceptable(network, child, config):
                population[i] = child
                need -= 1
                break
    return population
