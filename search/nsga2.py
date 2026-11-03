"""
NSGA-II over scenario chromosomes, with DEAP.

Generation 0 is ``population`` random scenarios. Each later generation picks
parents by DEAP's dominance-and-crowding tournament, crosses pairs
(``cx_scenario``) and mutates the children (``mut_scenario``), rebalances the
offspring's section mixes, evaluates them, and keeps the best ``population``
of parents and offspring by NSGA-II's order, a duplicate of a better one's
findings last (``select``). Before each selection every individual's rarity
(f3, ``objectives.rarity``) is computed anew from all the search's runs so far,
one evaluated again after a kill once; the Pareto archive keeps the values of
each evaluation, rarity 0, so that it holds the margins and kinds found. Every
evaluation is logged and checkpointed in the search's ``RunStore``; started on
a store that holds a checkpoint, a search goes on from it and evaluates what
the uninterrupted search would have.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Sequence, Tuple

from deap import base, tools

from ..genetic.scenario_chromosome import (ScenarioKinds, ScenarioOperatorConfig, cx_scenario, mut_scenario,
                                           random_scenario, rebalance_scenarios)
from ..scenario.scenario_chromosome import Scenario
from .evaluation import Evaluation
from .objectives import WEIGHTS, rarity
from .store import Checkpoint, Member, RunStore

__all__ = ["Evaluate", "Individual", "NSGA2Config", "ScenarioFitness", "SearchResult", "run_nsga2", "select"]

_logger = logging.getLogger("NSGA2")

Evaluate = Callable[[Scenario, int, int], Evaluation]
"""``evaluate(scenario, generation, individual)``, e.g. an ``ApolloEvaluator``."""


class ScenarioFitness(base.Fitness):
    """An ``Objectives`` as a DEAP fitness: the blame margin minimised, the violation kinds and rarity maximised."""

    weights = WEIGHTS


class Individual:
    """
    One scenario of the search, with its fitness and, once evaluated, its record name and findings.

    Individuals are equal when their scenarios are, as DEAP's ``ParetoFront`` compares them.

    :param Scenario scenario: The chromosome.
    :param values: Its fitness values, or None until evaluated.
    :param str | None name: Its record name, once evaluated.
    :param signature: Its findings' keys (``search.signature``), once evaluated.
    :param transitions: Planning's changes of state in its run, once evaluated.
    """

    def __init__(self, scenario: Scenario, values: Tuple[float, ...] | None = None, name: str | None = None,
                 signature: Tuple[str, ...] = (), transitions: Tuple[str, ...] = ()) -> None:
        self.scenario = scenario
        self.fitness = ScenarioFitness()
        if values is not None:  # A checkpoint before rarity has two values: its rarity is computed anew.
            self.fitness.values = tuple(values) + (0.0,) * (len(WEIGHTS) - len(values))
        self.name = name
        self.signature = tuple(signature)
        self.transitions = tuple(transitions)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Individual) and self.scenario == other.scenario

    def __hash__(self) -> int:
        return hash(self.scenario)

    @classmethod
    def of(cls, member: Member) -> "Individual":
        """
        The individual a checkpoint stored as ``member``.

        :rtype: Individual
        """
        return cls(*member)

    @property
    def member(self) -> Member:
        """The individual as a checkpoint stores it. :rtype: Member"""
        return (self.scenario, self.fitness.values if self.fitness.valid else None, self.name, self.signature,
                self.transitions)


@dataclass(frozen=True)
class NSGA2Config:
    """
    :param int population: Individuals per generation; a positive multiple of 4, as DEAP's tournament needs.
    :param int generations: Generations after the random one.
    :param float | None hours: Search hours after which no evaluation starts; None for no limit.
    :param float cxpb: Chance that a pair of parents is crossed.
    :param float mutpb: Chance that a child is mutated.
    :param int seed: Seed of the search's random source.
    :param int variation_tries: Further mutations of a child still equal to one of its parents.
    :raises ValueError: A setting out of range.
    """

    population: int = 20
    generations: int = 30
    hours: float | None = None
    cxpb: float = 0.8
    mutpb: float = 0.3
    seed: int = 0
    variation_tries: int = 3

    def __post_init__(self) -> None:
        if self.population < 4 or self.population % 4:
            raise ValueError(f"population must be a positive multiple of 4, not {self.population}")
        if self.generations < 0:
            raise ValueError(f"generations must not be negative, not {self.generations}")
        if self.hours is not None and not self.hours > 0:
            raise ValueError(f"hours must be positive, not {self.hours}")
        for name in ("cxpb", "mutpb"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], not {getattr(self, name)}")
        if self.variation_tries < 0:
            raise ValueError(f"variation_tries must not be negative, not {self.variation_tries}")


class SearchResult(NamedTuple):
    """
    How a search ended.

    :param population: The last selected population.
    :param pareto: Every non-dominated individual evaluated, also in a generation left unfinished.
    :param int generation: The last generation evaluated, in full or in part.
    :param bool finished: True after the last generation; False when the hours ran out.
    """

    population: List[Individual]
    pareto: List[Individual]
    generation: int
    finished: bool


def run_nsga2(kinds: ScenarioKinds, evaluate: Evaluate, store: RunStore, config: NSGA2Config = NSGA2Config(), *,
              operators: ScenarioOperatorConfig = ScenarioOperatorConfig(),
              clock: Callable[[], float] = time.monotonic) -> SearchResult:
    """
    Run the search, or go on with the one ``store`` holds a checkpoint of.

    :param ScenarioKinds kinds: The map's section kinds, for generation and variation.
    :param Evaluate evaluate: Evaluates one scenario.
    :param RunStore store: Where the search logs and checkpoints.
    :param NSGA2Config config: Population, generations, hours, rates, seed.
    :param ScenarioOperatorConfig operators: The scenario operators' settings.
    :param clock: Seconds, for ``config.hours``.
    :rtype: SearchResult
    """
    rng = Random(config.seed)
    pareto = tools.ParetoFront()
    # Every evaluation, also those selection dropped, by name: one logged again after a kill replaces its
    # first line, which the checkpoint never took (``RunStore``).
    logged: Dict[str, Dict[str, Any]] = {entry["name"]: entry for entry in store.evaluations()}
    checkpoint = store.load_checkpoint()
    if checkpoint is None:
        generation, population, seconds = 0, [], 0.0
        scenarios = [random_scenario(kinds, rng, operators) for _ in range(config.population)]
        offspring = [Individual(s) for s in rebalance_scenarios(scenarios, kinds, rng)]
    else:
        rng.setstate(checkpoint.rng_state)
        generation, seconds = checkpoint.generation, checkpoint.seconds
        population = [Individual.of(member) for member in checkpoint.population]
        offspring = [Individual.of(member) for member in checkpoint.offspring]
        pareto.update([Individual.of(member) for member in checkpoint.pareto])
        pareto.update([i for i in offspring if i.fitness.valid])  # Checkpoints archived them by generation.
        _logger.info("Resuming generation %d with %d of %d evaluated", generation,
                     sum(i.fitness.valid for i in offspring), len(offspring))
    started = clock() - seconds

    def save() -> None:
        store.save_checkpoint(Checkpoint(generation, tuple(i.member for i in population),
                                         tuple(i.member for i in offspring), rng.getstate(),
                                         tuple(i.member for i in pareto), clock() - started))

    if checkpoint is None:
        save()
    while True:
        for index, individual in enumerate(offspring):
            if individual.fitness.valid:
                continue
            if config.hours is not None and clock() - started >= config.hours * 3600:
                save()
                _logger.info("Out of time after %g h, in generation %d; --resume with more --hours goes on",
                             config.hours, generation)
                return SearchResult(population, list(pareto), generation, False)
            evaluation = evaluate(individual.scenario, generation, index)
            individual.fitness.values = evaluation.objectives
            individual.name = evaluation.name
            individual.signature = evaluation.signature
            individual.transitions = evaluation.transitions
            logged[evaluation.name] = evaluation.to_json()
            store.save_scenario(evaluation.name, individual.scenario)
            store.log(logged[evaluation.name])
            pareto.update([individual])  # Every evaluation, so that a stop in a generation keeps its finds.
            store.save_pareto([{"name": i.name, "margin": i.fitness.values[0], "kinds": round(i.fitness.values[1])}
                               for i in pareto])
            store.save_findings(_findings(logged.values()))
            save()
            _logger.info("%s", evaluation.summary())
        counts = _counts(logged.values())
        for individual in population + offspring:
            margin, violated = individual.fitness.values[:2]
            individual.fitness.values = margin, violated, rarity(individual.signature, individual.transitions, counts)
        population = select(population + offspring, config.population)
        offspring = []
        _logger.info("Generation %d: least margin %.2f m, most kinds %d, most rarity %.2f, Pareto front of %d",
                     generation, min(i.fitness.values[0] for i in population),
                     max(round(i.fitness.values[1]) for i in population),
                     max(i.fitness.values[2] for i in population), len(pareto))
        if generation >= config.generations:
            save()
            return SearchResult(population, list(pareto), generation, True)
        generation += 1
        offspring = [Individual(s) for s in _vary(population, kinds, rng, config, operators)]
        save()


def _vary(population: Sequence[Individual], kinds: ScenarioKinds, rng: Random, config: NSGA2Config,
          operators: ScenarioOperatorConfig) -> List[Scenario]:
    """
    The next offspring: parents by tournament, crossed in pairs and mutated; a
    child equal to one of its parents is mutated up to ``variation_tries``
    more times; then the section mixes are rebalanced.
    """
    parents = _tournament(population, rng)
    children: List[Scenario] = []
    for first, second in zip(parents[::2], parents[1::2]):
        pair = (first.scenario, second.scenario)
        crossed = cx_scenario(*pair, kinds, rng, operators) if rng.random() < config.cxpb else pair
        for child in crossed:
            if rng.random() < config.mutpb:
                child = mut_scenario(child, kinds, rng, operators)
            for _ in range(config.variation_tries):
                if child not in pair:
                    break
                child = mut_scenario(child, kinds, rng, operators)
            children.append(child)
    return rebalance_scenarios(children, kinds, rng)


def select(individuals: Sequence[Individual], k: int) -> List[Individual]:
    """
    NSGA-II's environmental selection, clearing duplicates: in its order (by
    front, then the less crowded first) an individual whose findings a better
    one already made is a duplicate, and duplicates come only after every
    other. Each finding so keeps its best individual, a child that does better
    on it replacing its parent, and the rest of the population goes on to
    other findings instead of copies of one. Individuals that found nothing
    are never duplicates: they spread by crowding as in NSGA-II.

    :param individuals: Parents and offspring, evaluated.
    :param int k: How many to keep.
    :rtype: List[Individual]
    """
    fronts = tools.sortNondominated(list(individuals), len(individuals))
    ordered: List[Individual] = []
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
        ordered += sorted(front, key=lambda i: -i.fitness.crowding_dist)  # Stable: ties keep NSGA-II's order.
    seen: set = set()
    unique: List[Individual] = []
    duplicates: List[Individual] = []
    for individual in ordered:
        keys = set(individual.signature)
        if keys and keys <= seen:
            duplicates.append(individual)
        else:
            unique.append(individual)
            seen |= keys
    return (unique + duplicates)[:k]


def _counts(entries: Iterable[Dict[str, Any]]) -> Counter:
    """For each finding's key and each of Planning's changes, the logged evaluations that had it, for rarity."""
    return Counter(key for entry in entries
                   for key in set(entry.get("signature", ())) | set(entry.get("transitions", ())))


def _findings(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each key the logged evaluations' findings have, by key, the best that
    found it: least margin, then most kinds, then the first.
    """
    findings: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        best = {"name": entry["name"], "margin": entry["margin"], "kinds": entry["kinds"]}
        for key in entry.get("signature", ()):
            known = findings.get(key)
            if known is None or (best["margin"], -best["kinds"]) < (known["margin"], -known["kinds"]):
                findings[key] = best
    return [{"key": key, **best} for key, best in sorted(findings.items())]


def _tournament(individuals: Sequence[Individual], rng: Random) -> List[Individual]:
    """
    DEAP's ``selTournamentDCD`` choosing ``len(individuals)``, drawing from
    ``rng``: two shuffles, each paired off; a pair's winner dominates the
    other, or else is less crowded, or else wins a coin toss.
    """
    def winner(first: Individual, second: Individual) -> Individual:
        a, b = first.fitness, second.fitness
        if a.dominates(b):
            return first
        if b.dominates(a):
            return second
        if a.crowding_dist != b.crowding_dist:
            return first if a.crowding_dist > b.crowding_dist else second
        return first if rng.random() <= 0.5 else second

    chosen: List[Individual] = []
    for _ in range(2):
        shuffled = rng.sample(list(individuals), len(individuals))
        chosen += [winner(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
    return chosen
