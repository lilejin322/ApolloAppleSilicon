"""
Generic operators on ``scenario.sections.Section``: random generation, set
crossover and mutation, plus population-wide type rebalancing.

Type-specific parts (drawing, mutating and crossing one agent, rebalancing a
population of agents) come from a ``SectionKind``; ``npc``, ``bicycle`` and
``pedestrian`` provide them. Every operator returns a valid section; ``fits``
lets a caller add constraints beyond the section's own (the scenario layer
passes its cross-type ones).
"""

from copy import copy
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Callable, Dict, Generic, List, Mapping, Sequence, Tuple, Type, TypeVar

from ...scenario.map_budget import count_for, density_for
from ...scenario.sections import Section, SectionRules

T = TypeVar("T")
S = TypeVar("S", bound=Section)
Fits = Callable[[Any], bool]
"""Predicate of a candidate agent. True when the caller accepts it beyond the section's own conflict check."""

MUTATION_ACTIONS: Tuple[str, ...] = ("count", "agent")
"""``mut_section`` actions: ``"count"`` changes the section size and ``"agent"`` mutates one agent."""


@dataclass(frozen=True)
class SectionOperatorConfig:
    """
    In-type operator settings of one section: size prior, mutation weights and retries.

    :param Mapping[int, float] | None count_weights: Prior of section sizes (size -> weight), used by
        generation and kept by the add/delete Metropolis step. None: uniform over
        the rules' ``min_count .. max_count``.
    :param Mapping[str, float] mutation_weights: ``"count"`` (add or delete one agent, each
        proposed equally often) and ``"agent"`` (bottom-layer mutation of one agent).
    :param float pair_crossover_rate: Chance per crossover to also recombine one
        agent of each child with the bottom-layer crossover.
    :param int attempts: Draws per operator before it gives up and returns its input.
    """
    count_weights: Mapping[int, float] | None = None
    """Prior of section sizes (size -> weight). None: uniform over ``min_count .. max_count``."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {"count": 1.0, "agent": 2.0})
    """Weights of ``"count"`` and ``"agent"``."""
    pair_crossover_rate: float = 0.3
    """Chance per crossover to recombine one agent of each child."""
    attempts: int = 30
    """Draws per operator before it gives up and returns its input."""
    compliance_flip_rate: float = 0.1
    """Chance that an agent mutation flips the ``obeys_signals`` gene instead of ``mutate_agent``."""
    compliance_swap_rate: float = 0.5
    """Chance that an agent crossover swaps the two children's ``obeys_signals`` genes."""

    def prior(self, rules: SectionRules) -> Dict[int, float]:
        """
        Count prior restricted to ``rules``' size range.

        ``count_weights`` of None is uniform over ``min_count .. max_count``. Weights outside that range are dropped.

        :param SectionRules rules: In-type count bounds.
        :returns: Size to weight, one entry for every size in range.
        :rtype: Dict[int, float]
        :raises ValueError: If the weights inside the range sum to zero.
        """
        sizes: range = range(rules.min_count, rules.max_count + 1)
        weights: Mapping[int, float] = self.count_weights or {n: 1.0 for n in sizes}
        prior: Dict[int, float] = {n: float(weights.get(n, 0.0)) for n in sizes}
        if sum(prior.values()) <= 0:
            raise ValueError("count_weights give no weight within min_count..max_count")
        return prior


class SectionKind(Generic[T]):
    """
    Type-specific GA parts of one section type, bound to its map context.

    ``section`` is the ``Section`` subclass. ``context`` is what its agents replay on.
    ``rules`` are the in-type constraints, defaulting to the section's ``RULES``.

    :param Type[Section] section: The ``Section`` subclass.
    :param Any context: What its agents replay on (``Section.replay``).
    :param SectionRules | None rules: In-type constraints; None uses the section's ``RULES``.
    """

    def __init__(self, section: Type[Section], context: Any, rules: SectionRules | None = None) -> None:
        """
        Bind one section class to its map context and rules.

        :param Type[Section] section: The ``Section`` subclass this kind builds.
        :param Any context: What its agents replay on.
        :param SectionRules | None rules: In-type constraints. None uses ``section.RULES``.
        """
        self.section: Type[Section] = section
        self.context: Any = context
        self.signals: Any = None
        """Signal plans the replays obey (``with_signals``); None replays without signals."""
        self.rules: SectionRules = rules or section.RULES

    SECTION: Type[Section] = Section
    """The ``Section`` subclass this kind builds."""

    def build(self, agents: Sequence[T], density: float | None = None) -> Section:
        """
        A section of ``agents`` carrying the relative ``density`` gene; None derives it from the size.

        :rtype: Section
        """
        if density is None:
            density = density_for(len(agents), self.rules.min_count, self.rules.max_count)
        return self.section(tuple(agents), density)

    def target(self, section: Section) -> int:
        """
        Size the ``density`` gene of ``section`` asks for under this kind's (map-budgeted) rules.

        :rtype: int
        """
        if section.density is None:
            return min(max(len(section.agents), self.rules.min_count), self.rules.max_count)
        return count_for(section.density, self.rules.min_count, self.rules.max_count)

    @classmethod
    def default_rules(cls) -> SectionRules:
        """
        The section class's ``RULES``.

        :returns: ``SECTION.RULES``.
        :rtype: SectionRules
        """
        return cls.SECTION.RULES

    def random_agent(self, rng: Random) -> T | None:
        """
        Draw one agent, or None when the draw is impossible.

        :param Random rng: Random source.
        :returns: An agent, or None.
        :rtype: T | None
        :raises NotImplementedError: Subclasses draw the agent.
        """
        raise NotImplementedError

    def mutate_agent(self, agent: T, rng: Random) -> T:
        """
        Bottom-layer mutation of ``agent``.

        :param T agent: Agent to mutate.
        :param Random rng: Random source.
        :returns: The mutant.
        :rtype: T
        :raises NotImplementedError: Subclasses mutate the agent.
        """
        raise NotImplementedError

    def crossover_agents(self, first: T, second: T, rng: Random) -> Tuple[T, T]:
        """
        Bottom-layer crossover of two agents.

        :param T first: Leading parent.
        :param T second: Other parent.
        :param Random rng: Random source.
        :returns: ``(child led by first, child led by second)``.
        :rtype: Tuple[T, T]
        :raises NotImplementedError: Subclasses cross the agents.
        """
        raise NotImplementedError

    def rebalance(self, agents: Sequence[T], rng: Random) -> List[T]:
        """
        Restore the type composition of a population of agents; identity by default.

        :param Sequence[T] agents: Agents to rebalance. The input sequence is not modified.
        :param Random rng: Random source for a subclass that changes agents.
        :returns: A new list. The base class copies ``agents`` unchanged.
        :rtype: List[T]
        """
        return list(agents)

    def with_signals(self, signals: Any) -> "SectionKind[T]":
        """
        A shallow copy of this kind whose replays obey ``signals``.

        :param SignalSection | None signals: The scenario's signal plans.
        :returns: ``self`` when the plans are the same, else a copy.
        :rtype: SectionKind[T]
        """
        if signals is self.signals:
            return self
        bound = copy(self)
        bound.signals = signals
        return bound

    def conflict(self, agent: T, others: Sequence[T]) -> str | None:
        """
        Why ``agent`` conflicts with ``others`` inside this section, or None.

        :param T agent: Candidate agent.
        :param Sequence[T] others: Agents already in the section beside it.
        :returns: The section's conflict reason, or None.
        :rtype: str | None
        """
        return self.section.conflict(agent, others, self.context, self.rules, self.signals)



def flip_compliance(agent: T) -> T:
    """
    ``agent`` with its ``obeys_signals`` gene flipped.

    :param T agent: NPC, bicycle or pedestrian.
    :returns: The copy.
    :rtype: T
    """
    return replace(agent, obeys_signals=not agent.obeys_signals)


def swap_compliance(first: T, second: T) -> Tuple[T, T]:
    """
    The two agents with their ``obeys_signals`` genes exchanged (uniform crossover of the flag).

    :param T first: First child.
    :param T second: Second child.
    :returns: The pair; unchanged objects when the flags agree.
    :rtype: Tuple[T, T]
    """
    if first.obeys_signals == second.obeys_signals:
        return first, second
    return replace(first, obeys_signals=second.obeys_signals), replace(second, obeys_signals=first.obeys_signals)

def _fits(kind: SectionKind, agent: Any, others: Sequence[Any], fits: Fits | None) -> bool:
    """
    Whether ``agent`` fits among ``others``: no in-type conflict, and ``fits`` accepts it when given.

    :param SectionKind kind: Section kind supplying the conflict check.
    :param Any agent: Candidate agent.
    :param Sequence[Any] others: Agents already accepted beside it.
    :param Fits | None fits: Extra predicate, or None.
    :returns: True when both checks pass.
    :rtype: bool
    """
    return kind.conflict(agent, others) is None and (fits is None or fits(agent))


def fill(kind: SectionKind[T], agents: Sequence[T], size: int, rng: Random, config: SectionOperatorConfig,
         fits: Fits | None = None) -> List[T]:
    """
    ``agents`` plus fresh random agents, as generation draws them, that fit, up to ``size`` (fewer if none fit).

    :param SectionKind[T] kind: Section kind that draws agents.
    :param Sequence[T] agents: Agents already accepted. The input sequence is not modified.
    :param int size: Desired number of agents.
    :param Random rng: Random source for new agents.
    :param SectionOperatorConfig config: Supplies ``attempts``. Drawing stops after ``attempts * max(1, size)`` agents that do not fit.
    :param Fits | None fits: Extra predicate, or None. Defaults to None.
    :returns: The accepted agents, shorter than ``size`` when draws run out.
    :rtype: List[T]
    """
    agents: List[T] = list(agents)
    misses: int = 0
    while len(agents) < size and misses < config.attempts * max(1, size):
        agent: T | None = kind.random_agent(rng)
        if agent is not None and _fits(kind, agent, agents, fits):
            agents.append(agent)
        else:
            misses += 1
    return agents


def random_section(kind: SectionKind[T], rng: Random, config: SectionOperatorConfig = SectionOperatorConfig(),
                   fits: Fits | None = None, *, size: int | None = None) -> Section:
    """
    A random section: a size from the count prior, then random agents that fit, one by one.

    :param SectionKind[T] kind: Section kind that draws and checks agents.
    :param Random rng: Random source for the size and the agents.
    :param SectionOperatorConfig config: Size prior and draw limits. Defaults to ``SectionOperatorConfig()``.
    :param Fits | None fits: Extra predicate, or None. Defaults to None.
    :param int | None size: Section size. None draws one from the count prior.
    :returns: A section of the drawn agents.
    :rtype: Section
    :raises ValueError: If fewer than ``kind.rules.min_count`` agents can be placed.
    """
    if size is None:
        prior: Dict[int, float] = config.prior(kind.rules)
        size: int = rng.choices(list(prior), weights=list(prior.values()))[0]
    placed: List[T] = fill(kind, (), size, rng, config, fits)
    if len(placed) < kind.rules.min_count:
        raise ValueError(f"could not place {kind.rules.min_count} {kind.section.__name__} agents")
    return kind.build(placed, density_for(size, kind.rules.min_count, kind.rules.max_count))


def adapt_section(section: Section, kind: SectionKind[T], rng: Random,
                  config: SectionOperatorConfig = SectionOperatorConfig(), fits: Fits | None = None) -> Section:
    """
    Resize ``section`` to the count its ``density`` gene gives under ``kind``'s map budget: keep the
    agents that fit (in order) up to that count, refill with random agents. Used when a section's
    genes move to a map of another size; a section already at its target is returned unchanged.

    :returns: The adapted section (same ``density``).
    :rtype: Section
    """
    size: int = kind.target(section)
    kept: List[T] = []
    for agent in section.agents[:size]:
        try:  # agents of another map may name lanes this one lacks: drop them, keep the density gene
            if kind.section((agent,)).is_valid(kind.context, replace(kind.rules, min_count=0), kind.signals) \
                    and _fits(kind, agent, kept, fits):
                kept.append(agent)
        except ValueError:
            continue
    if kept == list(section.agents):
        return section
    return kind.build(fill(kind, kept, size, rng, config, fits), section.density)


def _refit(kind: SectionKind[T], agents: Sequence[T], size: int, rng: Random, config: SectionOperatorConfig,
           fits: Fits | None) -> List[T]:
    """
    Keep ``agents`` in order while they fit, drop the rest, refill to ``size`` with random agents.

    :param SectionKind[T] kind: Section kind that checks and draws agents.
    :param Sequence[T] agents: Candidates in the order they are considered.
    :param int size: Desired number of agents.
    :param Random rng: Random source for replacements.
    :param SectionOperatorConfig config: Draw limits passed to ``fill``.
    :param Fits | None fits: Extra predicate, or None.
    :returns: The kept agents, refilled up to ``size``.
    :rtype: List[T]
    """
    kept: List[T] = []
    for agent in agents:
        if _fits(kind, agent, kept, fits):
            kept.append(agent)
    return fill(kind, kept, size, rng, config, fits)


def cx_section(first: Section, second: Section, kind: SectionKind[T] | Tuple[SectionKind[T], SectionKind[T]],
               rng: Random,
               config: SectionOperatorConfig = SectionOperatorConfig(),
               fits: Fits | Tuple[Fits | None, Fits | None] | None = None) -> Tuple[Section, Section]:
    """
    Set crossover. The parents' agents are pooled and shuffled; the first child
    takes as many as the first parent had, the second child the rest, so each
    child keeps a parent's size and the size distribution does not drift. Agents
    that conflict within their child (or fail ``fits``) are dropped and replaced
    by random agents. With ``pair_crossover_rate`` one agent of each child is
    then recombined with the bottom-layer crossover, kept only where it fits.
    ``fits`` may be one check for both children or a pair, one per child.

    :param Section first: Leading parent's section. Its size is the first child's size.
    :param Section second: Other parent's section. Its size is the second child's size.
    :param SectionKind[T] | Tuple[SectionKind[T], SectionKind[T]] kind: Section kind that checks agents and
        crosses one pair, or one kind per child (children obeying different signal plans).
    :param Random rng: Random source for the pool order, the pair crossover and its indexes.
    :param SectionOperatorConfig config: Pair-crossover rate and draw limits. Defaults to ``SectionOperatorConfig()``.
    :param Fits | Tuple[Fits | None, Fits | None] | None fits: One check for both children, a pair
        ``(first child, second child)``, or None. Defaults to None.
    :returns: ``(first child, second child)``.
    :rtype: Tuple[Section, Section]
    """
    fit_first: Fits | None
    fit_second: Fits | None
    fit_first, fit_second = fits if isinstance(fits, tuple) else (fits, fits)
    pool: List[T] = list(first.agents) + list(second.agents)
    rng.shuffle(pool)
    n: int = len(first.agents)
    kind_first, kind_second = kind if isinstance(kind, tuple) else (kind, kind)
    kind = kind_first
    children: List[List[T]] = [_refit(kind_first, pool[:n], len(first.agents), rng, config, fit_first),
                               _refit(kind_second, pool[n:], len(second.agents), rng, config, fit_second)]
    if children[0] and children[1] and rng.random() < config.pair_crossover_rate:
        i: int = rng.randrange(len(children[0]))
        j: int = rng.randrange(len(children[1]))
        a: T
        b: T
        a, b = kind.crossover_agents(children[0][i], children[1][j], rng)
        if rng.random() < config.compliance_swap_rate:
            a, b = swap_compliance(a, b)
        for child, index, new, fit, own in ((children[0], i, a, fit_first, kind_first),
                                            (children[1], j, b, fit_second, kind_second)):
            others: List[T] = child[:index] + child[index + 1:]
            if new is not child[index] and _fits(own, new, others, fit):
                child[index] = new
    return kind_first.build(children[0], first.density), kind_second.build(children[1], second.density)


def mut_count(section: Section, kind: SectionKind[T], rng: Random,
              config: SectionOperatorConfig = SectionOperatorConfig(), fits: Fits | None = None, *,
              add: bool | None = None, accept: float | None = None) -> Section:
    """
    Add one random agent or delete one, each proposed with chance 1/2 (``add``),
    accepted with the Metropolis ratio of the count prior (``accept``, drawn once
    per event), so sizes keep the generation distribution. A size beyond the
    bounds, or an add that finds no fitting agent, leaves the section unchanged.

    :param Section section: Section whose size may change.
    :param SectionKind[T] kind: Section kind that draws a replacement agent.
    :param Random rng: Random source for the add-or-delete choice, the acceptance draw and the deleted index.
    :param SectionOperatorConfig config: Count prior and draw limits. Defaults to ``SectionOperatorConfig()``.
    :param Fits | None fits: Extra predicate for an added agent, or None. Defaults to None.
    :param bool | None add: True to add, False to delete. None draws the choice with equal chance.
    :param float | None accept: Metropolis draw in ``[0, 1)``. None draws it.
    :returns: The changed section, or ``section`` when the proposal is rejected.
    :rtype: Section
    """
    add: bool = rng.random() < 0.5 if add is None else add
    accept: float = rng.random() if accept is None else accept
    prior: Dict[int, float] = config.prior(kind.rules)
    n: int = len(section.agents)
    target: int = n + 1 if add else n - 1
    if prior.get(target, 0.0) <= 0 or accept >= prior[target] / prior[n]:
        return section
    if add:
        agents: List[T] = fill(kind, section.agents, target, rng, config, fits)
        return kind.build(agents) if len(agents) == target else section
    drop: int = rng.randrange(n)
    return kind.build(section.agents[:drop] + section.agents[drop + 1:])


def mut_agent(section: Section, kind: SectionKind[T], rng: Random,
              config: SectionOperatorConfig = SectionOperatorConfig(), fits: Fits | None = None) -> Section:
    """
    The bottom-layer mutation on one picked agent; retried while the mutant does not fit.

    :param Section section: Section containing the agent.
    :param SectionKind[T] kind: Section kind that mutates and checks the agent.
    :param Random rng: Random source for the picked index and the mutation.
    :param SectionOperatorConfig config: Supplies ``attempts``. Defaults to ``SectionOperatorConfig()``.
    :param Fits | None fits: Extra predicate, or None. Defaults to None.
    :returns: The section with the mutant, or ``section`` when it is empty or no mutant fits.
    :rtype: Section
    """
    if not section.agents:
        return section
    index: int = rng.randrange(len(section.agents))
    others: Tuple[T, ...] = section.agents[:index] + section.agents[index + 1:]
    for _ in range(config.attempts):
        mutant: T = (flip_compliance(section.agents[index]) if rng.random() < config.compliance_flip_rate
                     else kind.mutate_agent(section.agents[index], rng))
        if mutant is section.agents[index]:
            continue
        if _fits(kind, mutant, others, fits):
            return kind.build(section.agents[:index] + (mutant,) + section.agents[index + 1:], section.density)
    return section


def mut_section(section: Section, kind: SectionKind[T], rng: Random,
                config: SectionOperatorConfig = SectionOperatorConfig(), fits: Fits | None = None) -> Section:
    """
    One mutation event: ``count`` or ``agent``, drawn once by ``config.mutation_weights``.

    :param Section section: Section to mutate.
    :param SectionKind[T] kind: Section kind passed to the chosen operator.
    :param Random rng: Random source for the action and the operator.
    :param SectionOperatorConfig config: Action weights. Defaults to ``SectionOperatorConfig()``.
    :param Fits | None fits: Extra predicate passed to the chosen operator, or None. Defaults to None.
    :returns: The mutated section.
    :rtype: Section
    :raises IndexError: If every mutation weight is missing or not positive.
    """
    names: List[str] = [name for name in MUTATION_ACTIONS if config.mutation_weights.get(name, 0) > 0]
    action: str = rng.choices(names, weights=[config.mutation_weights[name] for name in names])[0]
    if action == "count":
        return mut_count(section, kind, rng, config, fits)
    return mut_agent(section, kind, rng, config, fits)


def rebalance_sections(sections: Sequence[Section], kind: SectionKind[T], rng: Random,
                       fits: Callable[[int, Any], bool] | None = None, *, passes: int = 1,
                       signals: Sequence[Any] | None = None) -> List[Section]:
    """
    Restore the type composition over a population of sections with the bottom
    layer's rebalance helper on all their agents together. A changed agent that
    no longer fits its section (or ``fits(section_index, agent)``) keeps its old self.
    With ``passes`` > 1 the helper runs again on the result while it still changes
    something, so changes rejected by the section constraints are retried on other agents.

    :param Sequence[Section] sections: Sections whose agents are rebalanced together. The input sequence is not modified.
    :param SectionKind[T] kind: Section kind whose ``rebalance`` proposes the new agents.
    :param Random rng: Random source passed to ``kind.rebalance``.
    :param Callable[[int, Any], bool] | None fits: ``fits(section_index, agent)`` for an extra check, or None.
        Defaults to None.
    :param int passes: Times the helper may run while the result still changes. Defaults to 1.
    :param Sequence[SignalSection] | None signals: Each section's signal plans (in-section conflicts replay under them).
    :returns: A new list of the same length. Sections that changed are new objects; the others are reused.
    :rtype: List[Section]
    """
    sections: List[Section] = list(sections)
    for _ in range(passes):
        result: List[Section] = _rebalance_once(sections, kind, rng, fits, signals)
        changed: bool = any(new is not old for new, old in zip(result, sections))
        sections = result
        if not changed:
            break
    return sections


def _rebalance_once(sections: Sequence[Section], kind: SectionKind[T], rng: Random,
                    fits: Callable[[int, Any], bool] | None, signals: Sequence[Any] | None = None) -> List[Section]:
    """
    One pass: rebalance every agent together, then keep a change only where it still fits its section.

    :param Sequence[Section] sections: Sections in population order.
    :param SectionKind[T] kind: Section kind whose ``rebalance`` and ``conflict`` are used.
    :param Random rng: Random source passed to ``kind.rebalance``.
    :param Callable[[int, Any], bool] | None fits: Extra check ``fits(section_index, agent)``, or None.
    :returns: The sections, with a section rebuilt only when one of its agents changed.
    :rtype: List[Section]
    """
    flat: List[Tuple[int, int]] = [(i, j) for i, section in enumerate(sections) for j in range(len(section.agents))]
    if not flat:
        return list(sections)
    before: List[T] = [sections[i].agents[j] for i, j in flat]
    after: List[T] = kind.rebalance(before, rng)
    agents: List[List[T]] = [list(section.agents) for section in sections]
    for (i, j), old, new in zip(flat, before, after):
        if new is old:
            continue
        others: List[T] = agents[i][:j] + agents[i][j + 1:]
        own: SectionKind[T] = kind if signals is None else kind.with_signals(signals[i])
        if own.conflict(new, others) is None and (fits is None or fits(i, new)):
            agents[i][j] = new
    return [kind.build(row, section.density) if row != list(section.agents) else section
            for row, section in zip(agents, sections)]
