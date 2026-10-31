"""
Mutation of whole scenarios.

``mut_scenario`` draws one target from ``mutation_weights``: the ego, or one
traffic section. ``mut_ego`` retries the ego's own mutation while its new start
clashes with the traffic already in the scenario.
"""

from random import Random
from typing import List, Tuple

from ...scenario import ADAgent
from ...scenario.scenario_chromosome import Scenario, interaction
from ...scenario.sections import Section
from ..ad_agent import mut_ad_agent
from ..section import mut_section
from ..traffic_signal import mut_signal_section
from .chromosome import ScenarioKinds, ScenarioOperatorConfig, _with

TARGETS: Tuple[str, ...] = ("ego",) + Scenario.SECTIONS + ("signals",)
"""Mutation targets: the ego, then ``Scenario.SECTIONS``, then the signal plans."""


def mut_ego(scenario: Scenario, kinds: ScenarioKinds, rng: Random,
            config: ScenarioOperatorConfig = ScenarioOperatorConfig()) -> Scenario:
    """
    The ego's own mutation, retried while the traffic clashes with the new ego's start; else unchanged.

    Each try calls ``mut_ad_agent``. An unchanged ego is skipped. A changed ego is kept only when
    ``interaction`` finds no clash between it and the agents already in the scenario (``others`` empty,
    so only the ego's start is checked). After ``config.attempts`` failures the original scenario is returned.

    :param Scenario scenario: Scenario whose ego is mutated.
    :param ScenarioKinds kinds: Ego route space, map context and cross-type rules.
    :param Random rng: Random source passed to ``mut_ad_agent``.
    :param ScenarioOperatorConfig config: Retry count and the ego operators. Defaults to ``ScenarioOperatorConfig()``.
    :returns: The scenario with the accepted ego, or the original scenario when every attempt fails.
    :rtype: Scenario
    :raises ValueError: If ``mut_ad_agent`` accepts no route and the original ego is invalid.
    """
    for _ in range(config.attempts):
        ego: ADAgent = mut_ad_agent(scenario.ego, kinds.context.ego_space, rng, config.ad)
        if ego == scenario.ego:
            continue
        candidate: Scenario = _with(scenario, "ego", ego)
        if all(interaction(candidate, kinds.context, kinds.rules, section, agent, others=()) is None
               for _, section in candidate.sections() for agent in section.agents):
            return candidate
    return scenario


def mut_signals(scenario: Scenario, kinds: ScenarioKinds, rng: Random,
                config: ScenarioOperatorConfig = ScenarioOperatorConfig()) -> Scenario:
    """
    Mutate the signal plans (``mut_signal_section``), retried up to ``config.attempts``
    times while the scenario under the new plans is invalid: complying agents re-time
    their stops, which may make bodies overlap.

    :param Scenario scenario: Scenario whose plans are mutated.
    :param ScenarioKinds kinds: Map context, rules and signal timings.
    :param Random rng: Random source.
    :param ScenarioOperatorConfig config: Signal operators and ``attempts``.
    :returns: The scenario with new plans, or the original when no valid mutant is found.
    :rtype: Scenario
    """
    for _ in range(config.attempts):
        signals = mut_signal_section(scenario.signals, kinds.timings, rng, config.signals)
        if signals == scenario.signals:
            continue
        candidate: Scenario = _with(scenario, "signals", signals)
        if candidate.is_valid(kinds.context, kinds.rules):
            return candidate
    return scenario


def mut_scenario(scenario: Scenario, kinds: ScenarioKinds, rng: Random,
                 config: ScenarioOperatorConfig = ScenarioOperatorConfig()) -> Scenario:
    """
    One mutation event, its target drawn once by ``config.mutation_weights``: the ego
    (``mut_ego``) or one section (``mut_section``: add, delete, or mutate one
    agent, checked against the other sections and the ego), or the signal plans
    (``mut_signal_section``; only a candidate when the scenario has plans and
    ``config.evolve_signals``).

    :param Scenario scenario: Scenario to mutate.
    :param ScenarioKinds kinds: Section kinds and the cross-type fit test.
    :param Random rng: Random source for the target and the chosen mutation.
    :param ScenarioOperatorConfig config: Target weights, ego operators and per-section operators.
        Defaults to ``ScenarioOperatorConfig()``.
    :returns: The mutated scenario, or the original when the chosen section does not change.
    :rtype: Scenario
    :raises IndexError: If every mutation weight is missing or not positive.
    :raises ValueError: If the ego is chosen and ``mut_ad_agent`` accepts no route, and the original ego is invalid.
    """
    names: List[str] = [name for name in TARGETS if config.mutation_weights.get(name, 0) > 0
                        and (name != "signals" or (config.evolve_signals and scenario.signals.plans))]
    target: str = rng.choices(names, weights=[config.mutation_weights[name] for name in names])[0]
    if target == "ego":
        return mut_ego(scenario, kinds, rng, config)
    if target == "signals":
        return mut_signals(scenario, kinds, rng, config)
    section: Section = mut_section(getattr(scenario, target), kinds.of(target, scenario.signals), rng,
                                   config.sections[target],
                                   kinds.fits(scenario, target))
    return scenario if section is getattr(scenario, target) else _with(scenario, target, section)
