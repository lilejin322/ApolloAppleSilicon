"""
Crossover of whole scenarios.

``cx_scenario`` leads each child with one parent. The egos stay with their lead,
or are recombined when ``ego_crossover_rate`` hits. Each traffic section is then
set-crossed and checked against the child built so far.
"""

from random import Random
from typing import List, Tuple

from ...scenario import ADAgent
from ...scenario.scenario_chromosome import Scenario
from ...scenario.sections import Section
from ..ad_agent import cx_ad_agent
from ..section import cx_section
from ..traffic_signal import cx_signal_section
from .chromosome import ScenarioKinds, ScenarioOperatorConfig, _with


def cx_scenario(first: Scenario, second: Scenario, kinds: ScenarioKinds, rng: Random,
                config: ScenarioOperatorConfig = ScenarioOperatorConfig()) -> Tuple[Scenario, Scenario]:
    """
    Two children, led by ``first`` and ``second``. Each keeps its lead's ego, or with
    ``ego_crossover_rate`` the two egos' routes are recombined (``cx_ad_agent``).
    Then section by section (NPCs, bicycles, pedestrians) the parents' sections
    are set-crossed (``cx_section``), every agent checked against its child's ego
    and the child's sections already built; clashing agents are dropped and redrawn.
    The signal plans are crossed per junction first (``cx_signal_section``), so every
    agent is checked replaying under its child's plans; with no plans on either parent
    the children keep empty sections.

    :param Scenario first: Leading parent of the first child.
    :param Scenario second: Leading parent of the second child.
    :param ScenarioKinds kinds: Section kinds, the ego route space and the cross-type fit test.
    :param Random rng: Random source for the ego crossover and each section crossover.
    :param ScenarioOperatorConfig config: Ego crossover rate, ego operators and per-section operators.
        Defaults to ``ScenarioOperatorConfig()``.
    :returns: ``(child led by first, child led by second)``.
    :rtype: Tuple[Scenario, Scenario]
    :raises ValueError: If the egos are recombined and ``cx_ad_agent`` accepts no route, and the leading parent is invalid.
    """
    egos: Tuple[ADAgent, ADAgent] = (first.ego, second.ego)
    if rng.random() < config.ego_crossover_rate:
        egos = cx_ad_agent(first.ego, second.ego, kinds.context.ego_space, rng, config.ad)
    children: List[Scenario] = [Scenario(egos[0]), Scenario(egos[1])]
    if first.signals.plans or second.signals.plans:     # Plans first: complying agents replay under them.
        signals = cx_signal_section(first.signals, second.signals, rng, config.signals)
        children = [_with(children[0], "signals", signals[0]), _with(children[1], "signals", signals[1])]
    for name in Scenario.SECTIONS:
        crossed: Tuple[Section, Section] = cx_section(getattr(first, name), getattr(second, name),
                                                      (kinds.of(name, children[0].signals),
                                                       kinds.of(name, children[1].signals)), rng,
                                                      config.sections[name],
                                                      (kinds.fits(children[0], name), kinds.fits(children[1], name)))
        children = [_with(children[0], name, crossed[0]), _with(children[1], name, crossed[1])]
    return children[0], children[1]
