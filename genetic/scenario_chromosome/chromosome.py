"""
The scenario chromosome's shared pieces: ``ScenarioOperatorConfig``, one map's
``ScenarioKinds``, random generation of a ``Scenario``, and population-wide
rebalancing of each section's mix.

A scenario is the ego plus NPC, bicycle and pedestrian sections
(``scenario.scenario_chromosome.Scenario``). Crossover and mutation live in
``crossover`` and ``mutation`` and use the config and kinds defined here.
"""

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Dict, List, Mapping, Sequence

from ...scenario import ADAgent
from ...scenario.scenario_chromosome import Scenario, ScenarioContext, ScenarioRules, interaction
from ...scenario.map_budget import MapBudget
from ...scenario.sections import Section
from ...scenario.traffic_signal import SignalTiming
from ..ad_agent import ADOperatorConfig
from ..bicycle import BicycleOperatorConfig
from ..npc_agent import NPCOperatorConfig
from ..pedestrian import PedestrianOperatorConfig
from ..section import BicycleKind, NPCKind, PedestrianKind, SectionKind, SectionOperatorConfig, random_section, \
                       rebalance_sections
from ..traffic_signal import SignalOperatorConfig, random_signal_section


@dataclass(frozen=True)
class ScenarioOperatorConfig:
    """
    Operator bundle for generating, crossing and mutating one scenario.

    :param ScenarioRules rules: Cross-type constraints.
    :param Mapping[str, SectionOperatorConfig] sections: Per section name, its ``SectionOperatorConfig``.
    :param Mapping[str, float] mutation_weights: How often a mutation event goes to the ego
        or to each section (``"ego"``, ``"npcs"``, ``"bicycles"``, ``"pedestrians"``) or to the
        signal plans (``"signals"``; drawn only when the scenario has plans).
    :param SignalOperatorConfig signals: Signal-plan operators (``mut_signal_section``, ``cx_signal_section``).
    :param float ego_crossover_rate: Chance per crossover that the children recombine the egos'
        routes (``cx_ad_agent``); otherwise each keeps its lead's ego.
    :param int attempts: Retries of an ego mutation that clashes with the traffic.
    :param ADOperatorConfig ad: Ego operators passed to generation, ``mut_ad_agent`` and
        ``cx_ad_agent``. ``max_start_t`` bounds a random ego's start.
    """
    rules: ScenarioRules = ScenarioRules()
    """Cross-type constraints."""
    sections: Mapping[str, SectionOperatorConfig] = field(default_factory=lambda: {
        "npcs": SectionOperatorConfig(), "bicycles": SectionOperatorConfig(), "pedestrians": SectionOperatorConfig()})
    """Per section name, its ``SectionOperatorConfig``."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "ego": 1.0, "npcs": 1.0, "bicycles": 1.0, "pedestrians": 1.0, "signals": 1.0})
    """How often a mutation event goes to the ego, each section, or the signal plans (``"signals"``)."""
    signals: SignalOperatorConfig = SignalOperatorConfig()
    """Signal-plan operators (``mut_signal_section``, ``cx_signal_section``)."""
    ego_crossover_rate: float = 0.5
    """Chance per crossover that the children recombine the egos' routes; otherwise each keeps its lead's ego."""
    attempts: int = 20
    """Retries of an ego mutation that clashes with the traffic."""
    ad: ADOperatorConfig = ADOperatorConfig()
    """Ego operators passed to generation, ``mut_ad_agent`` and ``cx_ad_agent``. ``max_start_t`` bounds a random ego's start."""


class ScenarioKinds:
    """
    The section kinds of one map, with each type's operators and the cross-type rules.

    ``context`` is the map the scenario replays on. ``rules`` are the cross-type
    constraints. ``kinds`` maps ``"npcs"``, ``"bicycles"`` and ``"pedestrians"``
    to the ``SectionKind`` built for that section. ``timings`` are the map's
    signal timings (``context.timings``); empty means signals are not evolved.
    """

    def __init__(self, context: ScenarioContext, rules: ScenarioRules = ScenarioRules(), *,
                 npc: NPCOperatorConfig = NPCOperatorConfig(), bicycle: BicycleOperatorConfig = BicycleOperatorConfig(),
                 pedestrian: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> None:
        """
        Build the three section kinds from one map context.

        Each kind receives its operator config and the section rules taken from ``rules``.

        :param ScenarioContext context: Map the scenario replays on.
        :param ScenarioRules rules: Cross-type constraints. Defaults to ``ScenarioRules()``.
        :param NPCOperatorConfig npc: NPC operators. Defaults to ``NPCOperatorConfig()``.
        :param BicycleOperatorConfig bicycle: Bicycle operators. Defaults to ``BicycleOperatorConfig()``.
        :param PedestrianOperatorConfig pedestrian: Pedestrian operators. Defaults to ``PedestrianOperatorConfig()``.
        """
        self.context: ScenarioContext = context
        self.rules: ScenarioRules = rules
        self.timings: Mapping[str, SignalTiming] = context.timings
        self.budget: MapBudget = MapBudget.of(context)
        """This map's count caps: each section's base ``max_count`` times ``budget.scale``."""
        sized = lambda kind: self.budget.rules(rules.section_rules(kind.default_rules()))
        self.kinds: Dict[str, SectionKind[Any]] = {
            "npcs": NPCKind(context.npc_spaces, npc, sized(NPCKind)),
            "bicycles": BicycleKind(context.bicycle_space, bicycle, sized(BicycleKind)),
            "pedestrians": PedestrianKind(context.walk_network, pedestrian, sized(PedestrianKind)),
        }

    def __getitem__(self, name: str) -> SectionKind[Any]:
        """
        The section kind named ``name``.

        :param str name: ``"npcs"``, ``"bicycles"`` or ``"pedestrians"``.
        :returns: That section's ``SectionKind``.
        :rtype: SectionKind[Any]
        :raises KeyError: If ``name`` is not one of those sections.
        """
        return self.kinds[name]

    def of(self, name: str, signals: Any) -> SectionKind[Any]:
        """
        The section kind named ``name`` whose replays obey ``signals``.

        :param str name: ``"npcs"``, ``"bicycles"`` or ``"pedestrians"``.
        :param SignalSection | None signals: The scenario's signal plans.
        :returns: ``self[name].with_signals(signals)``.
        :rtype: SectionKind[Any]
        """
        return self.kinds[name].with_signals(signals)

    def fits(self, scenario: Scenario, name: str) -> Callable[[Any], bool]:
        """
        Whether an agent may join section ``name`` of ``scenario``: cross-type constraints and the ego.

        :param Scenario scenario: Scenario the agent would join.
        :param str name: Section name, one of ``Scenario.SECTIONS``.
        :returns: A predicate of the candidate agent, true when ``interaction`` finds no clash.
        :rtype: Callable[[Any], bool]
        :raises AttributeError: If ``scenario`` has no attribute ``name``.
        """
        section: Section = getattr(scenario, name)
        return lambda agent: interaction(scenario, self.context, self.rules, section, agent) is None


def random_scenario(kinds: ScenarioKinds, rng: Random, config: ScenarioOperatorConfig = ScenarioOperatorConfig(),
                    ) -> Scenario:
    """
    A random ego, then each section drawn at random, its agents fitting the ego and the sections before it.

    The ego is drawn again until ``ego_space.poses`` accepts its route. Sections follow ``Scenario.SECTIONS``.
    When ``kinds.timings`` is non-empty, the signal plans are drawn by ``random_signal_section``.

    :param ScenarioKinds kinds: Map kinds and the cross-type fit test.
    :param Random rng: Random source for the ego and the sections.
    :param ScenarioOperatorConfig config: Section operators and the ego's ``max_start_t``.
        Defaults to ``ScenarioOperatorConfig()``.
    :returns: The generated scenario.
    :rtype: Scenario
    """
    while True:
        ego: ADAgent = ADAgent.random(kinds.context.ego_space, rng, max_start_t=config.ad.max_start_t)
        try:
            kinds.context.ego_space.poses(ego.route)
            break
        except Exception:
            continue
    scenario: Scenario = Scenario(ego)
    if kinds.timings:       # Plans first: complying agents replay under them.
        scenario = _with(scenario, "signals", random_signal_section(kinds.timings, rng))
    for name in Scenario.SECTIONS:
        section: Section = random_section(kinds.of(name, scenario.signals), rng, config.sections[name],
                                          kinds.fits(scenario, name))
        scenario = _with(scenario, name, section)
    return scenario


def _with(scenario: Scenario, name: str, section: Any) -> Scenario:
    """
    A copy of ``scenario`` with attribute ``name`` replaced.

    :param Scenario scenario: Scenario to copy.
    :param str name: ``"ego"``, ``"signals"`` or a name in ``Scenario.SECTIONS``.
    :param Section | ADAgent section: Replacement section, or the replacement ego when ``name`` is ``"ego"``.
    :returns: The copied scenario.
    :rtype: Scenario
    """
    from dataclasses import replace
    return replace(scenario, **{name: section})


REBALANCE_PASSES: Dict[str, int] = {"npcs": 1, "bicycles": 1, "pedestrians": 3}
"""Passes of ``rebalance_sections`` per section: crossing-kind changes clash with traffic more often."""


def rebalance_scenarios(scenarios: Sequence[Scenario], kinds: ScenarioKinds, rng: Random) -> List[Scenario]:
    """
    Restore the population-wide mixes of every section (vehicle types, rider types,
    pedestrian crossing kinds) with ``rebalance_sections``. A changed agent must still fit
    its section, the other sections and the ego, or it keeps its old self. Call it once per
    generation after variation, and once on generation 0. Signal plans are left untouched.

    :param Sequence[Scenario] scenarios: Scenarios to rebalance. The input sequence is not modified.
    :param ScenarioKinds kinds: Section kinds and the cross-type fit test.
    :param Random rng: Random source passed to ``rebalance_sections``.
    :returns: A new list of the same length. Scenarios that changed are new objects; the others are reused.
    :rtype: List[Scenario]
    """
    scenarios: List[Scenario] = list(scenarios)
    for name in Scenario.SECTIONS:
        sections: List[Section] = rebalance_sections([getattr(x, name) for x in scenarios], kinds[name], rng,
                                                     fits=lambda i, agent, name=name: kinds.fits(scenarios[i], name)(agent),
                                                     passes=REBALANCE_PASSES.get(name, 1),
                                                     signals=[x.signals for x in scenarios])
        scenarios = [x if s is getattr(x, name) else _with(x, name, s) for x, s in zip(scenarios, sections)]
    return scenarios
