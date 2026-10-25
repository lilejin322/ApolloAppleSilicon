"""
Top chromosome layer: a ``Scenario`` is the ego (``ADAgent``) plus one
section per agent type.

Cross-type constraints (``ScenarioRules``):

- no NPC, bicycle or pedestrian bodies overlap one another at any sampled
  time (``sections.first_overlap`` on the sections' common time grid), with
  the larger of the two types' margins. Replayed agents do not react to one
  another, so a car driving through a crossing walker, or a bicycle on a curb
  lane under a car, would be what the simulator shows;
- no agent starts on the ego's start: its first pose keeps
  ``ScenarioRules.spawn_clearance`` from the ego's body at its start pose;
- no agent passes through the ego while it still stands at its start (until
  ``ADAgent.start_t + ScenarioRules.ego_clear_time``). Afterwards Apollo
  drives the ego, so its pose is unknown and nothing further is checked.
"""

import shapely
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

from .ad_agent import ADAgent
from .routing import InvalidRoute, RouteSpace
from .sections import (
    BicycleSection, Body, InvalidSection, NPCSection, PedestrianSection, Section, SectionRules, Spaces, _corners,
    first_overlap,
)
from .walkways import WalkNetwork


@dataclass(frozen=True)
class ScenarioRules:
    """
    Cross-type constraints and the ego's size.

    :param float ego_length: Ego body length, in metres (Apollo's default Lincoln MKZ).
    :param float ego_width: Ego body width, in metres.
    :param float spawn_clearance: Metres between an agent's first pose and the ego's start body.
    :param float ego_clear_time: Seconds after the ego's ``start_t`` during which
        it is taken to stand at its start.
    :param float horizon: Seconds checked; the sections' grids use the same.
    :param float dt: Sampling step in seconds.
    """
    ego_length: float = 4.933
    """Ego body length, in metres (Apollo's default Lincoln MKZ)."""
    ego_width: float = 2.11
    """Ego body width, in metres."""
    spawn_clearance: float = 3.0
    """Metres between an agent's first pose and the ego's start body."""
    ego_clear_time: float = 2.0
    """Seconds after the ego's ``start_t`` during which it is taken to stand at its start."""
    horizon: float = 60.0
    """Seconds checked; the sections' grids use the same."""
    dt: float = 0.1
    """Sampling step in seconds."""

    def section_rules(self, rules: SectionRules) -> SectionRules:
        """
        ``rules`` on this scenario's time grid.

        :param SectionRules rules: Section constraints whose ``horizon`` and ``dt`` are replaced.
        :returns: ``rules`` on this scenario's time grid.
        :rtype: SectionRules
        """
        from dataclasses import replace
        return replace(rules, horizon=self.horizon, dt=self.dt)

    @property
    def times(self) -> np.ndarray:
        """
        Sample times from 0 through ``horizon``, in steps of ``dt``.

        The ``1e-9`` past ``horizon`` keeps the last sample when ``horizon``
        is a multiple of ``dt``.

        :returns: An array of sample times from 0 through ``horizon``, in steps of ``dt``.
        :rtype: np.ndarray
        """
        return np.arange(0.0, self.horizon + 1e-9, self.dt)


@dataclass(frozen=True)
class ScenarioContext:
    """
    Everything a scenario replays on, all built from one map.

    :param RouteSpace ego_space: ``RouteSpace(service, ADAgent.rules(...))``.
    :param Spaces npc_spaces: Per vehicle type, ``NPCVehicleAgent.rules``.
    :param RouteSpace bicycle_space: ``BicycleAgent.space(service)``.
    :param WalkNetwork walk_network: ``walk_network(service)``.
    """
    ego_space: RouteSpace
    """``RouteSpace(service, ADAgent.rules(...))``."""
    npc_spaces: Spaces
    """Per vehicle type, ``NPCVehicleAgent.rules``."""
    bicycle_space: RouteSpace
    """``BicycleAgent.space(service)``."""
    walk_network: WalkNetwork
    """``walk_network(service)``."""

    def of(self, section: Section) -> Any:
        """
        Map context ``section`` replays on.

        :param Section section: Section whose context is requested.
        :returns: ``npc_spaces`` for an NPC section, ``bicycle_space`` for a bicycle section, otherwise ``walk_network``.
        :rtype: Any
        """
        if isinstance(section, NPCSection):
            return self.npc_spaces
        if isinstance(section, BicycleSection):
            return self.bicycle_space
        return self.walk_network


@dataclass(frozen=True)
class Scenario:
    """
    One scenario: the ego and its traffic.

    :param ADAgent ego: The ADS instance under test.
    :param NPCSection npcs: NPC vehicles.
    :param BicycleSection bicycles: Bicycles.
    :param PedestrianSection pedestrians: Pedestrians.
    """
    ego: ADAgent
    """The ADS instance under test."""
    npcs: NPCSection = field(default_factory=NPCSection)
    """NPC vehicles."""
    bicycles: BicycleSection = field(default_factory=BicycleSection)
    """Bicycles."""
    pedestrians: PedestrianSection = field(default_factory=PedestrianSection)
    """Pedestrians."""

    SECTIONS = ("npcs", "bicycles", "pedestrians")
    """Section attribute names, in the order ``validate`` checks them."""

    def sections(self) -> Iterator[Tuple[str, Section]]:
        """
        The traffic sections in ``SECTIONS`` order.

        :returns: ``(name, section)`` pairs for ``npcs``, ``bicycles``, and ``pedestrians``.
        :rtype: Iterator[Tuple[str, Section]]
        """
        for name in self.SECTIONS:
            yield name, getattr(self, name)

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A mapping of ``ego`` and each section name to its ``to_dict``.
        :rtype: Dict
        """
        return {"ego": self.ego.to_dict(), **{name: section.to_dict() for name, section in self.sections()}}

    @classmethod
    def from_dict(cls, data: Dict) -> "Scenario":
        """
        Build a scenario from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed scenario.
        :rtype: Scenario
        :raises KeyError: If ``ego`` or a section name is absent, or a nested dict lacks a required field.
        """
        return cls(ADAgent.from_dict(data["ego"]), NPCSection.from_dict(data["npcs"]),
                   BicycleSection.from_dict(data["bicycles"]), PedestrianSection.from_dict(data["pedestrians"]))

    def validate(self, context: ScenarioContext, rules: ScenarioRules = ScenarioRules()) -> None:
        """
        Check the ego route, each section, then each agent against the ego and later section types.

        :param ScenarioContext context: Map context the scenario replays on.
        :param ScenarioRules rules: Cross-type constraints and the ego's size. Defaults to ``ScenarioRules()``.
        :raises InvalidSection: If the ego, a section or a cross-type constraint is broken.
        """
        try:
            context.ego_space.poses(self.ego.route)
        except InvalidRoute as error:
            raise InvalidSection(f"ego: {error}") from error
        for name, section in self.sections():
            try:
                section.validate(context.of(section), rules.section_rules(section.RULES))
            except InvalidSection as error:
                raise InvalidSection(f"{name}: {error}") from error
        for position, (name, section) in enumerate(self.sections()):
            for index, agent in enumerate(section.agents):
                why = interaction(self, context, rules, section, agent, others=self.SECTIONS[position + 1:])
                if why:
                    raise InvalidSection(f"{name} {index}: {why}")

    def is_valid(self, context: ScenarioContext, rules: ScenarioRules = ScenarioRules()) -> bool:
        """
        Whether ``validate`` accepts this scenario.

        :param ScenarioContext context: Map context passed to ``validate``.
        :param ScenarioRules rules: Cross-type constraints and the ego's size. Defaults to ``ScenarioRules()``.
        :returns: ``True`` if ``validate`` accepts this scenario; ``False`` if it raises ``InvalidSection``.
        :rtype: bool
        """
        try:
            self.validate(context, rules)
        except InvalidSection:
            return False
        return True


def ego_start_body(ego: ADAgent, context: ScenarioContext, rules: ScenarioRules) -> Body:
    """
    The ego standing at its start pose, on ``rules``' grid.

    :param ADAgent ego: Ego whose start pose and ``start_t`` are sampled.
    :param ScenarioContext context: Supplies ``ego_space`` for the start pose.
    :param ScenarioRules rules: Time grid and the ego's length and width.
    :returns: The ego standing at its start pose, on ``rules``' grid.
    :rtype: Body
    :raises InvalidRoute: If the ego route is illegal or cannot be sent unambiguously.
    """
    start = context.ego_space.poses(ego.route).start
    n = len(rules.times)
    return Body(np.full(n, start.x), np.full(n, start.y), np.full(n, start.heading), rules.ego_length,
                rules.ego_width, ego.start_t)


def ego_conflict(ego: ADAgent, context: ScenarioContext, rules: ScenarioRules, body: Body, margin: float) -> str | None:
    """
    Why an agent with ``body`` clashes with the ego's start, or None.

    The rectangles at the first sample, ungrown, must be at least ``spawn_clearance``
    apart. Until ``ego.start_t + ego_clear_time``, the bodies must not overlap when
    each is grown by ``margin`` / 2.

    :param ADAgent ego: Ego standing at its start.
    :param ScenarioContext context: Supplies ``ego_space`` for the start pose.
    :param ScenarioRules rules: Spawn clearance, clear time, and the ego's size.
    :param Body body: The other agent's sampled body.
    :param float margin: Metres of clearance for the overlap test; each body grows by half of it.
    :returns: Why an agent with ``body`` clashes with the ego's start, or None.
    :rtype: str | None
    :raises InvalidRoute: If the ego route is illegal or cannot be sent unambiguously.
    """
    ego_body = ego_start_body(ego, context, rules)
    first = np.array([0])
    gap = shapely.distance(shapely.polygons(_corners(ego_body, first, 0.0)), shapely.polygons(_corners(body, first, 0.0)))
    if float(gap[0]) < rules.spawn_clearance:
        return f"starts {float(gap[0]):.1f} m from the ego's start (< {rules.spawn_clearance} m)"
    standing = rules.times <= ego.start_t + rules.ego_clear_time
    t = first_overlap(ego_body, body, margin, rules.times)
    if t is not None and standing[int(round(t / rules.dt))]:
        return f"passes through the waiting ego at t={t:.1f} s"
    return None


def interaction(scenario: Scenario, context: ScenarioContext, rules: ScenarioRules, section: Section, agent: Any,
                *, others: Tuple[str, ...] | None = None) -> str | None:
    """
    Why ``agent`` (of ``section``'s type) clashes with the ego or with the agents
    of the other-type sections of ``scenario`` (only those named in ``others``
    if given), or None. Its own type is ``Section.conflict``'s job.

    Cross-type overlaps use the larger of the two sections' margins.

    :param Scenario scenario: Scenario whose ego and other sections are checked.
    :param ScenarioContext context: Map context the sections replay on.
    :param ScenarioRules rules: Cross-type constraints and the shared time grid.
    :param Section section: Section ``agent`` belongs to.
    :param Any agent: Agent being checked.
    :param Tuple[str, ...] | None others: Section names to compare against. ``None`` compares every other type.
    :returns: Why ``agent`` clashes with the ego or another type, or None.
    :rtype: str | None
    """
    own = rules.section_rules(type(section).RULES)
    body = type(section).body(agent, context.of(section), own)
    why = ego_conflict(scenario.ego, context, rules, body, own.margin)
    if why:
        return why
    for name, other in scenario.sections():
        if type(other) is type(section) or (others is not None and name not in others):
            continue
        theirs = rules.section_rules(type(other).RULES)
        margin = max(own.margin, theirs.margin)
        for index, them in enumerate(other.agents):
            t = first_overlap(body, type(other).body(them, context.of(other), theirs), margin, rules.times)
            if t is not None:
                return f"overlaps {name} {index} at t={t:.1f} s"
    return None
