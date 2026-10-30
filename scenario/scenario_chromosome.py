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
from typing import Any, Dict, Iterator, Mapping, Tuple

from .ad_agent import ADAgent
from .map_budget import MapBudget
from .routing import InvalidRoute, RouteSpace
from .sections import (
    queued, spawn_distance,
    BicycleSection, Body, InvalidSection, NPCSection, PedestrianSection, Section, SectionRules, Spaces, _corners,
    first_overlap,
)
from .traffic_signal import InvalidSignalPlan, SignalSection, SignalTiming
from .walkways import WalkNetwork


@dataclass(frozen=True)
class ScenarioRules:
    """
    Cross-type constraints and the ego's size.

    :param float ego_length: Ego body length, in metres (Apollo's default Lincoln MKZ).
    :param float ego_width: Ego body width, in metres.
    :param float spawn_clearance: Metres between an agent's first pose and the ego's start body.
    :param float ego_side_clearance: Metres between the ego and a lane-aware vehicle not in its queue.
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
    ego_side_clearance: float = 1.0
    """Metres between the ego and a lane-aware vehicle (NPC) that spawns beside
    it rather than in its queue: side by side in an adjacent lane is realistic."""
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
    :param Mapping[str, SignalTiming] timings: ``signal_timings(service)``, in control-unit order.
        Empty (the default) for a map without signals or when signals are not evolved.
    """
    ego_space: RouteSpace
    """``RouteSpace(service, ADAgent.rules(...))``."""
    npc_spaces: Spaces
    """Per vehicle type, ``NPCVehicleAgent.rules``."""
    bicycle_space: RouteSpace
    """``BicycleAgent.space(service)``."""
    walk_network: WalkNetwork
    """``walk_network(service)``."""
    timings: Mapping[str, SignalTiming] = field(default_factory=dict)
    """``signal_timings(service)``, in control-unit order; empty when signals are not evolved."""
    budget: MapBudget | None = None
    """Per-map agent count caps; None derives them from the map size (``MapBudget.of``)."""

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
    :param SignalSection signals: Traffic-signal plans. Side-mounted: not a body section, not in
        ``SECTIONS``, never checked for overlap.
    """
    ego: ADAgent
    """The ADS instance under test."""
    npcs: NPCSection = field(default_factory=NPCSection)
    """NPC vehicles."""
    bicycles: BicycleSection = field(default_factory=BicycleSection)
    """Bicycles."""
    pedestrians: PedestrianSection = field(default_factory=PedestrianSection)
    """Pedestrians."""
    signals: SignalSection = field(default_factory=SignalSection)
    """Traffic-signal plans; side-mounted, not in ``SECTIONS``."""

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

        :returns: A mapping of ``ego``, each section name and ``signals`` to its ``to_dict``.
        :rtype: Dict
        """
        return {"ego": self.ego.to_dict(), **{name: section.to_dict() for name, section in self.sections()},
                "signals": self.signals.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict) -> "Scenario":
        """
        Build a scenario from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed scenario.
        :rtype: Scenario
        A missing ``signals`` key (older scenarios) gives an empty ``SignalSection``.

        :raises KeyError: If ``ego`` or a section name is absent, or a nested dict lacks a required field.
        """
        signals = SignalSection.from_dict(data["signals"]) if "signals" in data else SignalSection()
        return cls(ADAgent.from_dict(data["ego"]), NPCSection.from_dict(data["npcs"]),
                   BicycleSection.from_dict(data["bicycles"]), PedestrianSection.from_dict(data["pedestrians"]),
                   signals)

    def validate(self, context: ScenarioContext, rules: ScenarioRules = ScenarioRules()) -> None:
        """
        Check the ego route, each section, each agent against the ego and later section types,
        then the signal plans against ``context.timings`` (skipped when both are empty).

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
                section.validate(context.of(section), MapBudget.of(context).rules(rules.section_rules(section.RULES)),
                                 self.signals)
            except InvalidSection as error:
                raise InvalidSection(f"{name}: {error}") from error
        for position, (name, section) in enumerate(self.sections()):
            for index, agent in enumerate(section.agents):
                why = interaction(self, context, rules, section, agent, others=self.SECTIONS[position + 1:])
                if why:
                    raise InvalidSection(f"{name} {index}: {why}")
        if self.signals.plans or context.timings:
            try:
                self.signals.validate(context.timings)
            except (InvalidSignalPlan, KeyError) as error:
                raise InvalidSection(f"signals: {error}") from error

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


def spawn_gap_between(a: SectionRules | None, b: SectionRules | None,
                      rules: "ScenarioRules" = None, queued_pair: bool = True) -> float:
    """
    Fewest metres between two spawn bodies (ungrown, at t = 0); ``None`` stands for the ego.

    Effective defaults (NPC 8, bicycle 3, pedestrian 1, ego clearance 3):

    ==========  ===========  =======  ==========
    pair        NPC          bicycle  pedestrian
    ==========  ===========  =======  ==========
    ego         8 / 1 side   3        3
    NPC         8 / 0.5 side 3        0.5
    bicycle                  3        0.5
    pedestrian                        0.5
    ==========  ===========  =======  ==========

    Vehicle pairs (ego or NPC with a lane-aware NPC) are lane-aware: the first
    number holds within one lane queue (``sections.queued``), the "side" number
    otherwise (adjacent lane, crossing, opposing). Same type: that type's
    ``spawn_gap_for``. Cross type: the smaller ``spawn_gap``, so a walker may wait
    at the kerb beside a car. Ego: ``max(spawn_clearance, spawn_gap)`` in a queue,
    ``ego_side_clearance`` beside it.

    :param SectionRules | None a: One side's rules, or None for the ego.
    :param SectionRules | None b: The other side's rules, or None for the ego.
    :param ScenarioRules rules: Supplies ``spawn_clearance``. Defaults to ``ScenarioRules()``.
    :param bool queued_pair: Whether the two spawn in one lane queue; only lane-aware pairs use it.
    :returns: The minimum edge-to-edge spawn distance, in metres.
    :rtype: float
    """
    rules = rules or ScenarioRules()
    if a is None or b is None:
        other = b if a is None else a
        if other is not None and other.lane_aware and not queued_pair:
            return rules.ego_side_clearance
        return max(rules.spawn_clearance, other.spawn_gap if other is not None else 0.0)
    if a is b or a == b:
        return a.side_gap if a.lane_aware and not queued_pair else a.spawn_gap
    return min(a.spawn_gap, b.spawn_gap)


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
                rules.ego_width, ego.start_t, ego.route.lanes[0])


def ego_conflict(ego: ADAgent, context: ScenarioContext, rules: ScenarioRules, body: Body, margin: float,
                 spawn: SectionRules | float = 0.0) -> str | None:
    """
    Why an agent with ``body`` clashes with the ego's start, or None.

    The rectangles at the first sample, ungrown, must be at least
    ``max(spawn_clearance, spawn_gap)`` apart, so an NPC keeps its car-to-car
    spawn gap from the ego in its lane queue; beside the ego (lane-aware types)
    ``ego_side_clearance`` applies instead. Until ``ego.start_t + ego_clear_time``, the bodies must not overlap when
    each is grown by ``margin`` / 2.

    :param ADAgent ego: Ego standing at its start.
    :param ScenarioContext context: Supplies ``ego_space`` for the start pose.
    :param ScenarioRules rules: Spawn clearance, clear time, and the ego's size.
    :param Body body: The other agent's sampled body.
    :param float margin: Metres of clearance for the overlap test; each body grows by half of it.
    :param SectionRules | float spawn: The agent type's rules (lane-aware for NPCs), or a plain spawn gap.
    :returns: Why an agent with ``body`` clashes with the ego's start, or None.
    :rtype: str | None
    :raises InvalidRoute: If the ego route is illegal or cannot be sent unambiguously.
    """
    ego_body = ego_start_body(ego, context, rules)
    own = spawn if isinstance(spawn, SectionRules) else SectionRules(spawn_gap=spawn)
    gap = spawn_distance(ego_body, body)
    need = spawn_gap_between(None, own, rules, queued(ego_body, body))
    if gap < need:
        return f"starts {gap:.1f} m from the ego's start (< {need} m)"
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

    Cross-type overlaps use the larger of the two sections' margins; cross-type
    spawn poses must be the smaller of the two ``spawn_gap`` apart.

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
    body = type(section).body(agent, context.of(section), own, scenario.signals)
    why = ego_conflict(scenario.ego, context, rules, body, own.margin, own)
    if why:
        return why
    for name, other in scenario.sections():
        if type(other) is type(section) or (others is not None and name not in others):
            continue
        theirs = rules.section_rules(type(other).RULES)
        margin = max(own.margin, theirs.margin)
        # Cross-type spawn gap: the smaller type's, so a pedestrian may spawn on the
        # sidewalk beside a parked-start car without needing the car-to-car 8 m.
        spawn_gap = spawn_gap_between(own, theirs, rules)
        for index, them in enumerate(other.agents):
            their_body = type(other).body(them, context.of(other), theirs, scenario.signals)
            t = first_overlap(body, their_body, margin, rules.times)
            if t is not None:
                return f"overlaps {name} {index} at t={t:.1f} s"
            gap = spawn_distance(body, their_body)
            if gap < spawn_gap:
                return f"spawns {gap:.1f} m from {name} {index} (< {spawn_gap} m)"
    return None
