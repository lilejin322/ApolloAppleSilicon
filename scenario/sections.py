"""
Middle chromosome layer: a section is a variable-length set of agents of one
type (``NPCSection``, ``BicycleSection``, ``PedestrianSection``).

A section is valid when every agent is valid on its own and, within the type:

- the count lies in ``SectionRules.min_count .. max_count``;
- no two bodies overlap at any sampled time. Poses come from
  ``ReplayTrajectory.state_at``, which is deterministic, on a fixed grid of
  ``SectionRules.dt`` up to ``SectionRules.horizon``. An agent stands at its
  start before ``start_t`` and stays at its end after arriving, as the replay
  places it; bodies grow by ``SectionRules.margin`` (half each side);
- start times are spread out within a queue: no two agents queued one behind
  the other (``queued``) start within ``SectionRules.min_start_gap``, as a
  follower does not pull away at the instant its leader does. Agents elsewhere
  may start together.

Type composition is a population property, not a section one (a section of
three cannot hold one e-bike in six); ``genetic.section`` restores it across
a population with the bottom layer's rebalance helpers.

Cross-type constraints and the ego live in ``scenario_chromosome``.
"""

import shapely
import numpy as np
from math import cos, hypot, pi, radians, sin
from dataclasses import dataclass
from weakref import WeakKeyDictionary
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Mapping, NamedTuple, TypeVar, Tuple

from .bicycle import RIDER_SPECS, BicycleAgent
from .npc_agent import VEHICLE_SPECS, NPCVehicleAgent, VehicleType
from .pedestrian import WALKER_SPECS, PedestrianAgent
from .obstacle_ids import BICYCLE_ID_BASE, NPC_ID_BASE, PEDESTRIAN_ID_BASE, assign_ids, dedupe_lineage
from .bike_connectors import parallel_car_lanes
from .routing import InvalidRoute, RouteSpace
from .signal_compliance import active
from .walkways import WalkNetwork

T = TypeVar("T")

Spaces = Mapping[VehicleType, RouteSpace]


class InvalidSection(ValueError):
    """
    A section, or a scenario, breaks a validity constraint.
    """


@dataclass(frozen=True)
class SectionRules:
    """
    In-type constraints of one section.

    :param int min_count: Fewest agents.
    :param int max_count: Most agents.
    :param float horizon: Seconds of scenario time checked for overlaps.
    :param float dt: Sampling step in seconds. At 0.1 s two cars closing at
        30 m/s move 3 m per step, less than a car length, so none passes through another.
    :param float margin: Metres between bodies: each body grows by half of it on every side.
    :param float min_start_gap: Fewest seconds between the start times of two agents in one queue (``queued``).
    :param float spawn_gap: Fewest metres between two ungrown bodies at their
        first pose (t = 0, where every agent stands at its start). Stronger than
        ``margin``, which only stops bodies touching: it keeps agents from all
        spawning in one cluster.
    :param float side_gap: With ``lane_aware``, the spawn gap for two bodies
        that are not in one queue (side by side in adjacent lanes, crossing, or
        opposing); ``spawn_gap`` then applies only within a queue.
    :param bool lane_aware: Whether ``spawn_gap`` applies only to queued pairs (see ``queued``).
    """
    min_count: int = 0
    """Fewest agents."""
    max_count: int = 6
    """Most agents."""
    horizon: float = 60.0
    """Seconds of scenario time checked for overlaps."""
    dt: float = 0.1
    """Sampling step in seconds. At 0.1 s two cars closing at 30 m/s move 3 m
    per step, less than a car length, so none passes through another."""
    margin: float = 0.3
    """Metres between bodies: each body grows by half of it on every side."""
    min_start_gap: float = 0.3
    """Fewest seconds between two start times."""
    spawn_gap: float = 0.0
    """Fewest metres between two ungrown bodies at their first pose (t = 0)."""
    side_gap: float = 0.0
    """With ``lane_aware``, the spawn gap for two bodies not in one queue."""
    lane_aware: bool = False
    """Whether ``spawn_gap`` applies only to queued pairs, ``side_gap`` to the rest."""

    def spawn_gap_for(self, a: "Body", b: "Body") -> float:
        """
        The spawn gap this type needs between ``a`` and ``b``.

        :param Body a: One body.
        :param Body b: The other body.
        :returns: ``spawn_gap``, or ``side_gap`` for a lane-aware type when the two are not queued.
        :rtype: float
        """
        if self.lane_aware and not queued(a, b):
            return self.side_gap
        return self.spawn_gap

    def __post_init__(self) -> None:
        """
        Post-init check.

        :raises ValueError: Unless ``0 <= min_count <= max_count``, ``horizon``
            and ``dt`` are positive, and ``margin`` and ``min_start_gap`` are nonnegative.
        """
        if not 0 <= self.min_count <= self.max_count:
            raise ValueError("need 0 <= min_count <= max_count")
        if self.horizon <= 0 or self.dt <= 0 or self.margin < 0 or self.min_start_gap < 0 or self.spawn_gap < 0 or self.side_gap < 0:
            raise ValueError("horizon and dt must be positive, margin, min_start_gap and spawn_gap nonnegative")

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


class Body(NamedTuple):
    """
    An agent's sampled poses on the rules' time grid, with its size.
    
    :param np.ndarray x:
    :param np.ndarray y: 
    :param np.ndarray heading: 
    :param float length: 
    :param float width: 
    :param float start_t: 
    """
    x: np.ndarray
    """Map X at each sample time, in metres."""
    y: np.ndarray
    """Map Y at each sample time, in metres."""
    heading: np.ndarray
    """Heading at each sample time, in radians."""
    length: float
    """Body length, in metres."""
    width: float
    """Body width, in metres."""
    start_t: float
    """Scenario time the agent starts, in seconds."""
    lane: str | None = None
    """Map lane the agent starts on (its route's first lane), or None without a lane route."""

    @property
    def radius(self) -> float:
        """
        Half the diagonal: radius of the circle around the rectangular body.

        :returns: Circumscribed radius, in metres.
        :rtype: float
        """
        return hypot(self.length, self.width) / 2


def _corners(body: Body, index: np.ndarray, grow: float) -> np.ndarray:
    """
    Four corners of ``body`` at the samples ``index`` selects, each side grown by ``grow``.

    Forward is along ``heading`` and left is positive width, so the ring runs
    front-left, rear-left, rear-right, front-right.

    :param Body body: Sampled poses and size.
    :param np.ndarray index: Indices into ``body``'s sample grid.
    :param float grow: Metres added on every side.
    :returns: Map coordinates in metres, shape ``(len(index), 4, 2)``.
    :rtype: np.ndarray
    """
    half_l, half_w = body.length / 2 + grow, body.width / 2 + grow
    c, s = np.cos(body.heading[index]), np.sin(body.heading[index])
    x, y = body.x[index], body.y[index]
    out = np.empty((len(index), 4, 2))
    for k, (u, v) in enumerate(((half_l, half_w), (-half_l, half_w), (-half_l, -half_w), (half_l, -half_w))):
        out[:, k, 0] = x + u * c - v * s
        out[:, k, 1] = y + u * s + v * c
    return out


def first_overlap(a: Body, b: Body, margin: float, times: np.ndarray) -> float | None:
    """
    The first sampled time at which the two bodies, grown by ``margin`` / 2 each, overlap; None if never.

    :param Body a: First sampled body.
    :param Body b: Second sampled body.
    :param float margin: Metres of clearance; each body grows by half of it on every side.
    :param np.ndarray times: Sample times in seconds, aligned with both bodies' poses.
    :returns: The first sampled time at which the two bodies, grown by ``margin`` / 2 each, overlap; None if never.
    :rtype: float | None
    """
    near = np.hypot(a.x - b.x, a.y - b.y) < a.radius + b.radius + margin
    index = np.flatnonzero(near)
    if not len(index):
        return None
    hits = shapely.intersects(shapely.polygons(_corners(a, index, margin / 2)),
                              shapely.polygons(_corners(b, index, margin / 2)))
    hit = np.flatnonzero(hits)
    return float(times[index[hit[0]]]) if len(hit) else None


_BODY_CACHE: "WeakKeyDictionary[Any, Dict]" = WeakKeyDictionary()
"""Sampled bodies by map context, then ``(agent, horizon, dt)``. Weak keys let the map be collected."""
_BODY_CACHE_SIZE = 4096
"""Entries per context at which that context's cache is cleared."""


def spawn_distance(a: Body, b: Body) -> float:
    """
    Metres between the two ungrown bodies at their first sample (t = 0).

    Every agent stands at its start pose until its ``start_t``, so this is the
    distance between where the two spawn.

    :param Body a: One body.
    :param Body b: The other body.
    :returns: Edge-to-edge distance of the oriented rectangles, 0 if they touch.
    :rtype: float
    """
    first = np.array([0])
    return float(shapely.distance(shapely.polygons(_corners(a, first, 0.0)),
                                  shapely.polygons(_corners(b, first, 0.0)))[0])


QUEUE_LATERAL = 1.6
"""Metres of lateral centre offset below which two aligned bodies share a lane.
Half of a ~3.2-3.7 m urban lane: adjacent-lane centres sit a full lane width apart."""
QUEUE_HEADING = radians(30.0)
"""Largest heading difference, in radians, for two bodies to count as one queue."""


def queued(a: Body, b: Body) -> bool:
    """
    Whether ``a`` and ``b`` start in one lane queue: the same start lane, or
    aligned headings with lateral centre offset under ``QUEUE_LATERAL`` in
    either body's frame.

    The lane ID catches curved lanes, where geometry alone drifts; geometry
    catches a car at the end of one lane and one at the start of its successor,
    or overlapping junction lanes. Adjacent lanes (centres a lane width apart)
    are not queued. Checked on borregas_ave and sunnyvale_loop (3.05-3.5 m lanes).

    :param Body a: One body, at its first sample.
    :param Body b: The other body, at its first sample.
    :returns: True when the two spawn one behind the other in one lane.
    :rtype: bool
    """
    if a.lane is not None and a.lane == b.lane:
        return True
    dh = abs((float(a.heading[0]) - float(b.heading[0]) + pi) % (2 * pi) - pi)
    if dh > QUEUE_HEADING:
        return False
    dx, dy = float(b.x[0] - a.x[0]), float(b.y[0] - a.y[0])
    return min(abs(-dx * sin(a.heading[0]) + dy * cos(a.heading[0])),
               abs(-dx * sin(b.heading[0]) + dy * cos(b.heading[0]))) < QUEUE_LATERAL


def _plans_of(signals: Any, signal_ids: Iterable[str]) -> Tuple[Any, ...]:
    """
    The distinct plans of ``signal_ids`` that ``signals`` holds, sorted by junction ID.

    :param SignalSection signals: Plans in force.
    :param Iterable[str] signal_ids: Signal IDs, possibly repeated or without a plan.
    :returns: The plans.
    :rtype: Tuple[SignalPlan, ...]
    """
    plans = {}
    for signal_id in signal_ids:
        try:
            plan = signals.plan_of(signal_id)
        except KeyError:
            continue
        plans[plan.junction_id] = plan
    return tuple(plans[key] for key in sorted(plans))


def sampled_body(agent: Any, trajectory: Callable[[], Any],
                 length: float, width: float, context: Any,
                 rules: SectionRules, signal_key: Any = None) -> Body:
    """
    ``agent``'s body on ``rules``' grid, cached per context (map), agent, grid and the signal
    plans its replay can meet. Past ``_BODY_CACHE_SIZE`` bodies the least recently used goes.

    :param Any agent: Agent sampled. Its ``start_t`` is stored, and it is part of the cache key.
    :param Callable[[], Any] trajectory: Builds the replay. Called only on a cache miss; ``state_at`` supplies x, y, and heading.
    :param float length: Body length, in metres.
    :param float width: Body width, in metres.
    :param Any context: Map context the poses belong to. A mapping of route spaces is cached on one of them; a context that cannot be weakly referenced is not cached.
    :param SectionRules rules: Time grid. ``horizon`` and ``dt`` are part of the cache key.
    :param Any signal_key: What of the signal plans the replay depends on (``Section.signal_plans``),
        or None for a replay without signals.
    :returns: ``agent``'s body on ``rules``' grid, cached per context (map), agent and grid.
    :rtype: Body
    """
    anchor = context
    if isinstance(context, Mapping) and context:        # NPC spaces: key on one of its (weakly referable) spaces.
        anchor = context[min(context, key=lambda kind: kind.name)]
    try:
        cache = _BODY_CACHE.setdefault(anchor, {})
    except TypeError:                                   # A context that takes no weak reference.
        cache = {}
    key = (agent, rules.horizon, rules.dt, signal_key)
    body = cache.pop(key, None)                         # Re-inserted below: most recently used last.
    if body is None:
        replay = trajectory()
        poses = np.array([tuple(replay.state_at(t))[:3] for t in rules.times])
        lanes = getattr(getattr(agent, "route", None), "lanes", None)
        body = Body(poses[:, 0], poses[:, 1], poses[:, 2], length, width, agent.start_t, lanes[0] if lanes else None)
        if len(cache) >= _BODY_CACHE_SIZE:
            del cache[next(iter(cache))]
    cache[key] = body
    return body


@dataclass(frozen=True)
class Section(Generic[T]):
    """
    A variable-length set of agents of one type.

    Subclasses name the agent type and how to replay one on its map context:
    ``RouteSpace`` mappings for vehicles, a ``RouteSpace`` for bicycles, a
    ``WalkNetwork`` for pedestrians.

    :param Tuple[T, ...] agents: The agents, in no particular order.
    """
    agents: Tuple[T, ...] = ()
    """The agents, in no particular order."""
    density: float | None = None
    """Relative size gene in ``[0, 1]``, map-free: the size is ``count_for(density, min_count, max_count)``
    under the map's budgeted rules (``map_budget``). None (legacy) means ``len(agents)`` is the gene."""
    AGENT: ClassVar[type] = object
    """Agent type stored in ``agents``. Subclasses replace the ``object`` placeholder."""
    RULES: ClassVar[SectionRules] = SectionRules()
    """Default in-type constraints for this section."""
    ID_BASE: ClassVar[int | None] = None
    """First ``obstacle_id`` of the agent type's block; subclasses set it. Post-init dedupes ``lineage_id`` and renumbers densely."""

    def __post_init__(self) -> None:
        """
        Post-init check. ``agents`` is stored as a tuple.

        :raises TypeError: If an agent is not an instance of ``AGENT``.
        """
        object.__setattr__(self, "agents", tuple(self.agents))
        if self.density is not None:
            object.__setattr__(self, "density", min(max(float(self.density), 0.0), 1.0))
        for agent in self.agents:
            if not isinstance(agent, self.AGENT):
                raise TypeError(f"{type(self).__name__} holds {self.AGENT.__name__}, got {type(agent).__name__}")
        if self.ID_BASE is not None:
            agents = assign_ids(dedupe_lineage(self.agents), self.ID_BASE, self.AGENT.__name__)
            object.__setattr__(self, "agents", tuple(agents))

    def __len__(self) -> int:
        """
        Number of agents in this section.

        :returns: Number of agents in this section.
        :rtype: int
        """
        return len(self.agents)

    # Type-specific parts.
    @staticmethod
    def size(agent: T) -> Tuple[float, float]:
        """
        Body length and width of ``agent``, in metres. Subclasses implement this.

        :param T agent: Agent whose body size is requested.
        :returns: Body length and width, in metres.
        :rtype: Tuple[float, float]
        :raises NotImplementedError: If called on ``Section`` itself.
        """
        raise NotImplementedError

    @staticmethod
    def replay(agent: T, context: Any, signals: Any = None):
        """
        Motion ``agent`` replays on its map context. Subclasses implement this.

        :param T agent: Agent to replay.
        :param Any context: Map context for this agent type: ``RouteSpace`` mappings for vehicles, a ``RouteSpace`` for bicycles, a ``WalkNetwork`` for pedestrians.
        :returns: The trajectory describing the agent motion over scenario time.
        :rtype: ReplayTrajectory
        :raises NotImplementedError: If called on ``Section`` itself.
        """
        raise NotImplementedError

    @staticmethod
    def signal_plans(agent: T, context: Any, signals: Any) -> Tuple[Any, ...]:
        """
        Every plan of ``signals`` that ``agent``'s replay could stop for: a superset of the plans
        its compliance stops use (``signal_compliance``), so bodies cached by these plans stay
        exact when other junctions' plans change. Subclasses narrow this; here, every plan.

        :param T agent: Agent replayed.
        :param Any context: Its map context.
        :param SignalSection signals: Plans in force.
        :returns: The plans, sorted by junction ID.
        :rtype: Tuple[SignalPlan, ...]
        """
        return tuple(signals.plans)

    @staticmethod
    def check_agent(agent: T, context: Any) -> None:
        """
        Check ``agent`` on its map context. Subclasses implement this.

        :param T agent: Agent to check.
        :param Any context: Map context for this agent type: ``RouteSpace`` mappings for vehicles, a ``RouteSpace`` for bicycles, a ``WalkNetwork`` for pedestrians.
        :raises InvalidRoute: With the first problem found.
        :raises ValueError: If the context does not fit this agent.
        :raises NotImplementedError: If called on ``Section`` itself.
        """
        raise NotImplementedError

    # Generic parts.
    @classmethod
    def body(cls, agent: T, context: Any, rules: SectionRules | None = None, signals: Any = None) -> Body:
        """
        ``agent``'s sampled body on the section's time grid.

        :param T agent: Agent to sample.
        :param Any context: Map context passed to ``replay``.
        :param SectionRules | None rules: Time grid. Defaults to ``RULES``.
        :param SignalSection | None signals: Signal plans the replay obeys.
        :returns: ``agent``'s body on ``rules``' grid, cached per context, agent, and grid.
        :rtype: Body
        """
        rules = rules or cls.RULES
        length, width = cls.size(agent)
        plans = cls.signal_plans(agent, context, signals) if active(signals, agent) else ()
        return sampled_body(agent, lambda: cls.replay(agent, context, signals if plans else None), length, width,
                            context, rules, plans or None)

    @classmethod
    def conflict(cls, agent: T, others: Iterable[T], context: Any, rules: SectionRules | None = None,
                 signals: Any = None) -> str | None:
        """
        Why ``agent`` cannot join ``others`` in one section (start gap or overlap), or None.

        The first failing check wins: a start within ``min_start_gap`` of another agent in
        its queue (``queued``), then a body overlap, then spawn poses closer than ``spawn_gap``.

        :param T agent: Agent being added.
        :param Iterable[T] others: Agents already in the section.
        :param Any context: Map context passed to ``body``.
        :param SectionRules | None rules: Constraints. Defaults to ``RULES``.
        :returns: Why ``agent`` cannot join ``others`` in one section (start gap or overlap), or None.
        :rtype: str | None
        """
        rules = rules or cls.RULES
        mine = None
        for other in others:
            mine = mine or cls.body(agent, context, rules, signals)
            theirs = cls.body(other, context, rules, signals)
            if abs(agent.start_t - other.start_t) < rules.min_start_gap and queued(mine, theirs):
                return (f"start times {agent.start_t:.2f} and {other.start_t:.2f} closer than "
                        f"{rules.min_start_gap} s in one queue")
            t = first_overlap(mine, theirs, rules.margin, rules.times)
            if t is not None:
                return f"bodies overlap at t={t:.1f} s"
            gap, need = spawn_distance(mine, theirs), rules.spawn_gap_for(mine, theirs)
            if gap < need:
                return f"spawns {gap:.1f} m from another (< {need} m)"
        return None

    def validate(self, context: Any, rules: SectionRules | None = None, signals: Any = None) -> None:
        """
        Check the count, each agent, and that no earlier agent conflicts with it.

        :param Any context: Map context passed to ``check_agent`` and ``conflict``.
        :param SectionRules | None rules: Constraints. Defaults to ``RULES``.
        :raises InvalidSection: If the count, an agent, the start-time spread or an overlap breaks the rules.
        """
        rules = rules or self.RULES
        if not rules.min_count <= len(self.agents) <= rules.max_count:
            raise InvalidSection(f"{len(self.agents)} agents, outside {rules.min_count}..{rules.max_count}")
        for index, agent in enumerate(self.agents):
            try:
                self.check_agent(agent, context)
            except (InvalidRoute, ValueError) as error:
                raise InvalidSection(f"agent {index}: {error}") from error
            why = self.conflict(agent, self.agents[:index], context, rules, signals)
            if why:
                raise InvalidSection(f"agent {index}: {why}")

    def is_valid(self, context: Any, rules: SectionRules | None = None, signals: Any = None) -> bool:
        """
        Whether ``validate`` accepts this section.

        :param Any context: Map context passed to ``validate``.
        :param SectionRules | None rules: Constraints. Defaults to ``RULES``.
        :returns: ``True`` if ``validate`` accepts this section; ``False`` if it raises ``InvalidSection``.
        :rtype: bool
        """
        try:
            self.validate(context, rules, signals)
        except InvalidSection:
            return False
        return True

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A mapping ``{"agents": [...]}`` of each agent's ``to_dict``, in section order.
        :rtype: Dict
        """
        data: Dict = {"agents": [agent.to_dict() for agent in self.agents]}
        if self.density is not None:
            data["density"] = self.density
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Section[T]":
        """
        Build a section from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed section.
        :rtype: Section[T]
        :raises KeyError: If ``agents`` is absent, or an agent dict lacks a required field.
        """
        return cls(tuple(cls.AGENT.from_dict(item) for item in data["agents"]), data.get("density"))


@dataclass(frozen=True)
class NPCSection(Section[NPCVehicleAgent]):
    """
    NPC vehicles, replayed on per-type ``RouteSpace`` mappings (``Spaces``).
    """
    agents: Tuple[NPCVehicleAgent, ...] = ()
    """The NPC vehicles, in no particular order."""
    AGENT: ClassVar[type] = NPCVehicleAgent
    """Agent type stored in ``agents``."""
    ID_BASE: ClassVar[int | None] = NPC_ID_BASE
    """First ``obstacle_id`` of this type's block."""
    # Lane-aware: 8 m edge to edge within one lane queue (about a 1 s headway at
    # urban 30 km/h; no bumper-to-bumper spawns). Side by side in adjacent lanes,
    # or crossing/opposing, only 0.5 m: two 2.5 m trucks on 3.3 m lanes leave ~0.8 m.
    RULES: ClassVar[SectionRules] = SectionRules(max_count=6, margin=0.4, min_start_gap=0.5, spawn_gap=8.0,
                                                 side_gap=0.5, lane_aware=True)
    """At most 6 vehicles, 0.4 m between bodies, and 0.5 s between starts in one queue."""

    @staticmethod
    def size(agent: NPCVehicleAgent) -> Tuple[float, float]:
        """
        Body length and width of ``agent`` from ``VEHICLE_SPECS``.

        :param NPCVehicleAgent agent: Vehicle whose body size is requested.
        :returns: Body length and width, in metres.
        :rtype: Tuple[float, float]
        """
        spec = VEHICLE_SPECS[agent.vehicle_type]
        return spec.length, spec.width

    @staticmethod
    def replay(agent: NPCVehicleAgent, context: Spaces, signals: Any = None):
        """
        Motion ``agent`` replays on the route space for its vehicle type.

        :param NPCVehicleAgent agent: Vehicle to replay.
        :param Spaces context: Route spaces keyed by vehicle type.
        :returns: The trajectory describing the vehicle motion over scenario time.
        :rtype: ReplayTrajectory
        :raises KeyError: If ``context`` has no space for ``agent.vehicle_type``.
        """
        return agent.trajectory(context[agent.vehicle_type], signals)

    @staticmethod
    def signal_plans(agent: NPCVehicleAgent, context: Spaces, signals: Any) -> Tuple[Any, ...]:
        """
        Plans of the signals controlling a lane of the route (``Section.signal_plans``).

        :param NPCVehicleAgent agent: Vehicle replayed.
        :param Spaces context: Route spaces by vehicle type.
        :param SignalSection signals: Plans in force.
        :returns: The plans, sorted by junction ID.
        :rtype: Tuple[SignalPlan, ...]
        """
        relations = context[agent.vehicle_type].service.relations
        return _plans_of(signals, (s for lane in agent.route.lanes for s in relations.signals_for_lane.get(lane, ())))

    @staticmethod
    def check_agent(agent: NPCVehicleAgent, context: Spaces) -> None:
        """
        Check ``agent`` on the route space for its vehicle type.

        :param NPCVehicleAgent agent: Vehicle to check.
        :param Spaces context: Route spaces keyed by vehicle type.
        :raises InvalidRoute: If ``context`` has no space for this type, or with the first problem ``validate`` finds.
        :raises ValueError: If the space allows U-turns too tight for this type.
        """
        if agent.vehicle_type not in context:
            raise InvalidRoute(f"no route space for {agent.vehicle_type.name}")
        agent.validate(context[agent.vehicle_type])


@dataclass(frozen=True)
class BicycleSection(Section[BicycleAgent]):
    """
    Bicycles, replayed on the bicycles' ``RouteSpace`` (``BicycleAgent.space``).
    """
    agents: Tuple[BicycleAgent, ...] = ()
    """The bicycles, in no particular order."""
    AGENT: ClassVar[type] = BicycleAgent
    """Agent type stored in ``agents``."""
    ID_BASE: ClassVar[int | None] = BICYCLE_ID_BASE
    """First ``obstacle_id`` of this type's block."""
    # spawn_gap 3 m: about one bicycle length plus riding clearance.
    RULES: ClassVar[SectionRules] = SectionRules(max_count=4, margin=0.3, min_start_gap=0.3, spawn_gap=3.0)
    """At most 4 bicycles, 0.3 m between bodies, and 0.3 s between starts."""

    @staticmethod
    def size(agent: BicycleAgent) -> Tuple[float, float]:
        """
        Body length and width of ``agent`` from ``RIDER_SPECS``.

        :param BicycleAgent agent: Bicycle whose body size is requested.
        :returns: Body length and width, in metres.
        :rtype: Tuple[float, float]
        """
        spec = RIDER_SPECS[agent.rider_type]
        return spec.length, spec.width

    @staticmethod
    def replay(agent: BicycleAgent, context: RouteSpace, signals: Any = None):
        """
        Motion ``agent`` replays on the bicycles' route space.

        :param BicycleAgent agent: Bicycle to replay.
        :param RouteSpace context: Route space built with ``BicycleAgent.space``.
        :returns: The trajectory describing the bicycle motion over scenario time.
        :rtype: ReplayTrajectory
        """
        return agent.trajectory(context, signals)

    @staticmethod
    def signal_plans(agent: BicycleAgent, context: RouteSpace, signals: Any) -> Tuple[Any, ...]:
        """
        Plans of the signals controlling a map lane of the route, or a car lane a connector of it
        rides beside (``bike_connectors.parallel_car_lanes``; ``Section.signal_plans``).

        :param BicycleAgent agent: Bicycle replayed.
        :param RouteSpace context: The bicycles' route space.
        :param SignalSection signals: Plans in force.
        :returns: The plans, sorted by junction ID.
        :rtype: Tuple[SignalPlan, ...]
        """
        relations = context.service.relations
        lanes = list(agent.route.lanes)
        for lane_id in agent.route.lanes:
            connector = context.virtual_lane(lane_id)
            if connector is not None:
                lanes.extend(car for cars in parallel_car_lanes(context.service, connector) for car in cars)
        return _plans_of(signals, (s for lane in lanes for s in relations.signals_for_lane.get(lane, ())))

    @staticmethod
    def check_agent(agent: BicycleAgent, context: RouteSpace) -> None:
        """
        Check ``agent`` on the bicycles' route space.

        :param BicycleAgent agent: Bicycle to check.
        :param RouteSpace context: Route space built with ``BicycleAgent.space``.
        :raises InvalidRoute: With the first problem ``validate`` finds.
        """
        agent.validate(context)


@dataclass(frozen=True)
class PedestrianSection(Section[PedestrianAgent]):
    """
    Pedestrians, replayed on the map's ``WalkNetwork``.
    """
    agents: Tuple[PedestrianAgent, ...] = ()
    """The pedestrians, in no particular order."""
    AGENT: ClassVar[type] = PedestrianAgent
    """Agent type stored in ``agents``."""
    ID_BASE: ClassVar[int | None] = PEDESTRIAN_ID_BASE
    """First ``obstacle_id`` of this type's block."""
    # spawn_gap 0.5 m: pairs and small groups walk close; still no stacked spawns.
    RULES: ClassVar[SectionRules] = SectionRules(max_count=6, margin=0.1, min_start_gap=0.2, spawn_gap=0.5)
    """At most 6 pedestrians, 0.1 m between bodies, and 0.2 s between starts."""

    @staticmethod
    def size(agent: PedestrianAgent) -> Tuple[float, float]:
        """
        Body length and width of ``agent`` from ``WALKER_SPECS``.

        :param PedestrianAgent agent: Pedestrian whose body size is requested.
        :returns: Body length and width, in metres.
        :rtype: Tuple[float, float]
        """
        spec = WALKER_SPECS[agent.walker_type]
        return spec.length, spec.width

    @staticmethod
    def replay(agent: PedestrianAgent, context: WalkNetwork, signals: Any = None):
        """
        Motion ``agent`` replays on the map's walk network.

        :param PedestrianAgent agent: Pedestrian to replay.
        :param WalkNetwork context: Walk network supplying sidewalk and excursion geometry.
        :returns: The trajectory describing the pedestrian motion over scenario time.
        :rtype: ReplayTrajectory
        """
        return agent.trajectory(context, signals)

    @staticmethod
    def signal_plans(agent: PedestrianAgent, context: WalkNetwork, signals: Any) -> Tuple[Any, ...]:
        """
        Plans of the vehicle lights the walk's signalised crossings follow (``Section.signal_plans``).

        :param PedestrianAgent agent: Walker replayed.
        :param WalkNetwork context: The walk network.
        :param SignalSection signals: Plans in force.
        :returns: The plans, sorted by junction ID.
        :rtype: Tuple[SignalPlan, ...]
        """
        controlled = set(signals.crosswalks)
        try:
            crossings = [context.crossing(e.kind, e.entry, e.exit) for e in agent.route.excursions]
        except InvalidRoute:
            return tuple(signals.plans)                 # the replay raises anyway; key on everything
        return _plans_of(signals, (signals.parallel_signal(c) for crossing in crossings if crossing is not None
                                   for c in crossing.crosswalks if c in controlled))

    @staticmethod
    def check_agent(agent: PedestrianAgent, context: WalkNetwork) -> None:
        """
        Check ``agent`` on the map's walk network.

        :param PedestrianAgent agent: Pedestrian to check.
        :param WalkNetwork context: Walk network supplying sidewalk and excursion geometry.
        :raises InvalidRoute: With the first problem ``validate`` finds.
        """
        agent.validate(context)
