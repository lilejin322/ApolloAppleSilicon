"""
Traffic-signal chromosome: one ``SignalPlan`` per signalised junction, held by a ``SignalSection``.

A plan does not colour signals one by one; it runs the junction's phases in
turn, and each phase is one
synchronous (EQ) group of ``MapRelations.signal_graph``. So signals of one group
always show the same colour, and two groups are never green or yellow together
(their NE relation holds by construction).

Each phase runs green -> yellow -> all-red, then the next phase starts.
Phase timings are not free genes. They come from the junction's geometry and
from a hidden demand gene:

- yellow (ITE): ``t_r + v / (2 a)``, with v the fastest approach speed limit;
- all-red (ITE): ``(W + L) / v``, with W the junction extent along the approach;
- cycle (Webster): ``C0 = (1.5 L + 5) / (1 - Y)``. L is the lost time per cycle,
  the sum over phases of ``startup + yellow + all_red - extension``. Y is the sum
  of the critical flow ratios ``q_i / s``, with ``q_i`` the demand gene
  (veh/h/lane) of phase i's critical lane;
- greens: Webster's split of ``C0 - L`` in proportion to ``y_i``. Each green is
  then clamped to the region's minimum and maximum green and to the pedestrian
  WALK + flashing DON'T WALK of the crosswalks the phase serves.

The GA mutates demand and offset only and recomputes every timing, so each
offspring is Webster-consistent by construction (``SignalTiming.plan``).

Colours are pure functions of the absolute scenario time ``t`` held by the
scenario runner. Pedestrian heads (``ped_state``) are pure functions of ``t``
too.
"""

import networkx as nx
from enum import IntEnum
from shapely import union_all
from dataclasses import dataclass, field
from math import atan2, ceil, cos, hypot, pi, sin
from typing import Any, Dict, Mapping, Sequence, Tuple
from weakref import WeakKeyDictionary

from ..map.geometry import polygon_to_shape

EPS = 1e-6
"""
Epsilon
"""


class SignalColor(IntEnum):
    """
    Values of ``apollo.perception.TrafficLight.Color``.
    """
    UNKNOWN = 0
    """Unknown Signal"""
    RED = 1
    """Red Signal"""
    YELLOW = 2
    """Yellow Signal"""
    GREEN = 3
    """Green Signal"""


class PedState(IntEnum):
    """
    Pedestrian signal head state.
    """
    DONT_WALK = 0
    """"""
    WALK = 1
    """"""
    FLASHING_DONT_WALK = 2
    """"""


@dataclass(frozen=True)
class CycleTier:
    """
    Cycle-length class of a junction (FHWA Signal Timing Manual, 2nd ed., ch. 6).

    A junction falls into the first tier, from the largest, whose every threshold it meets.

    :param str name: Tier name.
    :param tuple bounds: Smallest and largest cycle, in seconds.
    :param int min_phases: Fewest phases.
    :param float min_speed: Lowest fastest-approach speed limit, in m/s.
    :param int min_lanes: Fewest controlled lanes (all phases).
    """

    name: str
    bounds: Tuple[float, float]
    min_phases: int = 0
    min_speed: float = 0.0
    min_lanes: int = 0

    def admits(self, phases: int, speed: float, lanes: int) -> bool:
        return phases >= self.min_phases and speed >= self.min_speed - EPS and lanes >= self.min_lanes


LOCAL = CycleTier("local", (40.0, 90.0))
COLLECTOR = CycleTier("collector", (45.0, 120.0), min_lanes=9)
ARTERIAL_SPEED = CycleTier("arterial", (60.0, 150.0), min_speed=15.6)        # >= 35 mph
ARTERIAL_PHASES = CycleTier("arterial", (60.0, 150.0), min_phases=3)
MAJOR = CycleTier("major", (80.0, 180.0), min_phases=4, min_speed=15.6)
CYCLE_TIERS = (LOCAL, COLLECTOR, ARTERIAL_SPEED, ARTERIAL_PHASES, MAJOR)
"""Default tiers. Speeds and phase counts set the road class; the lane count separates
a small local crossing from a collector at low speed limits."""


@dataclass(frozen=True)
class RegionProfile:
    """
    Signal-timing practice of one region. Defaults follow US practice (MUTCD 2009 and ITE).

    :param str name: Profile name.
    :param float reaction_time: Perception-reaction time in the ITE yellow formula, in seconds.
    :param float deceleration: Comfortable deceleration in the ITE yellow formula, in m/s².
    :param tuple yellow_bounds: Smallest and largest yellow, in seconds (MUTCD 4D.26: 3 to 6 s).
    :param tuple all_red_bounds: Smallest and largest red clearance, in seconds.
    :param float vehicle_length: L in the all-red formula, in metres.
    :param float default_speed: Approach speed in m/s when the map has no limit (0 or missing).
    :param float startup_lost: Start-up lost time per phase, in seconds (HCM: 2 s).
    :param float extension: Part of yellow + all-red still used by traffic, in seconds (HCM: 2 s).
    :param float saturation_flow: Saturation flow s, in veh/h/lane.
    :param tuple major_demand: Demand range of a major phase's critical lane, in veh/h/lane.
    :param tuple minor_demand: Demand range of a minor phase's critical lane, in veh/h/lane.
    :param float max_flow_ratio: Cap on Y. A larger Y is oversaturated; it is scaled
        down to this cap before Webster, which gives the longest cycle.
    :param tuple cycle_bounds: Cycle bounds of a junction built without a tier (collector), in seconds.
    :param tuple cycle_tiers: ``CycleTier`` per junction class, smallest first; see ``cycle_tier``.
    :param float max_cycle_cap: Hard ceiling on any cycle, in seconds, even when pedestrians need more.
    :param tuple major_green: Minimum and maximum green of a major phase, in seconds.
    :param tuple minor_green: Minimum and maximum green of a minor phase, in seconds.
    :param float major_ratio: A phase whose road-class weight is below this fraction of
        the largest weight is minor; the others are major.
    :param float ped_speed: Walking speed for the pedestrian clearance, in m/s
        (MUTCD 4E.06: 3.5 ft/s).
    :param float min_walk: Shortest WALK interval, in seconds (MUTCD 4E.06: 7 s).
    :param bool countdown: Whether pedestrian heads show a countdown (informational).
    """

    name: str = "US-MUTCD"
    reaction_time: float = 1.0
    deceleration: float = 3.0
    yellow_bounds: Tuple[float, float] = (3.0, 6.0)
    all_red_bounds: Tuple[float, float] = (1.0, 2.5)
    vehicle_length: float = 5.0
    default_speed: float = 11.18
    startup_lost: float = 2.0
    extension: float = 2.0
    saturation_flow: float = 1800.0
    major_demand: Tuple[float, float] = (600.0, 900.0)
    minor_demand: Tuple[float, float] = (200.0, 400.0)
    max_flow_ratio: float = 0.85
    cycle_bounds: Tuple[float, float] = (40.0, 120.0)
    cycle_tiers: Tuple["CycleTier", ...] = ()
    max_cycle_cap: float = 180.0
    major_green: Tuple[float, float] = (12.0, 60.0)
    minor_green: Tuple[float, float] = (6.0, 40.0)
    major_ratio: float = 0.8
    ped_speed: float = 1.07
    min_walk: float = 7.0
    countdown: bool = True


US_MUTCD = RegionProfile(cycle_tiers=CYCLE_TIERS)
TAIWAN = RegionProfile(
    name="TW-MOTC", cycle_tiers=CYCLE_TIERS, yellow_bounds=(3.0, 4.0), all_red_bounds=(1.0, 2.0), default_speed=11.11,
    saturation_flow=1900.0, ped_speed=1.0, min_walk=7.0, countdown=True,
)
"""Taiwan practice (MOTC Traffic Engineering Manual; Rules for Road Traffic Signs, Markings and
Signals): yellow 3 s up to 50 km/h and 4 s above, all-red 1 to 2 s, walking speed 1.0 m/s,
countdown pedestrian heads. See docs/traffic-signal-chromosome.md for sources and assumptions."""

MAP_REGIONS: Dict[str, RegionProfile] = {
    "borregas_ave": US_MUTCD, "borregas_ave_fix": US_MUTCD, "san_mateo": US_MUTCD,
    "sunnyvale_loop": US_MUTCD, "sunnyvale_big_loop": US_MUTCD, "sunnyvale_with_two_offices": US_MUTCD,
    "shalun": TAIWAN,
}


def region_for_map(name: str) -> RegionProfile:
    """The map's region profile; US practice for unknown maps."""
    return MAP_REGIONS.get(name, US_MUTCD)


def cycle_tier(region: RegionProfile, phases: int, speed: float, lanes: int,
               ped_floor: float = 0.0) -> Tuple[str, Tuple[float, float]]:
    """
    The junction's (tier name, cycle bounds).

    :param float ped_floor: Shortest cycle the minimum greens (pedestrians included) allow.
        Above the tier's maximum, the maximum rises to it (rounded up to 5 s), up to
        ``region.max_cycle_cap``: pedestrian green wins over the tier, never the other way.
    """
    name, (low, high) = "collector", region.cycle_bounds
    for tier in reversed(region.cycle_tiers):
        if tier.admits(phases, speed, lanes):
            name, (low, high) = tier.name, tier.bounds
            break
    if ped_floor > high + EPS:
        high = min(region.max_cycle_cap, max(high, _round_up(ped_floor, 5.0)))
    return name, (low, high)


def _round_up(value: float, step: float = 0.1) -> float:
    return round(ceil(value / step - 1e-9) * step, 6)


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


@dataclass(frozen=True)
class Crosswalk:
    """A crosswalk at a junction, with the phase whose green it walks in."""

    crosswalk_id: str
    length: float
    phase: int


@dataclass(frozen=True)
class SignalPlan:
    """
    The signal plan of one junction (the gene).

    ``demand`` and ``offset`` are the free genes. ``green``, ``yellow`` and ``all_red``
    are derived from them by ``SignalTiming.plan`` and are stored so consumers need no map.

    :param str junction_id: Junction, or ``+``-joined junctions sharing signals, or a lone signal ID.
    :param tuple phases: Signal-ID groups in running order; one group is one EQ set.
    :param tuple demand: Critical-lane demand of each phase, in veh/h/lane.
    :param tuple green: Green of each phase, in seconds.
    :param float yellow: Yellow after every green, in seconds.
    :param float all_red: All-red after every yellow, in seconds.
    :param float offset: Seconds into the cycle at t = 0, in [0, cycle).
    :param tuple walks: (crosswalk ID, phase, WALK, FLASHING DON'T WALK) per crosswalk.
    """

    junction_id: str
    phases: Tuple[Tuple[str, ...], ...]
    demand: Tuple[float, ...]
    green: Tuple[float, ...]
    yellow: float
    all_red: float
    offset: float
    walks: Tuple[Tuple[str, int, float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(tuple(group) for group in self.phases))
        object.__setattr__(self, "demand", tuple(float(q) for q in self.demand))
        object.__setattr__(self, "green", tuple(float(g) for g in self.green))
        object.__setattr__(self, "walks", tuple(tuple(w) for w in self.walks))
        if not self.phases or len(self.green) != len(self.phases) or len(self.demand) != len(self.phases):
            raise ValueError("need one green and one demand per phase")
        if min(self.green) <= 0 or self.yellow <= 0 or self.all_red < 0:
            raise ValueError("green and yellow must be positive, all-red nonnegative")
        if not 0 <= self.offset < self.cycle:
            raise ValueError("offset must lie in [0, cycle)")

    @property
    def cycle(self) -> float:
        return sum(self.green) + len(self.phases) * (self.yellow + self.all_red)

    @property
    def signals(self) -> Tuple[str, ...]:
        return tuple(s for group in self.phases for s in group)

    def phase_at(self, t: float) -> Tuple[int, SignalColor, float]:
        """(running phase, its colour, seconds since that phase's green started) at time ``t``."""
        u = round((t + self.offset) % self.cycle, 6) % self.cycle
        for index, green in enumerate(self.green):
            span = green + self.yellow + self.all_red
            if u < span or index == len(self.green) - 1:
                if u < green:
                    return index, SignalColor.GREEN, u
                if u < green + self.yellow:
                    return index, SignalColor.YELLOW, u
                return index, SignalColor.RED, u
            u -= span
        raise AssertionError("unreachable")

    def color(self, signal_id: str, t: float) -> SignalColor:
        """Colour of ``signal_id`` at time ``t``: its phase's colour while that phase runs, red otherwise."""
        index, color, _ = self.phase_at(t)
        if signal_id in self.phases[index]:
            return color
        if signal_id not in self.signals:
            raise KeyError(signal_id)
        return SignalColor.RED

    def ped_state(self, crosswalk_id: str, t: float) -> PedState:
        """
        The crosswalk's pedestrian head at time ``t``. It shows WALK from the start of its
        phase's green, then FLASHING DON'T WALK ending at the onset of yellow, then DON'T WALK.
        """
        for cid, phase, walk, clearance in self.walks:
            if cid == crosswalk_id:
                index, _, since = self.phase_at(t)
                if index != phase or since >= round(walk + clearance, 6):
                    return PedState.DONT_WALK
                return PedState.WALK if since < round(walk, 6) else PedState.FLASHING_DONT_WALK
        raise KeyError(crosswalk_id)

    def to_dict(self) -> dict:
        return {"junction_id": self.junction_id, "phases": [list(p) for p in self.phases],
                "demand": list(self.demand), "green": list(self.green), "yellow": self.yellow,
                "all_red": self.all_red, "offset": self.offset, "walks": [list(w) for w in self.walks]}

    @classmethod
    def from_dict(cls, data: dict) -> "SignalPlan":
        return cls(data["junction_id"], data["phases"], data["demand"], data["green"], data["yellow"],
                   data["all_red"], data["offset"], data.get("walks", ()))


class InvalidSignalPlan(ValueError):
    """A plan breaks its junction's timing rules."""


@dataclass(frozen=True)
class SignalTiming:
    """
    What a junction's geometry and region fix: phases, clearances, green bounds and
    the demand range of each phase. ``plan`` derives a Webster plan from the genes.

    :param str junction_id: As in ``SignalPlan``.
    :param tuple phases: EQ groups in running order.
    :param tuple major: Whether each phase is major (road class: lanes x speed limit).
    :param tuple speeds: Approach speed limit of each phase, in m/s.
    :param tuple widths: Junction extent along each phase's approach, in metres.
    :param float yellow: ITE yellow, rounded up to 0.1 s and clamped to the region.
    :param float all_red: ITE all-red, rounded up to 0.1 s and clamped to the region.
    :param tuple min_green: Each phase's minimum green, including its pedestrians.
    :param tuple max_green: Each phase's maximum green.
    :param tuple crosswalks: The junction's crosswalks with the phase they walk in.
    :param RegionProfile region: The region profile.
    :param str tier: Cycle tier name (``cycle_tier``).
    :param tuple cycle_bounds: The junction's (min, max) cycle; ``None`` means the region's.
    """

    junction_id: str
    phases: Tuple[Tuple[str, ...], ...]
    major: Tuple[bool, ...]
    speeds: Tuple[float, ...]
    widths: Tuple[float, ...]
    yellow: float
    all_red: float
    min_green: Tuple[float, ...]
    max_green: Tuple[float, ...]
    crosswalks: Tuple[Crosswalk, ...]
    region: RegionProfile = field(default=US_MUTCD, compare=False)
    tier: str = "collector"
    cycle_bounds: Tuple[float, float] | None = None

    @property
    def bounds(self) -> Tuple[float, float]:
        """(min, max) cycle of this junction, in seconds."""
        return self.cycle_bounds if self.cycle_bounds is not None else self.region.cycle_bounds

    @property
    def clearance(self) -> float:
        return len(self.phases) * (self.yellow + self.all_red)

    def demand_bounds(self, index: int) -> Tuple[float, float]:
        return self.region.major_demand if self.major[index] else self.region.minor_demand

    @property
    def lost_time(self) -> float:
        """L: per phase, start-up lost time plus the clearance not used by traffic."""
        r = self.region
        return len(self.phases) * (r.startup_lost + max(0.0, self.yellow + self.all_red - r.extension))

    def webster(self, demand: Sequence[float]) -> Tuple[float, Tuple[float, ...]]:
        """Webster's (clamped cycle, greens) before the min/max green clamp."""
        r = self.region
        ratios = [q / r.saturation_flow for q in demand]
        total = sum(ratios)
        if total > r.max_flow_ratio:            # Oversaturated: keep the split, cap Y.
            ratios = [y * r.max_flow_ratio / total for y in ratios]
            total = r.max_flow_ratio
        lost = self.lost_time
        cycle = _clamp((1.5 * lost + 5) / (1 - total), self.bounds)
        n = len(self.phases)
        effective = cycle - lost
        # Displayed green = effective green + lost - (yellow + all-red), per phase.
        offset = (lost / n) - (self.yellow + self.all_red)
        return cycle, tuple(effective * y / total + offset for y in ratios)

    def plan(self, demand: Sequence[float], phase: float) -> SignalPlan:
        """
        The plan for ``demand`` (veh/h/lane per phase) with offset ``phase`` x cycle.

        :param float phase: Offset as a fraction of the cycle, in [0, 1).
        """
        _, greens = self.webster(demand)
        greens = self._fit([_clamp(g, (lo, hi)) for g, lo, hi in zip(greens, self.min_green, self.max_green)])
        cycle = sum(greens) + self.clearance
        walks = tuple((c.crosswalk_id, c.phase, round(greens[c.phase] - c.length / self.region.ped_speed, 3),
                       round(c.length / self.region.ped_speed, 3)) for c in self.crosswalks)
        return SignalPlan(self.junction_id, self.phases, tuple(demand), greens, self.yellow, self.all_red,
                          (phase % 1.0) * cycle, walks)

    def _fit(self, greens: list) -> Tuple[float, ...]:
        """
        Greens moved back into the cycle bounds after the min/max green clamp: phases with
        room give back (or take) time in proportion to it. Minimum greens, and so the
        pedestrian intervals, are never cut. Rounded to 0.1 s towards the bounds.
        """
        low, high = self.bounds
        lows, highs = self.min_green, self.max_green
        for _ in range(len(greens) + 1):
            cycle = sum(greens) + self.clearance
            if cycle > high + EPS:
                excess, room = cycle - high, [g - lo for g, lo in zip(greens, lows)]
            elif cycle < low - EPS:
                excess, room = cycle - low, [g - hi for g, hi in zip(greens, highs)]
            else:
                break
            total = sum(room)
            if abs(total) < EPS:
                break
            share = min(1.0, excess / total)
            greens = [g - r * share for g, r in zip(greens, room)]
        out = [round(g, 1) for g in greens]
        for i, (g, lo, hi) in enumerate(zip(out, lows, highs)):
            out[i] = min(max(g, lo), hi)
        # Rounding drift: trim (or add) 0.1 s steps on phases with room.
        for _ in range(4 * len(out)):
            cycle = sum(out) + self.clearance
            if cycle > high + EPS:
                i = max(range(len(out)), key=lambda k: out[k] - lows[k])
                if out[i] - 0.1 < lows[i] - EPS:
                    break
                out[i] = round(out[i] - 0.1, 1)
            elif cycle < low - EPS:
                i = max(range(len(out)), key=lambda k: highs[k] - out[k])
                if out[i] + 0.1 > highs[i] + EPS:
                    break
                out[i] = round(out[i] + 0.1, 1)
            else:
                break
        return tuple(out)

    def validate(self, plan: SignalPlan) -> None:
        """
        :raises InvalidSignalPlan: If the plan is not this junction's Webster plan for its own
            genes, or a demand, green, clearance or pedestrian interval leaves its bounds.
        """
        if plan.junction_id != self.junction_id or plan.phases != self.phases:
            raise InvalidSignalPlan(f"{plan.junction_id}: phases differ from the junction's EQ groups")
        for index, q in enumerate(plan.demand):
            lo, hi = self.demand_bounds(index)
            if not lo - EPS <= q <= hi + EPS:
                raise InvalidSignalPlan(f"{plan.junction_id}: demand {q} outside {lo}..{hi}")
        expected = self.plan(plan.demand, plan.offset / plan.cycle)
        if (expected.green != plan.green or abs(expected.yellow - plan.yellow) > EPS
                or abs(expected.all_red - plan.all_red) > EPS or expected.walks != plan.walks):
            raise InvalidSignalPlan(f"{plan.junction_id}: timings are not the Webster plan of its demand")
        for green, lo, hi in zip(plan.green, self.min_green, self.max_green):
            if not lo - 0.05 - EPS <= green <= hi + 0.05 + EPS:
                raise InvalidSignalPlan(f"{plan.junction_id}: green {green} outside {lo}..{hi}")
        low, high = self.bounds
        if not low - EPS <= plan.cycle <= min(high, self.region.max_cycle_cap) + EPS:
            raise InvalidSignalPlan(f"{plan.junction_id}: cycle {plan.cycle:.1f} outside {low}..{high} ({self.tier})")
        for cid, phase, walk, clearance in plan.walks:
            if walk < self.region.min_walk - 0.05 - EPS or walk + clearance > plan.green[phase] + EPS:
                raise InvalidSignalPlan(f"{plan.junction_id}: crosswalk {cid} does not fit its green")

    def is_valid(self, plan: SignalPlan) -> bool:
        try:
            self.validate(plan)
        except InvalidSignalPlan:
            return False
        return True


# Junction analysis --------------------------------------------------------------------------

CROSSWALK_DISTANCE = 3.0
"""Metres between a crosswalk and a junction polygon for the crosswalk to belong to the junction."""
CROSSWALK_PARALLEL = cos(pi / 4)
"""|cos| between a crosswalk's walking axis and a phase's approach above which it walks in that phase."""
DEFAULT_WIDTH = 20.0
"""Junction extent in metres when the junction has no usable polygon."""

_TIMINGS: "WeakKeyDictionary[Any, dict]" = WeakKeyDictionary()


def _circular_mean(angles: Sequence[float]) -> float | None:
    x, y = sum(map(cos, angles)), sum(map(sin, angles))
    return None if hypot(x, y) < 1e-9 else atan2(y, x)


def _extent(area, heading: float | None) -> float:
    if area is None or area.is_empty:
        return DEFAULT_WIDTH
    rect = area.minimum_rotated_rectangle
    coords = list(getattr(rect, "exterior", rect).coords) if not rect.is_empty else []
    if heading is None:
        if len(coords) < 4:
            return DEFAULT_WIDTH
        sides = [hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(2)]
        return max(sides)
    points = [p for geom in getattr(area, "geoms", [area]) if hasattr(geom, "exterior") for p in geom.exterior.coords]
    if not points:
        return DEFAULT_WIDTH
    along = [x * cos(heading) + y * sin(heading) for x, y in points]
    return max(along) - min(along)


def _walk_axis(shape) -> Tuple[float, float]:
    """(length, direction) of a crosswalk's long side: people walk along it."""
    rect = shape.minimum_rotated_rectangle
    c = list(rect.exterior.coords)
    sides = [(hypot(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]),
              atan2(c[i + 1][1] - c[i][1], c[i + 1][0] - c[i][0])) for i in range(2)]
    return max(sides)


def control_units(service) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    """
    Signalised control units and their phases. Junctions sharing a signal merge into one unit;
    a signal at no junction is a unit of its own. Phases are the EQ components of the signal graph
    restricted to the unit, ordered by their smallest signal ID.
    """
    relations = service.relations
    graph = relations.signal_graph
    eq = nx.Graph()
    eq.add_nodes_from(graph)
    eq.add_edges_from((a, b) for a, b, r in graph.edges(data="relation") if r == "EQ")
    joint = nx.Graph()
    for junction_id, signals in relations.signals_at_junction.items():
        for s in signals:
            joint.add_edge(("j", junction_id), ("s", s))
    for s in graph:
        joint.add_node(("s", s))
    units = {}
    for component in nx.connected_components(joint):
        junctions = sorted(k for kind, k in component if kind == "j")
        signals = sorted(k for kind, k in component if kind == "s")
        if not signals:
            continue
        sub = eq.subgraph(signals)
        phases = sorted((tuple(sorted(c)) for c in nx.connected_components(sub)), key=lambda p: p[0])
        units["+".join(junctions) if junctions else signals[0]] = tuple(phases)
    return dict(sorted(units.items()))


def signal_timings(service, region: RegionProfile = US_MUTCD) -> Dict[str, SignalTiming]:
    """``SignalTiming`` of every control unit of the map (cached per map and region)."""
    try:
        cache = _TIMINGS.setdefault(service, {})
    except TypeError:
        cache = {}
    if region in cache:
        return cache[region]
    relations = service.relations
    areas = relations._junction_areas
    crosswalk_shapes = {}
    for cid, crosswalk in service.crosswalk_table.items():
        try:
            shape = polygon_to_shape(crosswalk.polygon)
        except ValueError:
            shape = None
        if shape is not None and not shape.is_empty and shape.area > 0:
            crosswalk_shapes[cid] = shape
    result = {}
    for unit, phases in control_units(service).items():
        parts = [areas.get(j) for j in unit.split("+") if j in areas]
        parts = [a for a in parts if a is not None]
        area = union_all(parts) if parts else None
        speeds, widths, weights, headings = [], [], [], []
        lane_count = 0
        for group in phases:
            lanes = sorted({lane for s in group for lane in relations.lanes_controlled_by_signal[s]})
            limits = [service.get_lane_by_id(lane).speed_limit for lane in lanes]
            limits = [v for v in limits if v and v > 0.5]
            speed = max(limits) if limits else region.default_speed
            approaches = [a for a in (relations._approach(s) for s in group) if a is not None]
            heading = _circular_mean(approaches) if approaches else None
            speeds.append(speed)
            headings.append(heading)
            widths.append(_extent(area, heading))
            weights.append(max(1, len(lanes)) * speed)
            lane_count += len(lanes)
        yellow = _clamp(_round_up(max(region.reaction_time + v / (2 * region.deceleration) for v in speeds)),
                        region.yellow_bounds)
        all_red = _clamp(_round_up(max((w + region.vehicle_length) / v for w, v in zip(widths, speeds))),
                         region.all_red_bounds)
        top = max(weights)
        major = tuple(w >= region.major_ratio * top for w in weights)
        crosswalks = []
        if area is not None and not area.is_empty:
            for cid, shape in sorted(crosswalk_shapes.items()):
                if shape.distance(area) > CROSSWALK_DISTANCE:
                    continue
                length, axis = _walk_axis(shape)
                best, score = None, CROSSWALK_PARALLEL
                for index, heading in enumerate(headings):
                    if heading is not None and abs(cos(axis - heading)) > score:
                        best, score = index, abs(cos(axis - heading))
                if best is not None:
                    crosswalks.append(Crosswalk(cid, round(length, 2), best))
        min_green, max_green = [], []
        for index in range(len(phases)):
            lo, hi = region.major_green if major[index] else region.minor_green
            ped = [region.min_walk + c.length / region.ped_speed for c in crosswalks if c.phase == index]
            lo = _round_up(max([lo] + ped))
            min_green.append(lo)
            max_green.append(max(hi, lo))
        ped_floor = sum(min_green) + len(phases) * (yellow + all_red)
        tier, bounds = cycle_tier(region, len(phases), max(speeds), lane_count, ped_floor)
        result[unit] = SignalTiming(unit, phases, major, tuple(round(v, 3) for v in speeds),
                                    tuple(round(w, 2) for w in widths), yellow, all_red,
                                    tuple(min_green), tuple(max_green), tuple(crosswalks), region,
                                    tier, bounds)
    cache[region] = result
    return result


# Section ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalSection:
    """
    The signal plans of every control unit of one map.

    :param tuple plans: One ``SignalPlan`` per control unit, in unit order.
    """

    plans: Tuple[SignalPlan, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plans", tuple(self.plans))
        index = {}
        for plan in self.plans:
            for s in plan.signals:
                if s in index:
                    raise ValueError(f"signal {s} in two plans")
                index[s] = plan
        object.__setattr__(self, "_by_signal", index)
        object.__setattr__(self, "_by_crosswalk", {w[0]: p for p in self.plans for w in p.walks})

    def __len__(self) -> int:
        return len(self.plans)

    def plan_of(self, signal_id: str) -> SignalPlan:
        return self._by_signal[signal_id]

    def color(self, signal_id: str, t: float, force_green: bool = False) -> SignalColor:
        """Colour of a signal at absolute scenario time ``t``; GREEN everywhere when ``force_green``."""
        plan = self._by_signal[signal_id]
        return SignalColor.GREEN if force_green else plan.color(signal_id, t)

    def colors(self, t: float, force_green: bool = False) -> Dict[str, SignalColor]:
        return {s: self.color(s, t, force_green) for s in sorted(self._by_signal)}

    def ped_state(self, crosswalk_id: str, t: float) -> PedState:
        """Pedestrian head of a signalised crosswalk at time ``t``."""
        return self._by_crosswalk[crosswalk_id].ped_state(crosswalk_id, t)

    @property
    def crosswalks(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_crosswalk))

    def detection(self, t: float, message_cls: Any, force_green: bool = False) -> Any:
        """
        An ``apollo.perception.TrafficLightDetection`` for time ``t``.

        :param message_cls: The message class, e.g. ``DescriptorBundle(...).message(
            "apollo.perception.TrafficLightDetection")``.
        :param bool force_green: Show every signal green (the old FORCE_INVALID_TRAFFIC_CONTROL).
        """
        message = message_cls()
        message.header.timestamp_sec = t
        message.contain_lights = bool(self._by_signal)
        for signal_id, color in self.colors(t, force_green).items():
            light = message.traffic_light.add()
            light.id = signal_id
            light.color = int(color)
            light.confidence = 1.0
            if not force_green:
                light.remaining_time = self._remaining(signal_id, t)
        return message

    def _remaining(self, signal_id: str, t: float) -> float:
        plan = self._by_signal[signal_id]
        now = plan.color(signal_id, t)
        step, ahead = 0.1, 0.0
        while ahead < plan.cycle and plan.color(signal_id, t + ahead + step) == now:
            ahead += step
        return round(ahead + step, 1)

    def validate(self, timings: Mapping[str, SignalTiming]) -> None:
        """
        :raises InvalidSignalPlan: If the plans do not cover the units of ``timings`` one to one,
            or a plan is invalid for its unit.
        """
        if [p.junction_id for p in self.plans] != list(timings):
            raise InvalidSignalPlan("plans do not match the map's control units")
        for plan in self.plans:
            timings[plan.junction_id].validate(plan)

    def is_valid(self, timings: Mapping[str, SignalTiming]) -> bool:
        try:
            self.validate(timings)
        except InvalidSignalPlan:
            return False
        return True

    def to_dict(self) -> dict:
        return {"plans": [p.to_dict() for p in self.plans]}

    @classmethod
    def from_dict(cls, data: dict) -> "SignalSection":
        return cls(tuple(SignalPlan.from_dict(p) for p in data["plans"]))
