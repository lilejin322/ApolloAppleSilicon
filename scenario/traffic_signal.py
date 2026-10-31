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
  then clamped to the region's minimum and maximum green, and to the time a
  pedestrian needs to start and finish each crosswalk the phase serves.

The GA mutates demand and offset only and recomputes every timing, so each
offspring is Webster-consistent by construction (``SignalTiming.plan``).

Colours are pure functions of the absolute scenario time ``t`` held by the
scenario runner. The maps carry vehicle heads only (three round lamps), so a
signalised crosswalk has no pedestrian head: its walkers follow the vehicle light
of the phase whose traffic runs parallel to it (``parallel_signal``).
"""

import networkx as nx
from bisect import bisect_right
from enum import IntEnum
from shapely import union_all
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass, field
from math import atan2, ceil, cos, hypot, pi, sin
from typing import Any, Dict, Mapping, Sequence, Tuple, Self, List

from ..map import MapService

EPS = 1e-6
"""Absolute tolerance for float comparisons of timings, speeds and demands."""


class SignalColor(IntEnum):
    """
    Values of ``apollo.perception.TrafficLight.Color``.
    """
    UNKNOWN = 0
    """Apollo's unset colour. A plan reports red, yellow or green."""
    RED = 1
    """Red: the running phase's all-red, and every signal outside that phase."""
    YELLOW = 2
    """Yellow of the running phase, after its green."""
    GREEN = 3
    """Green of the running phase."""


@dataclass(frozen=True)
class CycleTier:
    """
    Cycle-length class of a junction (FHWA Signal Timing Manual, 2nd ed., ch. 6).

    A junction falls into the first tier, from the largest, whose every threshold it meets.

    :param str name: Tier name.
    :param Tuple[float, float] bounds: Smallest and largest cycle, in seconds.
    :param int min_phases: Fewest phases.
    :param float min_speed: Lowest fastest-approach speed limit, in m/s.
    :param int min_lanes: Fewest controlled lanes (all phases).
    """
    name: str
    """Tier name."""
    bounds: Tuple[float, float]
    """Smallest and largest cycle, in seconds."""
    min_phases: int = 0
    """Fewest phases."""
    min_speed: float = 0.0
    """Lowest fastest-approach speed limit, in m/s."""
    min_lanes: int = 0
    """Fewest controlled lanes (all phases)."""

    def admits(self, phases: int, speed: float, lanes: int) -> bool:
        """
        Whether a junction meets every threshold of this tier.

        :param int phases: Phase count.
        :param float speed: Fastest-approach speed limit, in m/s.
        :param int lanes: Controlled lanes, all phases.
        :returns: ``True`` when each of ``phases``, ``speed`` and ``lanes`` is at least this tier's minimum.
        :rtype: bool
        """
        return phases >= self.min_phases and speed >= self.min_speed - EPS and lanes >= self.min_lanes


LOCAL = CycleTier("local", (40.0, 90.0))
"""Cycle 40 to 90 s, with no phase, speed, or lane threshold."""
COLLECTOR = CycleTier("collector", (45.0, 120.0), min_lanes=9)
"""Cycle 45 to 120 s for a junction with at least 9 controlled lanes."""
ARTERIAL_SPEED = CycleTier("arterial", (60.0, 150.0), min_speed=15.6)
"""Cycle 60 to 150 s when the fastest approach is at least 15.6 m/s (35 mph)."""
ARTERIAL_PHASES = CycleTier("arterial", (60.0, 150.0), min_phases=3)
"""Cycle 60 to 150 s for a junction with at least 3 phases."""
MAJOR = CycleTier("major", (80.0, 180.0), min_phases=4, min_speed=15.6)
"""Cycle 80 to 180 s for at least 4 phases and a fastest approach of 15.6 m/s (35 mph)."""
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
    :param Tuple[float, float] yellow_bounds: Smallest and largest yellow, in seconds (MUTCD 4D.26: 3 to 6 s).
    :param Tuple[float, float] all_red_bounds: Smallest and largest red clearance, in seconds.
    :param float vehicle_length: L in the all-red formula, in metres.
    :param float default_speed: Approach speed in m/s when the map has no limit (0 or missing).
    :param float startup_lost: Start-up lost time per phase, in seconds (HCM: 2 s).
    :param float extension: Part of yellow + all-red still used by traffic, in seconds (HCM: 2 s).
    :param float saturation_flow: Saturation flow s, in veh/h/lane.
    :param Tuple[float, float] major_demand: Demand range of a major phase's critical lane, in veh/h/lane.
    :param Tuple[float, float] minor_demand: Demand range of a minor phase's critical lane, in veh/h/lane.
    :param float max_flow_ratio: Cap on Y. A larger Y is oversaturated; it is scaled
        down to this cap before Webster, which gives the longest cycle.
    :param Tuple[float, float] cycle_bounds: Cycle bounds of a junction built without a tier (collector), in seconds.
    :param Tuple[CycleTier, ...] cycle_tiers: ``CycleTier`` per junction class, smallest first; see ``cycle_tier``.
    :param float max_cycle_cap: Hard ceiling on any cycle, in seconds, even when pedestrians need more.
    :param Tuple[float, float] major_green: Minimum and maximum green of a major phase, in seconds.
    :param Tuple[float, float] minor_green: Minimum and maximum green of a minor phase, in seconds.
    :param float major_ratio: A phase whose road-class weight is below this fraction of
        the largest weight is minor; the others are major.
    :param float ped_speed: Walking speed for the crossing time in a minimum green, in m/s
        (MUTCD 4E.06: 3.5 ft/s).
    :param float min_walk: Start-up time a pedestrian gets before the crossing time in a minimum
        green, in seconds (MUTCD 4E.06's shortest WALK: 7 s).
    """
    name: str = "US-MUTCD"
    """Profile name."""
    reaction_time: float = 1.0
    """Perception-reaction time in the ITE yellow formula, in seconds."""
    deceleration: float = 3.0
    """Comfortable deceleration in the ITE yellow formula, in m/s²."""
    yellow_bounds: Tuple[float, float] = (3.0, 6.0)
    """Smallest and largest yellow, in seconds (MUTCD 4D.26: 3 to 6 s)."""
    all_red_bounds: Tuple[float, float] = (1.0, 2.5)
    """Smallest and largest red clearance, in seconds."""
    vehicle_length: float = 5.0
    """L in the all-red formula, in metres."""
    default_speed: float = 11.18
    """Approach speed in m/s when the map has no limit (0 or missing)."""
    startup_lost: float = 2.0
    """Start-up lost time per phase, in seconds (HCM: 2 s)."""
    extension: float = 2.0
    """Part of yellow + all-red still used by traffic, in seconds (HCM: 2 s)."""
    saturation_flow: float = 1800.0
    """Saturation flow s, in veh/h/lane."""
    major_demand: Tuple[float, float] = (600.0, 900.0)
    """Demand range of a major phase's critical lane, in veh/h/lane."""
    minor_demand: Tuple[float, float] = (200.0, 400.0)
    """Demand range of a minor phase's critical lane, in veh/h/lane."""
    max_flow_ratio: float = 0.85
    """Cap on Y. A larger Y is oversaturated; it is scaled down to this cap before Webster, which gives the longest cycle."""
    cycle_bounds: Tuple[float, float] = (40.0, 120.0)
    """Cycle bounds of a junction built without a tier (collector), in seconds."""
    cycle_tiers: Tuple["CycleTier", ...] = ()
    """``CycleTier`` per junction class, smallest first; see ``cycle_tier``."""
    max_cycle_cap: float = 180.0
    """Hard ceiling on any cycle, in seconds, even when pedestrians need more."""
    major_green: Tuple[float, float] = (12.0, 60.0)
    """Minimum and maximum green of a major phase, in seconds."""
    minor_green: Tuple[float, float] = (6.0, 40.0)
    """Minimum and maximum green of a minor phase, in seconds."""
    major_ratio: float = 0.8
    """A phase whose road-class weight is below this fraction of the largest weight is minor; the others are major."""
    ped_speed: float = 1.07
    """Walking speed for the crossing time in a minimum green, in m/s (MUTCD 4E.06: 3.5 ft/s)."""
    min_walk: float = 7.0
    """Start-up time before the crossing time in a minimum green, in seconds (MUTCD 4E.06's shortest WALK: 7 s)."""


US_MUTCD = RegionProfile(cycle_tiers=CYCLE_TIERS)
"""US practice (MUTCD 2009 and ITE): yellow 3 to 6 s, all-red 1 to 2.5 s, walking
speed 1.07 m/s (3.5 ft/s), pedestrian start-up 7 s. Other fields keep
``RegionProfile`` defaults; cycle length follows ``CYCLE_TIERS``."""
TAIWAN = RegionProfile(
    name="TW-MOTC", cycle_tiers=CYCLE_TIERS, yellow_bounds=(3.0, 4.0), all_red_bounds=(1.0, 2.0), default_speed=11.11,
    saturation_flow=1900.0, ped_speed=1.0, min_walk=7.0,
)
"""Taiwan practice (MOTC Traffic Engineering Manual; Rules for Road Traffic Signs, Markings and
Signals): yellow 3 s up to 50 km/h and 4 s above, all-red 1 to 2 s, walking speed 1.0 m/s."""

MAP_REGIONS: Dict[str, RegionProfile] = {
    "borregas_ave": US_MUTCD, "borregas_ave_fix": US_MUTCD, "san_mateo": US_MUTCD,
    "sunnyvale_loop": US_MUTCD, "sunnyvale_big_loop": US_MUTCD, "sunnyvale_with_two_offices": US_MUTCD,
    "shalun": TAIWAN,
}
"""Region profile of each map. Borregas Ave, San Mateo and the Sunnyvale maps use
``US_MUTCD``; Shalun uses ``TAIWAN``. ``region_for_map`` returns ``US_MUTCD`` for any other name."""


def region_for_map(name: str) -> RegionProfile:
    """
    The map's region profile; US practice for unknown maps.

    :param str name: Map name, as keyed in ``MAP_REGIONS``.
    :returns: That map's profile, or ``US_MUTCD`` when ``name`` is absent.
    :rtype: RegionProfile
    """
    return MAP_REGIONS.get(name, US_MUTCD)


def cycle_tier(region: RegionProfile, phases: int, speed: float, lanes: int,
               ped_floor: float = 0.0) -> Tuple[str, Tuple[float, float]]:
    """
    The junction's (tier name, cycle bounds).

    The first tier in ``region.cycle_tiers``, from the largest, that admits
    ``phases``, ``speed`` and ``lanes``. With none, the name is ``"collector"``
    and the bounds are ``region.cycle_bounds``.

    :param RegionProfile region: The region profile, including its tiers.
    :param int phases: Phase count.
    :param float speed: Fastest-approach speed limit, in m/s.
    :param int lanes: Controlled lanes, all phases.
    :param float ped_floor: Shortest cycle the minimum greens (pedestrians included) allow.
        Above the tier's maximum, the maximum rises to it (rounded up to 5 s), up to
        ``region.max_cycle_cap``: pedestrian green wins over the tier, never the other way.
    :returns: ``(tier name, (minimum cycle, maximum cycle))``, in seconds.
    :rtype: Tuple[str, Tuple[float, float]]
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
    """
    ``value`` rounded up to a multiple of ``step``.

    A value already on a multiple stays there: ``1e-9`` absorbs floating-point
    noise before ``ceil``. The result is then rounded to 6 decimal places.

    :param float value: The quantity to round, in the same unit as ``step``.
    :param float step: Grid size. The default, 0.1, is the timing resolution in seconds.
    :returns: The rounded value.
    :rtype: float
    """
    return round(ceil(value / step - 1e-9) * step, 6)


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    """
    ``value`` limited to ``bounds``.

    :param float value: The quantity to limit, in the same unit as ``bounds``.
    :param Tuple[float, float] bounds: ``(minimum, maximum)``.
    :returns: ``value`` when it lies inside ``bounds``; the nearer endpoint otherwise.
    :rtype: float
    """
    return min(max(value, bounds[0]), bounds[1])


@dataclass(frozen=True)
class Crosswalk:
    """
    A crosswalk at a junction, with the phase whose traffic runs parallel to it.

    :param str crosswalk_id: Map crosswalk ID.
    :param float length: Crossing distance along the walk axis, in metres.
    :param int phase: Index of the phase whose light this crosswalk's walkers follow.
    """
    crosswalk_id: str
    """Map crosswalk ID."""
    length: float
    """Crossing distance along the walk axis, in metres."""
    phase: int
    """Index of the phase whose light this crosswalk's walkers follow."""


@dataclass(frozen=True)
class SignalPlan:
    """
    The signal plan of one junction (the gene).

    ``demand`` and ``offset`` are the free genes. ``green``, ``yellow`` and ``all_red``
    are derived from them by ``SignalTiming.plan`` and are stored so consumers need no map.

    :param str junction_id: Junction, or ``+``-joined junctions sharing signals, or a lone signal ID.
    :param Tuple[Tuple[str, ...], ...] phases: Signal-ID groups in running order; one group is one EQ set.
    :param Tuple[float, ...] demand: Critical-lane demand of each phase, in veh/h/lane.
    :param Tuple[float, ...] green: Green of each phase, in seconds.
    :param float yellow: Yellow after every green, in seconds.
    :param float all_red: All-red after every yellow, in seconds.
    :param float offset: Seconds into the cycle at t = 0, in [0, cycle).
    :param Tuple[Tuple[str, int], ...] crosswalks: (crosswalk ID, phase) per signalised crosswalk: its walkers
        follow that phase's light.
    """
    junction_id: str
    """Junction, or ``+``-joined junctions sharing signals, or a lone signal ID."""
    phases: Tuple[Tuple[str, ...], ...]
    """Signal-ID groups in running order; one group is one EQ set."""
    demand: Tuple[float, ...]
    """Critical-lane demand of each phase, in veh/h/lane."""
    green: Tuple[float, ...]
    """Green of each phase, in seconds."""
    yellow: float
    """Yellow after every green, in seconds."""
    all_red: float
    """All-red after every yellow, in seconds."""
    offset: float
    """Seconds into the cycle at t = 0, in [0, cycle)."""
    crosswalks: Tuple[Tuple[str, int], ...] = ()
    """(crosswalk ID, phase) per signalised crosswalk: its walkers follow that phase's light."""

    def __post_init__(self) -> None:
        """
        Freeze the sequences, reject a plan that cannot run, and index its switch points.

        ``phases``, ``demand``, ``green`` and ``crosswalks`` become tuples, and ``demand`` and ``green`` become floats.
        Every switch point (green, yellow and all-red onsets) is stored as a cycle position rounded to 1 us,
        the grid ``phase_at`` reads the cycle on, so a switch never lags by a float step.

        :raises ValueError: Unless there is one positive green and one demand per phase, ``yellow`` is positive,
            ``all_red`` is nonnegative, ``offset`` lies in ``[0, cycle)``, no signal is in two phases,
            and every crosswalk follows one of the phases.
        """
        object.__setattr__(self, "phases", tuple(tuple(group) for group in self.phases))
        object.__setattr__(self, "demand", tuple(float(q) for q in self.demand))
        object.__setattr__(self, "green", tuple(float(g) for g in self.green))
        object.__setattr__(self, "crosswalks", tuple(tuple(c) for c in self.crosswalks))
        if not self.phases or len(self.green) != len(self.phases) or len(self.demand) != len(self.phases):
            raise ValueError("need one green and one demand per phase")
        if min(self.green) <= 0 or self.yellow <= 0 or self.all_red < 0:
            raise ValueError("green and yellow must be positive, all-red nonnegative")
        if not 0 <= self.offset < self.cycle:
            raise ValueError("offset must lie in [0, cycle)")
        phase_of: Dict[str, int] = {}
        for index, group in enumerate(self.phases):
            for s in group:
                if s in phase_of:
                    raise ValueError(f"signal {s} in two phases")
                phase_of[s] = index
        onsets, ends, start = [], [], 0.0
        for green in self.green:
            onsets.append(round(start, 6))
            ends.append((round(start + green, 6), round(start + green + self.yellow, 6)))
            start += green + self.yellow + self.all_red
        parallel: Dict[str, str] = {}
        for cid, phase in self.crosswalks:
            if not 0 <= phase < len(self.phases) or not self.phases[phase]:
                raise ValueError(f"crosswalk {cid} follows no phase")
            parallel.setdefault(cid, min(self.phases[phase]))
        object.__setattr__(self, "_phase_of", phase_of)
        object.__setattr__(self, "_onsets", tuple(onsets))
        object.__setattr__(self, "_ends", tuple(ends))
        object.__setattr__(self, "_parallel", parallel)

    @property
    def cycle(self) -> float:
        """
        Length of one cycle, in seconds.

        The sum of the phase greens, plus one yellow and one all-red per phase.

        :returns: The cycle length.
        :rtype: float
        """
        return sum(self.green) + len(self.phases) * (self.yellow + self.all_red)

    @property
    def signals(self) -> Tuple[str, ...]:
        """
        Signal IDs of every phase, in running order.

        :returns: One ID per signal, phases in order and signals in their group order.
        :rtype: Tuple[str, ...]
        """
        return tuple(s for group in self.phases for s in group)

    def _position(self, t: float) -> Tuple[int, float]:
        """
        (running phase, cycle position) at time ``t``.

        :param float t: Absolute scenario time, in seconds.
        :returns: The phase whose span holds the position, and ``(t + offset)`` wrapped into ``[0, cycle)``
            and rounded to 1 us, the grid of the switch points.
        :rtype: Tuple[int, float]
        """
        u = round((t + self.offset) % self.cycle, 6) % self.cycle
        return bisect_right(self._onsets, u) - 1, u

    def _wait(self, position: float, u: float) -> float:
        """
        Seconds from cycle position ``u`` to the next time the cycle reaches ``position``.

        :param float position: Target cycle position.
        :param float u: Current cycle position.
        :returns: The wait, in ``(0, cycle]``: a full cycle when ``u`` is already at ``position``.
        :rtype: float
        """
        ahead = position - u
        return ahead if ahead > 0 else ahead + self.cycle

    def phase_at(self, t: float) -> Tuple[int, SignalColor, float]:
        """
        (running phase, its colour, seconds since that phase's green started) at time ``t``.

        The cycle position is ``(t + offset)`` wrapped into ``[0, cycle)`` and read to 1 us. Green, yellow and all-red follow in that order, and the third value keeps counting from the green.

        :param float t: Absolute scenario time, in seconds.
        :returns: ``(phase index, colour, seconds since that phase's green started)``.
        :rtype: Tuple[int, SignalColor, float]
        """
        index, u = self._position(t)
        green_end, yellow_end = self._ends[index]
        color = SignalColor.GREEN if u < green_end else SignalColor.YELLOW if u < yellow_end else SignalColor.RED
        return index, color, u - self._onsets[index]

    def color(self, signal_id: str, t: float) -> SignalColor:
        """
        Colour of ``signal_id`` at time ``t``: its phase's colour while that phase runs, red otherwise.

        :param str signal_id: A signal of this plan.
        :param float t: Absolute scenario time, in seconds.
        :returns: The running phase's colour when ``signal_id`` belongs to it, otherwise red.
        :rtype: SignalColor
        :raises KeyError: If ``signal_id`` is not one of this plan's signals.
        """
        phase = self._phase_of[signal_id]
        index, color, _ = self.phase_at(t)
        return color if index == phase else SignalColor.RED

    def parallel_signal(self, crosswalk_id: str) -> str:
        """
        The signal a crosswalk's walkers follow: one of the phase whose traffic runs parallel to it
        (all signals of a phase show one colour). The maps have no pedestrian heads.

        :param str crosswalk_id: A crosswalk of this plan.
        :returns: The smallest signal ID of that phase.
        :rtype: str
        :raises KeyError: If ``crosswalk_id`` is not one of this plan's crosswalks.
        """
        return self._parallel[crosswalk_id]

    def remaining(self, signal_id: str, t: float) -> float:
        """
        Seconds from ``t`` until ``signal_id`` changes colour.

        Green counts down to its yellow, yellow to the next colour (red, or green again for a lone
        phase without all-red), and red to the next start of the signal's green.

        :param str signal_id: A signal of this plan.
        :param float t: Absolute scenario time, in seconds.
        :returns: The countdown, in ``(0, cycle]``.
        :rtype: float
        :raises KeyError: If ``signal_id`` is not one of this plan's signals.
        """
        phase = self._phase_of[signal_id]
        index, u = self._position(t)
        if index == phase:
            for end in self._ends[phase]:
                if u < end:
                    return end - u
        return self._wait(self._onsets[phase], u)

    def next_green(self, signal_id: str, t: float) -> float:
        """
        First instant at or after ``t`` at which ``signal_id`` shows green.

        :param str signal_id: A signal of this plan.
        :param float t: Absolute scenario time, in seconds.
        :returns: ``t`` while the signal is green, otherwise the next start of its green.
        :rtype: float
        :raises KeyError: If ``signal_id`` is not one of this plan's signals.
        """
        phase = self._phase_of[signal_id]
        index, u = self._position(t)
        if index == phase and u < self._ends[phase][0]:
            return t
        return t + self._wait(self._onsets[phase], u)

    def to_dict(self) -> Dict[str, Any]:
        """
        A JSON-compatible mapping of this plan.

        Tuples become lists. The keys are ``junction_id``, ``phases``, ``demand``, ``green``, ``yellow``, ``all_red``, ``offset`` and ``crosswalks``.

        :returns: A new mapping that ``from_dict`` can rebuild.
        :rtype: Dict[str, Any]
        """
        return {"junction_id": self.junction_id, "phases": [list(p) for p in self.phases],
                "demand": list(self.demand), "green": list(self.green), "yellow": self.yellow,
                "all_red": self.all_red, "offset": self.offset, "crosswalks": [list(c) for c in self.crosswalks]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Build a plan from ``to_dict`` output.

        ``crosswalks`` may be absent and then defaults to none. The other keys are required.

        :param Dict[str, Any] data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed plan.
        :rtype: SignalPlan
        :raises KeyError: If a required field is absent.
        :raises ValueError: If the rebuilt plan fails ``__post_init__``.
        """
        return cls(data["junction_id"], data["phases"], data["demand"], data["green"], data["yellow"],
                   data["all_red"], data["offset"], data.get("crosswalks", ()))


class InvalidSignalPlan(ValueError):
    """
    A plan breaks its junction's timing rules.
    """


@dataclass(frozen=True)
class SignalTiming:
    """
    What a junction's geometry and region fix: phases, clearances, green bounds and
    the demand range of each phase. ``plan`` derives a Webster plan from the genes.

    :param str junction_id: As in ``SignalPlan``.
    :param Tuple[Tuple[str, ...], ...] phases: EQ groups in running order.
    :param Tuple[bool, ...] major: Whether each phase is major (road class: lanes x speed limit).
    :param Tuple[float, ...] speeds: Approach speed limit of each phase, in m/s.
    :param Tuple[float, ...] widths: Junction extent along each phase's approach, in metres.
    :param float yellow: ITE yellow, rounded up to 0.1 s and clamped to the region.
    :param float all_red: ITE all-red, rounded up to 0.1 s and clamped to the region.
    :param Tuple[float, ...] min_green: Each phase's minimum green, including its pedestrians.
    :param Tuple[float, ...] max_green: Each phase's maximum green.
    :param Tuple[Crosswalk, ...] crosswalks: The junction's crosswalks with the phase they walk in.
    :param RegionProfile region: The region profile.
    :param str tier: Cycle tier name (``cycle_tier``).
    :param Tuple[float, float] | None cycle_bounds: The junction's (min, max) cycle; ``None`` means the region's.
    """
    junction_id: str
    """As in ``SignalPlan``."""
    phases: Tuple[Tuple[str, ...], ...]
    """EQ groups in running order."""
    major: Tuple[bool, ...]
    """Whether each phase is major (road class: lanes x speed limit)."""
    speeds: Tuple[float, ...]
    """Approach speed limit of each phase, in m/s."""
    widths: Tuple[float, ...]
    """Junction extent along each phase's approach, in metres."""
    yellow: float
    """ITE yellow, rounded up to 0.1 s and clamped to the region."""
    all_red: float
    """ITE all-red, rounded up to 0.1 s and clamped to the region."""
    min_green: Tuple[float, ...]
    """Each phase's minimum green, including its pedestrians."""
    max_green: Tuple[float, ...]
    """Each phase's maximum green."""
    crosswalks: Tuple[Crosswalk, ...]
    """The junction's crosswalks with the phase they walk in."""
    region: RegionProfile = field(default=US_MUTCD, compare=False)
    """The region profile."""
    tier: str = "collector"
    """Cycle tier name (``cycle_tier``)."""
    cycle_bounds: Tuple[float, float] | None = None
    """The junction's (min, max) cycle; ``None`` means the region's."""

    @property
    def bounds(self) -> Tuple[float, float]:
        """
        (min, max) cycle of this junction, in seconds.

        ``cycle_bounds`` when it is set; otherwise ``region.cycle_bounds``.

        :returns: ``(minimum cycle, maximum cycle)``.
        :rtype: Tuple[float, float]
        """
        return self.cycle_bounds if self.cycle_bounds is not None else self.region.cycle_bounds

    @property
    def clearance(self) -> float:
        """
        Total yellow and all-red in one cycle, in seconds.

        One yellow and one all-red per phase.

        :returns: The total clearance.
        :rtype: float
        """
        return len(self.phases) * (self.yellow + self.all_red)

    def demand_bounds(self, index: int) -> Tuple[float, float]:
        """
        Demand range of phase ``index``, in veh/h/lane.

        A major phase uses ``region.major_demand``; a minor phase uses ``region.minor_demand``.

        :param int index: Phase index.
        :returns: ``(minimum demand, maximum demand)`` of that phase's critical lane.
        :rtype: Tuple[float, float]
        :raises IndexError: If ``index`` is outside ``major``.
        """
        return self.region.major_demand if self.major[index] else self.region.minor_demand

    @property
    def lost_time(self) -> float:
        """
        Lost time L of one cycle, in seconds.

        Per phase, the start-up lost time plus the unused clearance ``max(0, yellow + all_red - extension)``.

        :returns: The sum of that lost time over every phase.
        :rtype: float
        """
        r = self.region
        return len(self.phases) * (r.startup_lost + max(0.0, self.yellow + self.all_red - r.extension))

    def webster(self, demand: Sequence[float]) -> Tuple[float, Tuple[float, ...]]:
        """
        Webster's (clamped cycle, greens) before the min/max green clamp.

        ``Y`` is the sum of ``q / saturation_flow``. Above ``region.max_flow_ratio``
        the ratios keep their split and scale down to that cap, which gives the
        longest cycle. ``(1.5 L + 5) / (1 - Y)`` is then clamped to ``bounds``.
        Each displayed green is that phase's share of ``cycle - L``, shifted by
        ``L / n - (yellow + all_red)``.

        :param Sequence[float] demand: Critical-lane demand of each phase, in veh/h/lane.
        :returns: ``(cycle, displayed greens)``, with the greens not yet clamped
            to ``min_green`` and ``max_green``.
        :rtype: Tuple[float, Tuple[float, ...]]
        """
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

        Webster greens are clamped to each phase's minimum and maximum, then fitted
        back inside the cycle bounds. Each crosswalk follows its phase's light.

        :param Sequence[float] demand: Critical-lane demand of each phase, in veh/h/lane.
        :param float phase: Offset as a fraction of the cycle, wrapped into [0, 1).
        :returns: The junction's ``SignalPlan`` for these genes.
        :rtype: SignalPlan
        """
        _, greens = self.webster(demand)
        greens = self._fit([_clamp(g, (lo, hi)) for g, lo, hi in zip(greens, self.min_green, self.max_green)])
        cycle = sum(greens) + self.clearance
        return SignalPlan(self.junction_id, self.phases, tuple(demand), greens, self.yellow, self.all_red,
                          (phase % 1.0) * cycle, tuple((c.crosswalk_id, c.phase) for c in self.crosswalks))

    def default_plan(self) -> SignalPlan:
        """
        The plan used when signals are not evolved: every phase's demand at the
        middle of its range, rounded to 0.1 veh/h/lane like the genes, offset 0.

        :returns: The junction's Webster plan for that demand.
        :rtype: SignalPlan
        """
        demand = tuple(round(sum(self.demand_bounds(i)) / 2, 1) for i in range(len(self.phases)))
        return self.plan(demand, 0.0)

    def _fit(self, greens: List[float]) -> Tuple[float, ...]:
        """
        Greens moved back into the cycle bounds after the min/max green clamp: phases with
        room give back (or take) time in proportion to it. Minimum greens, and so the
        pedestrian intervals, are never cut. Rounded to 0.1 s towards the bounds.

        :param List[float] greens: One green per phase, in seconds, already clamped
            to ``min_green`` and ``max_green``.
        :returns: Those greens fitted to ``bounds`` and rounded to 0.1 s.
        :rtype: Tuple[float, ...]
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
        Check that ``plan`` is this junction's Webster plan and stays inside its bounds.

        :param SignalPlan plan: A plan for this junction.
        :raises InvalidSignalPlan: If the plan is not this junction's Webster plan for its own
            genes, or a demand, green or clearance leaves its bounds.
        """
        if plan.junction_id != self.junction_id or plan.phases != self.phases:
            raise InvalidSignalPlan(f"{plan.junction_id}: phases differ from the junction's EQ groups")
        for index, q in enumerate(plan.demand):
            lo, hi = self.demand_bounds(index)
            if not lo - EPS <= q <= hi + EPS:
                raise InvalidSignalPlan(f"{plan.junction_id}: demand {q} outside {lo}..{hi}")
        expected = self.plan(plan.demand, plan.offset / plan.cycle)
        if (expected.green != plan.green or abs(expected.yellow - plan.yellow) > EPS
                or abs(expected.all_red - plan.all_red) > EPS or expected.crosswalks != plan.crosswalks):
            raise InvalidSignalPlan(f"{plan.junction_id}: timings are not the Webster plan of its demand")
        for green, lo, hi in zip(plan.green, self.min_green, self.max_green):
            if not lo - 0.05 - EPS <= green <= hi + 0.05 + EPS:
                raise InvalidSignalPlan(f"{plan.junction_id}: green {green} outside {lo}..{hi}")
        low, high = self.bounds
        if not low - EPS <= plan.cycle <= min(high, self.region.max_cycle_cap) + EPS:
            raise InvalidSignalPlan(f"{plan.junction_id}: cycle {plan.cycle:.1f} outside {low}..{high} ({self.tier})")

    def is_valid(self, plan: SignalPlan) -> bool:
        """
        Whether ``validate`` accepts this plan.

        :param SignalPlan plan: A plan for this junction.
        :returns: ``True`` if ``validate`` accepts ``plan``; ``False`` if it raises ``InvalidSignalPlan``.
        :rtype: bool
        """
        try:
            self.validate(plan)
        except InvalidSignalPlan:
            return False
        return True


# Junction Analysis --------------------------------------------------------------------------

CROSSWALK_DISTANCE = 3.0
"""Metres between a crosswalk and a junction polygon for the crosswalk to belong to the junction."""
CROSSWALK_PARALLEL = cos(pi / 4)
"""|cos| between a crosswalk's walking axis and a phase's approach axis above which its walkers follow that phase."""
DEFAULT_WIDTH = 20.0
"""Junction extent in metres when the junction has no usable polygon."""



def _axis(angles: Sequence[float]) -> float | None:
    """
    Mean axis of directions, in radians: a direction and its opposite lie on one axis.

    A phase usually serves two opposing approaches, whose unit vectors cancel. Doubling
    each angle maps both onto the same vector, and half the direction of the doubled
    sum is their common axis. Perpendicular axes that cancel, and an empty sequence,
    have no axis.

    :param Sequence[float] angles: Directions in radians.
    :returns: The mean axis in ``(-pi/2, pi/2]``, or ``None`` when the doubled-vector sum is shorter than ``1e-9``.
    :rtype: float | None
    """
    x, y = sum(cos(2 * a) for a in angles), sum(sin(2 * a) for a in angles)
    return None if hypot(x, y) < 1e-9 else atan2(y, x) / 2


def _extent(area: BaseGeometry | None, heading: float | None) -> float:
    """
    Junction extent in metres, along the ``heading`` axis when it is known.

    With a heading, the result is the span of the polygon's exterior projected
    onto that direction. Without one, it is the longer side of the minimum
    rotated rectangle. A missing or empty area, or a shape with no exterior,
    yields ``DEFAULT_WIDTH``.

    :param BaseGeometry | None area: Junction polygon, possibly a multi-polygon, or ``None``.
    :param float | None heading: Approach axis in radians (either direction along it), or ``None``.
    :returns: The extent in metres.
    :rtype: float
    """
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


def _walk_axis(shape: Polygon) -> Tuple[float, float]:
    """
    (length, direction) of a crosswalk's long side: people walk along it.

    The long side of the minimum rotated rectangle. Its direction is the edge's
    angle, so the opposite direction is the same axis.

    :param Polygon shape: Crosswalk polygon.
    :returns: ``(length in metres, direction in radians)``.
    :rtype: Tuple[float, float]
    """
    rect = shape.minimum_rotated_rectangle
    c = list(rect.exterior.coords)
    sides = [(hypot(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]),
              atan2(c[i + 1][1] - c[i][1], c[i + 1][0] - c[i][0])) for i in range(2)]
    return max(sides)


def control_units(service: MapService) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
    """
    Signalised control units and their phases. Junctions sharing a signal merge into one unit;
    a signal at no junction is a unit of its own. Phases are the EQ components of the signal graph
    restricted to the unit, ordered by their smallest signal ID.

    :param MapService service: Map service whose ``relations`` supply the signal graph and junction membership.
    :returns: ``{unit id: phases}``, sorted by unit id. A unit id joins its junction IDs with ``+``,
        or is the signal ID when the unit has no junction. Each phase lists its signal IDs in order.
    :rtype: Dict[str, Tuple[Tuple[str, ...], ...]]
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


def signal_timings(service: MapService, region: RegionProfile = US_MUTCD) -> Dict[str, SignalTiming]:
    """
    ``SignalTiming`` of every control unit of the map (cached per map and region).

    Phases come from ``control_units``. Yellow, all-red, green bounds, crosswalks
    and the cycle tier come from the junction geometry and ``region``. Built once per
    map and region (``MapService.derived``), shared by every handle of the map.

    :param MapService service: Map service, as in ``control_units``.
    :param RegionProfile region: Timing practice. Defaults to ``US_MUTCD``.
    :returns: ``{unit id: SignalTiming}`` in ``control_units`` order. A later call
        on the same map and region returns the same mapping.
    :rtype: Dict[str, SignalTiming]
    """
    return service.derived(("signal_timings", region), lambda view: _build_timings(view, region))


def _build_timings(service: MapService, region: RegionProfile) -> Dict[str, SignalTiming]:
    """
    ``signal_timings``, uncached.

    :param MapService service: The map's own view (``MapService.derived``).
    :param RegionProfile region: Timing practice.
    :returns: As ``signal_timings``.
    :rtype: Dict[str, SignalTiming]
    """
    relations = service.relations
    areas = relations.junction_areas
    crosswalk_shapes = {cid: area for cid, area in relations.crosswalk_areas.items() if area is not None}
    result = {}
    for unit, phases in control_units(service).items():
        parts = [areas.get(j) for j in unit.split("+") if j in areas]
        parts = [a for a in parts if a is not None]
        area = union_all(parts) if parts else None
        speeds, widths, weights, axes = [], [], [], []
        lane_count = 0
        for group in phases:
            lanes = sorted({lane for s in group for lane in relations.lanes_controlled_by_signal[s]})
            limits = [service.get_lane_by_id(lane).speed_limit for lane in lanes]
            limits = [v for v in limits if v and v > 0.5]
            speed = max(limits) if limits else region.default_speed
            approaches = [a for a in (relations._approach(s) for s in group) if a is not None]
            axis = _axis(approaches)
            speeds.append(speed)
            axes.append(axis)
            widths.append(_extent(area, axis))
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
                length, walk = _walk_axis(shape)
                best, score = None, CROSSWALK_PARALLEL
                for index, axis in enumerate(axes):
                    if axis is not None and abs(cos(walk - axis)) > score:
                        best, score = index, abs(cos(walk - axis))
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
    return result


# Section ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalSection:
    """
    The signal plans of every control unit of one map.

    :param Tuple[SignalPlan, ...] plans: One ``SignalPlan`` per control unit, in unit order.
    """
    plans: Tuple[SignalPlan, ...] = ()
    """One ``SignalPlan`` per control unit, in unit order."""

    def __post_init__(self) -> None:
        """
        Freeze the plans and index them by signal and crosswalk.

        ``plans`` becomes a tuple. ``_by_signal`` maps each signal ID to its plan,
        and ``_by_crosswalk`` maps each crosswalk ID to its plan.

        :raises ValueError: If one signal ID belongs to two plans.
        """
        object.__setattr__(self, "plans", tuple(self.plans))
        index = {}
        for plan in self.plans:
            for s in plan.signals:
                if s in index:
                    raise ValueError(f"signal {s} in two plans")
                index[s] = plan
        object.__setattr__(self, "_by_signal", index)
        object.__setattr__(self, "_by_crosswalk", {c[0]: p for p in self.plans for c in p.crosswalks})

    @classmethod
    def default(cls, timings: Mapping[str, SignalTiming]) -> "SignalSection":
        """
        Every control unit's ``SignalTiming.default_plan``, in unit order: the
        lights of a scenario whose signals are not evolved. They still cycle,
        and Apollo and the replayed agents see the same ones.

        :param Mapping[str, SignalTiming] timings: ``signal_timings(service)``.
        :rtype: SignalSection
        """
        return cls(tuple(timing.default_plan() for timing in timings.values()))

    def __len__(self) -> int:
        """
        Number of control units in this section.

        :returns: The number of plans.
        :rtype: int
        """
        return len(self.plans)

    def plan_of(self, signal_id: str) -> SignalPlan:
        """
        The plan that contains ``signal_id``.

        :param str signal_id: A signal of this section.
        :returns: That signal's ``SignalPlan``.
        :rtype: SignalPlan
        :raises KeyError: If ``signal_id`` is not one of this section's signals.
        """
        return self._by_signal[signal_id]

    def color(self, signal_id: str, t: float) -> SignalColor:
        """
        Colour of a signal at absolute scenario time ``t``.

        :param str signal_id: A signal of this section.
        :param float t: Absolute scenario time, in seconds.
        :returns: That plan's colour (``SignalPlan.color``).
        :rtype: SignalColor
        :raises KeyError: If ``signal_id`` is not one of this section's signals.
        """
        return self._by_signal[signal_id].color(signal_id, t)

    def colors(self, t: float) -> Dict[str, SignalColor]:
        """
        Colour of every signal at time ``t``.

        :param float t: Absolute scenario time, in seconds.
        :returns: ``{signal id: colour}``, sorted by signal ID.
        :rtype: Dict[str, SignalColor]
        """
        return {s: self.color(s, t) for s in sorted(self._by_signal)}

    def remaining(self, signal_id: str, t: float) -> float:
        """
        Seconds from ``t`` until a signal changes colour.

        :param str signal_id: A signal of this section.
        :param float t: Absolute scenario time, in seconds.
        :returns: That plan's countdown (``SignalPlan.remaining``).
        :rtype: float
        :raises KeyError: If ``signal_id`` is not one of this section's signals.
        """
        return self._by_signal[signal_id].remaining(signal_id, t)

    def next_green(self, signal_id: str, t: float) -> float:
        """
        First instant at or after ``t`` at which a signal shows green.

        :param str signal_id: A signal of this section.
        :param float t: Absolute scenario time, in seconds.
        :returns: As ``SignalPlan.next_green``.
        :rtype: float
        :raises KeyError: If ``signal_id`` is not one of this section's signals.
        """
        return self._by_signal[signal_id].next_green(signal_id, t)

    def parallel_signal(self, crosswalk_id: str) -> str:
        """
        The signal a signalised crosswalk's walkers follow (``SignalPlan.parallel_signal``).

        :param str crosswalk_id: A crosswalk of this section.
        :returns: A signal of the phase whose traffic runs parallel to the crosswalk.
        :rtype: str
        :raises KeyError: If ``crosswalk_id`` is not one of this section's crosswalks.
        """
        return self._by_crosswalk[crosswalk_id].parallel_signal(crosswalk_id)

    @property
    def crosswalks(self) -> Tuple[str, ...]:
        """
        Signalised crosswalk IDs of this section: their walkers follow ``parallel_signal``.

        :returns: The IDs, sorted.
        :rtype: Tuple[str, ...]
        """
        return tuple(sorted(self._by_crosswalk))

    def validate(self, timings: Mapping[str, SignalTiming]) -> None:
        """
        Check that these plans are the Webster plans of ``timings``, one unit each, in unit order.

        :param Mapping[str, SignalTiming] timings: ``SignalTiming`` of each control unit, in unit order.
        :raises InvalidSignalPlan: If the plans do not cover the units of ``timings`` one to one,
            or a plan is invalid for its unit.
        """
        if [p.junction_id for p in self.plans] != list(timings):
            raise InvalidSignalPlan("plans do not match the map's control units")
        for plan in self.plans:
            timings[plan.junction_id].validate(plan)

    def is_valid(self, timings: Mapping[str, SignalTiming]) -> bool:
        """
        Whether ``validate`` accepts this section.

        :param Mapping[str, SignalTiming] timings: As in ``validate``.
        :returns: ``True`` if ``validate`` accepts this section; ``False`` if it raises ``InvalidSignalPlan``.
        :rtype: bool
        """
        try:
            self.validate(timings)
        except InvalidSignalPlan:
            return False
        return True

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        A JSON-compatible mapping of this section.

        :returns: ``{"plans": [...]}``, each plan from ``SignalPlan.to_dict``.
        :rtype: Dict[str, List[Dict[str, Any]]]
        """
        return {"plans": [p.to_dict() for p in self.plans]}

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, Any]]]) -> Self:
        """
        Build a section from ``to_dict`` output.

        :param Dict[str, List[Dict[str, Any]]] data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed section.
        :rtype: SignalSection
        :raises KeyError: If ``plans`` is absent, or a plan lacks a required field.
        :raises ValueError: If a rebuilt plan fails ``SignalPlan.__post_init__``.
        """
        return cls(tuple(SignalPlan.from_dict(p) for p in data["plans"]))
