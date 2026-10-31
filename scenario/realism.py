"""
Realistic distributions of replayed traffic genes, shared by generation and
the genetic operators so that variation keeps them (no drift).

Cruise speeds. A vehicle's target speed is drawn relative to the lane speed
limit where it applies (``reference_speed``): most drivers cruise at
0.7-1.0 of the limit, around a type-dependent mean (trucks and buses lower),
and a minority (``SLOW_SHARE``) is slower. Riders' and walkers' targets are
absolute (bike-lane limits rarely bind; walkers have none). ``draw_speed`` samples this prior and
``speed_step`` is a Metropolis step with the same prior as its stationary
distribution, so mutation does not push speeds towards their bounds.

Driving style. ``drive_style`` in [0, 1] picks an agent's comfortable
acceleration and braking within type ranges (``COMFORT``):
replay no longer always accelerates at the type's hard limit. Its prior is
uniform; operators copy it or take reflected Gaussian steps, which keep a
uniform distribution uniform. ``JERK_TARGET`` sets how gradually the replay
ramps acceleration in and out. A walker's ``walk_style`` plays the same part.

Mixes. ``VEHICLE_MIX``, ``RIDER_MIX`` and ``WALKER_MIX`` give the default
composition by type; ``EXCURSION_MIX`` and ``EXCURSION_COUNT_MIX`` how
pedestrians leave the sidewalk.

These are plausible literature-range defaults, not calibrated to measured data.
"""

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from random import Random
from typing import Mapping, TYPE_CHECKING, Tuple, TypeVar

if TYPE_CHECKING:
    from .bicycle import RiderSpec
    from .npc_agent import VehicleSpec
    from .pedestrian import WalkerSpec

ComfortSpec = TypeVar("ComfortSpec", "VehicleSpec", "RiderSpec", "WalkerSpec")
"""Motion specification types whose hard limits ``comfort_limits`` reads."""

DEFAULT_SPEED_LIMIT = 13.4
"""m/s (30 mph) assumed where a lane has no speed limit."""

CRUISE_FACTOR = {"CAR": 0.92, "VAN": 0.9, "TRUCK": 0.85, "BUS": 0.83}
"""Mean target speed as a share of the reference speed, by vehicle type."""
CRUISE_SIGMA = 0.08
CRUISE_RANGE = (0.7, 1.0)
"""The usual cruise factors: a normal truncated to this range."""
SLOW_SHARE = 0.1
SLOW_RANGE = (0.35, 0.7)
"""Share of slower drivers and their (uniform) cruise factors."""

RIDER_CRUISE = {"BICYCLE": (4.5, 0.8, (2.5, 7.0)), "E_BIKE": (6.0, 1.0, (3.5, 9.5))}
"""Riders' target speeds in m/s: mean, sigma and range of the usual component."""
RIDER_SLOW_RANGE = {"BICYCLE": (1.5, 2.5), "E_BIKE": (2.0, 3.5)}

WALKER_CRUISE = {
    "ADULT": (1.45, 0.25, (0.9, 2.2)),
    "ELDERLY": (1.2, 0.25, (0.6, 1.8)),
    "CHILD": (1.45, 0.3, (0.8, 2.3)),
    "RUNNER": (3.0, 0.5, (2.0, 4.5)),
}
"""Walkers' target speeds in m/s: mean, sigma and range of the usual component.
Crosswalk observations: 15th-percentile 1.25 m/s under 65 and 0.97 m/s over 65,
means about 1.5 and 1.25 m/s (Knoblauch, Pietrucha and Nitzburg, 1996, TRR 1538);
15th-percentile 1.15 and 0.92 m/s at 3,000 crossings, of which 19% ran or walked
and ran (TCRP Report 112 / NCHRP Report 562, 2006). School children walk at least
as fast as the guideline speeds (Chang et al., 2018, TRR 2672)."""
WALKER_SLOW_RANGE = {"ADULT": (0.6, 0.9), "ELDERLY": (0.4, 0.6), "CHILD": (0.6, 0.8), "RUNNER": (1.8, 2.0)}
"""The slow component: walkers with children, luggage or a phone, walking aids, joggers easing off."""

ABSOLUTE_CRUISE = {**RIDER_CRUISE, **WALKER_CRUISE}
"""Types whose target speeds are absolute rather than shares of the lane limit."""
_ABSOLUTE_SLOW_RANGE = {**RIDER_SLOW_RANGE, **WALKER_SLOW_RANGE}

COMFORT = {
    # (acceleration, braking) ranges in m/s^2, from the gentlest (drive_style 0)
    # to the briskest (1) ordinary driver. Cornering keeps the type's lateral
    # limit: urban turns are routinely driven at 2-3 m/s^2.
    "CAR": ((1.0, 2.5), (1.5, 3.0)),
    "VAN": ((0.9, 2.0), (1.4, 2.7)),
    "TRUCK": ((0.5, 1.2), (1.0, 2.0)),
    "BUS": ((0.6, 1.2), (1.0, 2.0)),
    "BICYCLE": ((0.5, 1.0), (0.8, 1.5)),
    "E_BIKE": ((0.7, 1.4), (1.0, 2.0)),
    # Walkers: video-tracked crossing pedestrians accelerated 0.68 m/s^2 on average,
    # 1.44 m/s^2 at most ("Microscopic Pedestrian Flow Characteristics",
    # arXiv:1610.00029); runners start harder.
    "ADULT": ((0.5, 1.4), (0.5, 1.5)),
    "ELDERLY": ((0.3, 0.8), (0.3, 1.0)),
    "CHILD": ((0.6, 1.8), (0.6, 1.8)),
    "RUNNER": ((1.0, 2.5), (1.0, 2.5)),
}
JERK_TARGET = 2.5
"""m/s^3: the replay rounds acceleration changes over a distance that keeps
jerk near this at cruise speed."""
STYLE_SIGMA = 0.15
"""Step of drive-style mutations."""


def _normal_cdf(x: float) -> float:
    """
    Evaluate the standard normal cumulative distribution function.

    :param float x: Standard-normal variate at which to evaluate the CDF.
    :returns: The standard normal cumulative probability at ``x``.
    :rtype: float
    """
    return 0.5 * (1 + erf(x / sqrt(2)))


def _usual(kind: str) -> Tuple[float, float, Tuple[float, float]]:
    """
    Return the mean, standard deviation, and bounds of the usual cruise component.

    :param str kind: Agent type name selecting the applicable speed prior.
    :returns: Mean, standard deviation, and lower/upper bounds of the usual component.
    :rtype: Tuple[float, float, Tuple[float, float]]
    """
    if kind in ABSOLUTE_CRUISE:
        return ABSOLUTE_CRUISE[kind]
    return CRUISE_FACTOR[kind], CRUISE_SIGMA, CRUISE_RANGE


def _slow(kind: str) -> Tuple[float, float]:
    """
    Return the lower and upper bounds of the slow cruise component.

    :param str kind: Agent type name selecting the applicable speed prior.
    :returns: The slow component lower and upper bounds.
    :rtype: Tuple[float, float]
    """
    return _ABSOLUTE_SLOW_RANGE[kind] if kind in _ABSOLUTE_SLOW_RANGE else SLOW_RANGE


def reference_speed(kind: str, limit: float | None, max_speed: float) -> float:
    """
    What target speeds of ``kind`` are relative to: the lane limit (or
    ``DEFAULT_SPEED_LIMIT``) capped by the type's ``max_speed`` for vehicles,
    1 for riders and walkers (``ABSOLUTE_CRUISE``).

    :param str kind: Agent type name selecting the applicable speed prior.
    :param float | None limit: Lane speed limit in m/s, or ``None`` if unspecified.
    :param float max_speed: Maximum allowed target speed, in m/s.
    :returns: The capped lane reference speed for vehicles, or 1 for absolute speed priors.
    :rtype: float
    """
    if kind in ABSOLUTE_CRUISE:
        return 1.0
    return min(limit if limit else DEFAULT_SPEED_LIMIT, max_speed)


def density(kind: str, speed: float, reference: float, low: float, high: float) -> float:
    """
    Unnormalized prior density of a target speed, zero outside [low, high].

    :param str kind: Agent type name selecting the applicable speed prior.
    :param float speed: Target speed, in m/s.
    :param float reference: Reference speed scaling the cruise prior, in m/s; absolute priors use 1.
    :param float low: Lowest allowed target speed, in m/s.
    :param float high: Highest allowed target speed, in m/s.
    :returns: The unnormalized mixture density, or zero outside the allowed support.
    :rtype: float
    """
    if not low <= speed <= high or reference <= 0:
        return 0.0
    factor = speed / reference
    mean, sigma, (a, b) = _usual(kind)
    usual = 0.0
    if a <= factor <= b:
        mass = _normal_cdf((b - mean) / sigma) - _normal_cdf((a - mean) / sigma)
        usual = exp(-0.5 * ((factor - mean) / sigma) ** 2) / (sigma * sqrt(2 * pi) * mass)
    c, d = _slow(kind)
    slow = 1 / (d - c) if c <= factor <= d else 0.0
    return (1 - SLOW_SHARE) * usual + SLOW_SHARE * slow


def draw_speed(rng: Random, kind: str, reference: float, low: float, high: float, tries: int = 200) -> float:
    """
    A target speed from the prior restricted to [low, high] (``high`` if it cannot hold any).

    :param Random rng: Random number generator used for sampling and selection.
    :param str kind: Agent type name selecting the applicable speed prior.
    :param float reference: Reference speed scaling the cruise prior, in m/s; absolute priors use 1.
    :param float low: Minimum allowed target speed, in m/s.
    :param float high: Maximum allowed target speed, in m/s.
    :param int tries: Maximum number of speed draws before using the bounded fallback.
    :returns: A sampled speed in m/s, or a bounded mean if the attempts are exhausted.
    :rtype: float
    """
    mean, sigma, (a, b) = _usual(kind)
    c, d = _slow(kind)
    for _ in range(tries):
        if rng.random() < SLOW_SHARE:
            factor = rng.uniform(c, d)
        else:
            # Truncate within the usual component: re-tossing the component
            # here would inflate the slow share beyond ``SLOW_SHARE``.
            factor = rng.gauss(mean, sigma)
            while not a <= factor <= b:
                factor = rng.gauss(mean, sigma)
        speed = factor * reference
        if low <= speed <= high:
            return speed
    return min(max(mean * reference, low), high)


def speed_step(rng: Random, kind: str, speed: float, reference: float, low: float, high: float,
               sigma: float) -> float | None:
    """
    A Metropolis step from ``speed``: a normal proposal, accepted with the prior
    density ratio. None when rejected. Repeated steps keep the prior.

    :param Random rng: Random number generator used for sampling and selection.
    :param str kind: Agent type name selecting the applicable speed prior.
    :param float speed: Target speed, in m/s.
    :param float reference: Reference speed scaling the cruise prior, in m/s; absolute priors use 1.
    :param float low: Minimum allowed target speed, in m/s.
    :param float high: Maximum allowed target speed, in m/s.
    :param float sigma: Standard deviation of the speed proposal, in m/s.
    :returns: The proposed speed in m/s, or ``None`` if the Metropolis step is rejected.
    :rtype: float | None
    """
    proposal = speed + rng.gauss(0, sigma)
    new = density(kind, proposal, reference, low, high)
    if new <= 0:
        return None
    old = density(kind, speed, reference, low, high)
    if old > 0 and rng.random() >= new / old:
        return None
    return proposal


@dataclass(frozen=True)
class ComfortLimits:
    """
    The motion limits one agent replays with (``replay.MotionLimits``): its type's hard limits,
    with the acceleration and comfortable braking of its style (``comfort_limits``).

    :param float max_speed: m/s.
    :param float max_acceleration: The style's acceleration, m/s^2.
    :param float max_deceleration: The type's hard braking limit, m/s^2.
    :param float max_lateral_acceleration: m/s^2.
    :param float comfortable_deceleration: The style's braking for stops, curves, limits and the end, m/s^2.
    :param float min_turn_radius: Metres.
    """
    max_speed: float
    """m/s."""
    max_acceleration: float
    """The style's acceleration, m/s^2."""
    max_deceleration: float
    """The type's hard braking limit, m/s^2."""
    max_lateral_acceleration: float
    """m/s^2."""
    comfortable_deceleration: float
    """The style's braking for stops, curves, limits and the end, m/s^2."""
    min_turn_radius: float
    """Metres."""


def comfort_limits(spec: ComfortSpec, kind: str, style: float) -> ComfortLimits:
    """
    The limits an agent of ``spec``'s type replays with at ``style``: acceleration and comfortable
    braking from ``COMFORT[kind]`` (the only source of comfortable braking), capped by the
    type's hard limits.

    :param ComfortSpec spec: The type's hard limits.
    :param str kind: Agent type name selecting the comfortable motion limits.
    :param float style: Style value in [0, 1] selecting comfortable acceleration and braking.
    :returns: The agent's motion limits.
    :rtype: ComfortLimits
    """
    (a0, a1), (d0, d1) = COMFORT[kind]
    return ComfortLimits(spec.max_speed, min(a0 + (a1 - a0) * style, spec.max_acceleration), spec.max_deceleration,
                         spec.max_lateral_acceleration, min(d0 + (d1 - d0) * style, spec.max_deceleration),
                         spec.min_turn_radius)


def style_step(rng: Random, style: float, sigma: float = STYLE_SIGMA) -> float:
    """
    A reflected normal step within [0, 1]; keeps a uniform distribution uniform.

    :param Random rng: Random number generator used for sampling and selection.
    :param float style: Style value in [0, 1] selecting comfortable acceleration and braking.
    :param float sigma: Standard deviation of the dimensionless style proposal.
    :returns: The proposed style reflected into [0, 1].
    :rtype: float
    """
    value = style + rng.gauss(0, sigma)
    while not 0 <= value <= 1:
        value = -value if value < 0 else 2 - value
    return value


VEHICLE_MIX = {"CAR": 0.721, "VAN": 0.197, "TRUCK": 0.077, "BUS": 0.005}
"""Default traffic composition by vehicle type: shares of urban vehicle-miles in
FHWA Highway Statistics 2024, Table VM-1, "All Urban" (motorcycles left out):
light-duty short wheelbase 1,624,415 -> CAR; light-duty long wheelbase 444,471
(large cars, vans, pickups, SUVs) -> VAN; single-unit 85,467 + combination
trucks 87,048 -> TRUCK; buses 11,995 -> BUS (millions of vehicle-miles).
https://www.fhwa.dot.gov/policyinformation/statistics/2024/vm1.cfm"""

RIDER_MIX = {"BICYCLE": 0.83, "E_BIKE": 0.17}
"""Default rider composition: about 17% of people biking were on e-bikes in
Portland's 2023 citywide counts (PBOT, 2023 Portland Bicycle Counts, Table 6).
https://www.portland.gov/transportation/walking-biking-transit-safety/documents/2023-portland-bicycle-count-report/download"""


WALKER_MIX = {"ADULT": 0.72, "ELDERLY": 0.12, "CHILD": 0.06, "RUNNER": 0.10}
"""Default walker composition; a plausible default, not calibrated. Runners stand
for the crossers TCRP Report 112 saw running (part of its 19% not walking)."""

EXCURSION_MIX = {"CROSSWALK": 0.75, "NEAR_CROSSWALK": 0.08, "MID_BLOCK": 0.08, "DIAGONAL": 0.03,
                 "ROADSIDE": 0.06}
"""Default share of each way of leaving the sidewalk; plausible, not calibrated.
89% of crossings observed at 20 US sites were in marked intersection crosswalks
(FHWA-HRT-13-098, Human Factors Assessment of Pedestrian Roadway Crossing Behavior);
at 47 Utah signalized crosswalks 2-3% crossed mid-block and 6-8% mostly outside the
markings (UDOT, arXiv:2408.08890)."""

EXCURSION_COUNT_MIX = {0: 0.15, 1: 0.65, 2: 0.20}
"""Default share of walks by how often they leave the sidewalk; plausible, not calibrated."""


def draw_type(rng: Random, weights: Mapping[str, float]) -> str:
    """
    A type name drawn with probability proportional to ``weights`` (name -> weight).

    :param Random rng: Random number generator used for sampling and selection.
    :param Mapping[str, float] weights: Relative nonnegative sampling weights keyed by type name.
    :returns: A positively weighted type name.
    :rtype: str
    :raises ValueError: If ``weights`` has no positive entry.
    """
    names = sorted(name for name, weight in weights.items() if weight > 0)
    if not names:
        raise ValueError("type weights must have a positive entry")
    return rng.choices(names, weights=[weights[name] for name in names])[0]


def _prior_cdf(kind: str, value: float) -> float:
    """
    CDF of ``kind``'s absolute target-speed prior (``draw_speed`` with reference 1, unrestricted).

    :param str kind: Agent type name selecting the applicable speed prior.
    :param float value: Target speed whose prior quantile is evaluated, in m/s.
    :returns: The cumulative probability under the source type speed prior.
    :rtype: float
    """
    mean, sigma, (a, b) = _usual(kind)
    c, d = _slow(kind)
    slow = min(max((value - c) / (d - c), 0.0), 1.0) if d > c else float(value >= c)
    za, zb = _normal_cdf((a - mean) / sigma), _normal_cdf((b - mean) / sigma)
    usual = min(max((_normal_cdf((value - mean) / sigma) - za) / (zb - za), 0.0), 1.0)
    return SLOW_SHARE * slow + (1 - SLOW_SHARE) * usual


def mapped_speed(kind: str, other: str, value: float) -> float:
    """
    The speed of type ``other`` at the same quantile of its prior as ``value``
    has under ``kind``'s (absolute priors, ``ABSOLUTE_CRUISE``): a brisk adult
    becomes a brisk runner, not a runner at walking pace. Values beyond the
    prior's support scale with its nearest end (a crossing sped up past the range).

    :param str kind: Source agent type name.
    :param str other: Destination agent type name.
    :param float value: Positive target speed of the source type, in m/s.
    :returns: The mapped destination-type target speed, in m/s.
    :rtype: float
    """
    low = min(_slow(kind)[0], _usual(kind)[2][0])
    high = max(_slow(kind)[1], _usual(kind)[2][1])
    new_low = min(_slow(other)[0], _usual(other)[2][0])
    new_high = max(_slow(other)[1], _usual(other)[2][1])
    if value >= high:
        return new_high * value / high
    if value <= low:
        return new_low * value / low
    quantile = _prior_cdf(kind, value)
    lo, hi = new_low, new_high
    for _ in range(60):
        middle = (lo + hi) / 2
        if _prior_cdf(other, middle) < quantile:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2
