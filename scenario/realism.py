"""
Realistic distributions of replayed traffic genes, shared by generation and
the genetic operators so that variation keeps them (no drift).

Cruise speeds. A vehicle's target speed is drawn relative to the lane speed
limit where it applies (``reference_speed``): most drivers cruise at
0.7-1.0 of the limit, around a type-dependent mean (trucks and buses lower),
and a minority (``SLOW_SHARE``) is slower. Riders' targets are absolute
(bike-lane limits rarely bind). ``draw_speed`` samples this prior and
``speed_step`` is a Metropolis step with the same prior as its stationary
distribution, so mutation does not push speeds towards their bounds.

Driving style. ``drive_style`` in [0, 1] picks an agent's comfortable
acceleration and braking within type ranges (``COMFORT``):
replay no longer always accelerates at the type's hard limit. Its prior is
uniform; operators copy it or take reflected Gaussian steps, which keep a
uniform distribution uniform. ``JERK_TARGET`` sets how gradually the replay
ramps acceleration in and out.

These are plausible literature-range defaults, not calibrated to measured data.
"""

from dataclasses import replace
from math import erf, exp, pi, sqrt
from random import Random

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
}
JERK_TARGET = 2.5
"""m/s^3: the replay rounds acceleration changes over a distance that keeps
jerk near this at cruise speed."""
STYLE_SIGMA = 0.15
"""Step of drive-style mutations."""


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _usual(kind: str):
    if kind in RIDER_CRUISE:
        return RIDER_CRUISE[kind]
    return CRUISE_FACTOR[kind], CRUISE_SIGMA, CRUISE_RANGE


def _slow(kind: str):
    return RIDER_SLOW_RANGE[kind] if kind in RIDER_SLOW_RANGE else SLOW_RANGE


def reference_speed(kind: str, limit: float | None, max_speed: float) -> float:
    """
    What target speeds of ``kind`` are relative to: the lane limit (or
    ``DEFAULT_SPEED_LIMIT``) capped by the type's ``max_speed`` for vehicles, 1 for riders.
    """
    if kind in RIDER_CRUISE:
        return 1.0
    return min(limit if limit else DEFAULT_SPEED_LIMIT, max_speed)


def density(kind: str, speed: float, reference: float, low: float, high: float) -> float:
    """Unnormalized prior density of a target speed, zero outside [low, high]."""
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
    """A target speed from the prior restricted to [low, high] (``high`` if it cannot hold any)."""
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
    """
    proposal = speed + rng.gauss(0, sigma)
    new = density(kind, proposal, reference, low, high)
    if new <= 0:
        return None
    old = density(kind, speed, reference, low, high)
    if old > 0 and rng.random() >= new / old:
        return None
    return proposal


def comfort_limits(spec, kind: str, style: float):
    """``spec`` with the acceleration and comfortable braking of ``style``."""
    (a0, a1), (d0, d1) = COMFORT[kind]
    acceleration = min(a0 + (a1 - a0) * style, spec.max_acceleration)
    braking = min(d0 + (d1 - d0) * style, spec.max_deceleration)
    return replace(spec, max_acceleration=acceleration, comfortable_deceleration=braking)


def style_step(rng: Random, style: float, sigma: float = STYLE_SIGMA) -> float:
    """A reflected normal step within [0, 1]; keeps a uniform distribution uniform."""
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


def draw_type(rng: Random, weights) -> str:
    """A type name drawn with probability proportional to ``weights`` (name -> weight)."""
    names = sorted(name for name, weight in weights.items() if weight > 0)
    if not names:
        raise ValueError("type weights must have a positive entry")
    return rng.choices(names, weights=[weights[name] for name in names])[0]
