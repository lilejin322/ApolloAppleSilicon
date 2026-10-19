"""
Speed-profile and start-time crossover and mutation shared by every replayed agent.

The operators read ``route``, ``speed_profile``, ``start_t``,
``spec.max_speed`` and ``lane_change_s``, and, given a route space,
``route_distance`` and ``route_point``; they change the profile or the start
time with ``dataclasses.replace``. NPC vehicles and bicycles both provide
these. Each returns a child that changes one gene, or None when it does not apply.
"""

from dataclasses import replace
from random import Random
from typing import Sequence, TypeVar

from ..scenario import InvalidRoute, RouteSpace, SpeedPoint
from ..scenario.replay import MAX_SPEED_POINTS
from .common import ReplayOperatorConfig, gauss_step

Replayed = TypeVar("Replayed")
"""An NPCVehicleAgent or BicycleAgent."""

__all__ = [
    "clamp_speeds", "cx_speed_profile", "cx_start_time", "mut_speed", "mut_speed_point", "mut_start_t", "mut_stop",
    "with_profile",
]


def clamp_speeds(profile: Sequence[SpeedPoint], max_speed: float) -> tuple[SpeedPoint, ...]:
    """The profile with every target speed at most ``max_speed``."""
    return tuple(replace(point, speed=min(point.speed, max_speed)) for point in profile)


def with_profile(agent: Replayed, profile: Sequence[SpeedPoint]) -> Replayed | None:
    """``agent`` with another speed profile, or None if the profile is malformed."""
    try:
        return replace(agent, speed_profile=tuple(profile))
    except ValueError:
        return None


def cx_speed_profile(first: Replayed, second: Replayed, rng: Random) -> Replayed | None:
    """
    Splice the speed profiles at a random cut along ``first``'s route:
    ``first``'s points before it, ``second``'s from it, clamped to ``first``'s
    maximum speed. Points are map positions, so only ``second``'s points on
    lanes of ``first``'s route can be taken; the cut is one of the two
    profiles' positions there, in route order (lane, then s).

    :returns: The child, or None if the splice equals ``first``'s profile or is
        malformed (more than four points, ending in a stop, or off the driven stretch).
    """
    order = {lane_id: index for index, lane_id in enumerate(first.route.lanes)}

    def key(point):
        return order[point.lane_id], point.s

    theirs = [point for point in second.speed_profile if point.lane_id in order]
    cuts = sorted({key(point) for point in first.speed_profile} | {key(point) for point in theirs})
    cut = rng.choice(cuts)
    profile = (tuple(point for point in first.speed_profile if key(point) < cut)
               + clamp_speeds([point for point in theirs if key(point) >= cut], first.spec.max_speed))
    return None if profile == first.speed_profile else with_profile(first, profile)


def cx_start_time(first: Replayed, second: Replayed) -> Replayed | None:
    """
    Give ``first`` the start time of ``second``.

    :returns: The child, or None if the start times are equal.
    """
    return None if first.start_t == second.start_t else replace(first, start_t=second.start_t)


def mut_speed(agent: Replayed, rng: Random, config: ReplayOperatorConfig) -> Replayed | None:
    """
    Move one nonzero target speed by a normal step (``config.sigma_speed``),
    within ``[config.min_speed, max_speed]``.

    :returns: The mutant, or None if every point is a stop.
    """
    profile = agent.speed_profile
    moving = [index for index, point in enumerate(profile) if point.speed > 0]
    if not moving:
        return None
    index = rng.choice(moving)
    max_speed = agent.spec.max_speed
    speed = min(max(profile[index].speed + rng.gauss(0, config.sigma_speed),
                    min(config.min_speed, max_speed)), max_speed)
    return with_profile(agent, profile[:index] + (replace(profile[index], speed=speed),) + profile[index + 1:])


def mut_speed_point(agent: Replayed, rng: Random, config: ReplayOperatorConfig,
                    space: RouteSpace | None = None) -> Replayed | None:
    """
    Add a speed point, remove one, or move one by a normal step of driven
    distance (``config.sigma_s`` metres) between its neighbours.

    Points are map positions (lane, s). With the agent's ``space``, an added
    point lies at a random driven distance along the route, and a moved point
    may cross onto the previous or next route lane. Without, an added point
    lies just after a random existing one on its lane and a moved point stays
    on its lane.

    :returns: The mutant, or None if the change would leave a malformed profile.
    """
    profile = list(agent.speed_profile)
    actions = (["add"] if len(profile) < MAX_SPEED_POINTS else []) + (["remove"] if len(profile) > 1 else []) + ["move"]
    action = rng.choice(actions)
    max_speed = agent.spec.max_speed
    if action == "remove":
        del profile[rng.randrange(len(profile))]
        return with_profile(agent, profile)
    try:
        marks = [agent.route_distance(space, point.lane_id, point.s) for point in profile] if space else None
    except InvalidRoute:
        return None
    if action == "add":
        speed = rng.uniform(min(config.min_speed, max_speed), max_speed)
        if marks is not None:
            distance = rng.uniform(0, space.length(agent.route, lane_change_s=agent.lane_change_s))
            if any(abs(distance - mark) < 1e-6 for mark in marks):
                return None
            lane_id, s = agent.route_point(space, distance)
            index = sum(mark < distance for mark in marks)
        else:
            index = rng.randrange(len(profile))
            lane_id, s = profile[index].lane_id, profile[index].s + abs(rng.gauss(0, config.sigma_s))
            index += 1
        profile.insert(index, SpeedPoint(lane_id, s, speed))
        return with_profile(agent, profile)
    index = rng.randrange(len(profile))
    step = rng.gauss(0, config.sigma_s)
    if marks is not None:
        distance = marks[index] + step
        low = marks[index - 1] if index else -1.0
        high = marks[index + 1] if index < len(marks) - 1 else float("inf")
        if not (low < distance < high and distance >= 0):
            return None
        lane_id, s = agent.route_point(space, distance)
    else:
        lane_id, s = profile[index].lane_id, profile[index].s + step
        if s < 0:
            return None
    profile[index] = replace(profile[index], lane_id=lane_id, s=s)
    return with_profile(agent, profile)


def mut_stop(agent: Replayed, rng: Random, config: ReplayOperatorConfig) -> Replayed | None:
    """
    Turn an inner moving point into a stop with a random dwell, turn a stop
    back into a moving point, or change a stop's dwell by a normal step (``config.sigma_t``).

    Inner points lie after the first point and before the last, which must keep
    moving. Points keep their map position.

    :returns: The mutant, or None if the profile has no inner point and no stop.
    """
    profile = agent.speed_profile
    stoppable = [index for index, point in enumerate(profile[:-1]) if point.speed > 0 and index > 0]
    stops = [index for index, point in enumerate(profile) if point.speed == 0]
    actions = (["stop"] if stoppable else []) + (["go", "dwell"] if stops else [])
    if not actions:
        return None
    action = rng.choice(actions)
    if action == "stop":
        index = rng.choice(stoppable)
        point = replace(profile[index], speed=0.0, dwell=rng.uniform(0, config.max_dwell))
    else:
        index = rng.choice(stops)
        if action == "go":
            max_speed = agent.spec.max_speed
            point = replace(profile[index], speed=rng.uniform(min(config.min_speed, max_speed), max_speed),
                            dwell=0.0)
        else:
            point = replace(profile[index],
                            dwell=gauss_step(rng, profile[index].dwell, config.sigma_t, config.max_dwell))
    return with_profile(agent, profile[:index] + (point,) + profile[index + 1:])


def mut_start_t(agent: Replayed, rng: Random, config: ReplayOperatorConfig) -> Replayed:
    """
    Move the start time by a normal step, within ``[0, config.max_start_t]``.

    :rtype: Replayed
    """
    return replace(agent, start_t=gauss_step(rng, agent.start_t, config.sigma_t, config.max_start_t))
