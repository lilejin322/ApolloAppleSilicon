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
from ..scenario.replay import MAX_SPEED_POINTS, point_reference
from ..scenario.realism import draw_speed, speed_step, style_step
from .common import ReplayOperatorConfig, gauss_step

Replayed = TypeVar("Replayed")
"""An NPCVehicleAgent or BicycleAgent."""

__all__ = [
    "clamp_speeds", "cx_drive_style", "cx_speed_profile", "mut_drive_style", "cx_start_time", "mut_speed", "mut_speed_point", "mut_start_t", "mut_stop",
    "with_profile",
]


def clamp_speeds(profile: Sequence[SpeedPoint], max_speed: float) -> tuple[SpeedPoint, ...]:
    """The profile with every target speed at most ``max_speed``."""
    return tuple(replace(point, speed=min(point.speed, max_speed)) for point in profile)


def kind_of(agent: Replayed) -> str:
    """The agent's vehicle or rider type name, which picks its realism priors."""
    vehicle_type = getattr(agent, "vehicle_type", None)
    return (vehicle_type if vehicle_type is not None else agent.rider_type).name


def with_profile(agent: Replayed, profile: Sequence[SpeedPoint]) -> Replayed | None:
    """``agent`` with another speed profile, or None if the profile is malformed."""
    try:
        return replace(agent, speed_profile=tuple(profile))
    except ValueError:
        return None


def cx_speed_profile(first: Replayed, second: Replayed, rng: Random, space: RouteSpace | None = None,
                     second_space: RouteSpace | None = None, min_speed: float = 0.0) -> Replayed | None:
    """
    Splice the speed profiles at a random cut along ``first``'s route:
    ``first``'s points before it, ``second``'s from it, clamped to ``first``'s
    maximum speed. The cut is one of the two profiles' positions, in route order.

    Points are map positions. Without a ``space`` only ``second``'s points on
    lanes of ``first``'s route can be taken. With ``first``'s ``space`` (and
    ``second_space`` when ``second`` drives another space), ``second``'s other
    points move to the same share of the driven distance on ``first``'s route,
    so a splice between different routes keeps as many points as its parents
    instead of thinning the profile generation after generation.

    :returns: The child, or None if the splice equals ``first``'s profile or is
        malformed (more than four points, ending in a stop, or off the driven stretch).
    """
    if space is None:
        order = {lane_id: index for index, lane_id in enumerate(first.route.lanes)}

        def key(point):
            return order[point.lane_id], point.s

        theirs = [point for point in second.speed_profile if point.lane_id in order]
        cuts = sorted({key(point) for point in first.speed_profile} | {key(point) for point in theirs})
        cut = rng.choice(cuts)
        profile = (tuple(point for point in first.speed_profile if key(point) < cut)
                   + clamp_speeds([point for point in theirs if key(point) >= cut], first.spec.max_speed))
        return None if profile == first.speed_profile else with_profile(first, profile)
    second_space = second_space or space
    try:
        total = space.length(first.route, lane_change_s=first.lane_change_s)
        other_total = second_space.length(second.route, lane_change_s=second.lane_change_s)
        mine = [(first.route_distance(space, point.lane_id, point.s), point) for point in first.speed_profile]
        theirs = []
        on_route = set(first.route.lanes)
        for point in second.speed_profile:
            if point.lane_id in on_route:
                distance = first.route_distance(space, point.lane_id, point.s)
                theirs.append((distance, point))
            else:
                share = second.route_distance(second_space, point.lane_id, point.s) / other_total if other_total > 0 else 0.0
                lane_id, s = first.route_point(space, share * total)
                speed = point.speed
                if speed > 0:
                    # Keep the point's cruise factor: its speed relative to the lane it moves to.
                    kind, top = kind_of(first), first.spec.max_speed
                    speed *= (point_reference(space, lane_id, kind, top)
                              / point_reference(second_space, point.lane_id, kind_of(second), second.spec.max_speed))
                    speed = max(speed, min_speed)  # a rescaled point stays a moving target
                theirs.append((first.route_distance(space, lane_id, s), replace(point, lane_id=lane_id, s=s, speed=speed)))
    except (InvalidRoute, KeyError):
        return None
    cut = rng.choice(sorted({d for d, _ in mine} | {d for d, _ in theirs}))
    placed = [(d, p) for d, p in mine if d < cut] + [(d, p) for d, p in sorted(theirs, key=lambda item: item[0]) if d >= cut]
    spare = [(d, p) for d, p in mine if d >= cut] + [(d, p) for d, p in theirs if d < cut]
    # The child keeps the lead parent's number of points: splices ran up to
    # eight points and were cut to four, so counts fell over generations
    # (-0.14 points per child). Excess points go at random (any of them: keeping
    # the first would leave a 1-point parent unchanged), missing ones come at
    # random from the parents' unused points.
    target = len(first.speed_profile)
    kept = _distinct(placed)
    while len(kept) > target:
        del kept[rng.randrange(len(kept))]
    while kept and kept[-1][1].speed == 0:  # a trailing stop is dropped below
        kept.pop()
    while len(kept) < target and spare:
        candidate = spare.pop(rng.randrange(len(spare)))
        kept = _distinct(kept + [candidate])
        while kept and kept[-1][1].speed == 0:
            kept.pop()
    if len(kept) != target:  # too few usable points: no count-keeping splice
        return None
    profile = tuple(p for _, p in kept[:MAX_SPEED_POINTS])
    profile = clamp_speeds(profile, first.spec.max_speed)
    while profile and profile[-1].speed == 0:
        profile = profile[:-1]
    if not profile or profile == first.speed_profile:
        return None
    return with_profile(first, profile)


def _distinct(placed: list) -> list:
    """(distance, point) pairs in driving order, one per distance."""
    kept = []
    for distance, point in sorted(placed, key=lambda item: item[0]):
        if kept and distance <= kept[-1][0] + 1e-6:
            continue
        kept.append((distance, point))
    return kept


def cx_start_time(first: Replayed, second: Replayed) -> Replayed | None:
    """
    Give ``first`` the start time of ``second``.

    :returns: The child, or None if the start times are equal.
    """
    return None if first.start_t == second.start_t else replace(first, start_t=second.start_t)


def mut_speed(agent: Replayed, rng: Random, config: ReplayOperatorConfig,
              space: RouteSpace | None = None) -> Replayed | None:
    """
    Move one nonzero target speed by a normal step (``config.sigma_speed``),
    within ``[config.min_speed, max_speed]``.

    With the agent's ``space`` the step is a Metropolis step under the
    realistic cruise prior of the point's lane (``realism``): a step the prior
    makes less likely may be rejected, so repeated mutation keeps cruise speeds
    distributed as at generation and does not push them to their bounds.

    :returns: The mutant, or None if every point is a stop or the step was rejected.
    """
    profile = agent.speed_profile
    moving = [index for index, point in enumerate(profile) if point.speed > 0]
    if not moving:
        return None
    index = rng.choice(moving)
    max_speed = agent.spec.max_speed
    low = min(config.min_speed, max_speed)
    if space is not None:
        kind = kind_of(agent)
        reference = point_reference(space, profile[index].lane_id, kind, max_speed)
        speed = speed_step(rng, kind, profile[index].speed, reference, low, max_speed, config.sigma_speed)
        if speed is None:
            return None
    else:
        speed = min(max(profile[index].speed + rng.gauss(0, config.sigma_speed), low), max_speed)
    return with_profile(agent, profile[:index] + (replace(profile[index], speed=speed),) + profile[index + 1:])


SPEED_POINT_ACTIONS = ("add", "remove", "move")
"""``mut_speed_point``'s actions, drawn with equal chance once per mutation event."""


def mut_speed_point(agent: Replayed, rng: Random, config: ReplayOperatorConfig,
                    space: RouteSpace | None = None, action: str | None = None) -> Replayed | None:
    """
    Add a speed point, remove one, or move one by a normal step of driven
    distance (``config.sigma_s`` metres) between its neighbours.

    Points are map positions (lane, s). With the agent's ``space``, an added
    point lies at a random driven distance along the route, and a moved point
    may cross onto the previous or next route lane. Without, an added point
    lies just after a random existing one on its lane and a moved point stays
    on its lane.

    ``action`` (one of ``SPEED_POINT_ACTIONS``; drawn with equal chance if None)
    is drawn once per mutation event by the agent's mutation and kept over
    retries; an action the profile does not allow (adding a fifth point,
    removing the only one) moves a point instead. Drawing among the allowed actions, and
    redrawing on retries, added points to 1-point profiles at +0.73 per mutant
    and removed them from 4-point ones at -0.51: point counts drifted from
    uniform over 1-4 towards 2-3. Equal add and remove chances at every count
    keep the generation-0 uniform counts.

    :returns: The mutant, or None if the change would leave a malformed profile.
    """
    profile = list(agent.speed_profile)
    action = rng.choice(SPEED_POINT_ACTIONS) if action is None else action
    if action not in SPEED_POINT_ACTIONS:
        raise ValueError(f"action must be one of {SPEED_POINT_ACTIONS}, got {action!r}")
    if (action == "add" and len(profile) >= MAX_SPEED_POINTS) or (action == "remove" and len(profile) <= 1):
        action = "move"
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
            kind = kind_of(agent)
            speed = draw_speed(rng, kind, point_reference(space, lane_id, kind, max_speed),
                               min(config.min_speed, max_speed), max_speed)
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


def cx_drive_style(first: Replayed, second: Replayed) -> Replayed | None:
    """
    Give ``first`` the drive style of ``second``.

    :returns: The child, or None if the styles are equal.
    """
    return None if first.drive_style == second.drive_style else replace(first, drive_style=second.drive_style)


def mut_drive_style(agent: Replayed, rng: Random) -> Replayed:
    """
    Step the drive style by a reflected normal step within [0, 1]
    (``realism.style_step``), which keeps its uniform prior.

    :rtype: Replayed
    """
    return replace(agent, drive_style=style_step(rng, agent.drive_style))
