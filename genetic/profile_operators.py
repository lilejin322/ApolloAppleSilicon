"""
Speed-profile, style, and start-time operators shared by replayed agents.

Operators return a replacement agent or ``None`` when no applicable or valid
proposal is produced. Some proposals can equal their input; callers such as
``vary`` enforce an actual change and check route and timing constraints.

Agents provide the attributes used by each operator, such as ``speed_profile``,
``spec.max_speed``, ``start_t``, or ``drive_style``. Lane-aware profile operators
also use ``route``, ``lane_change_s``, ``route_distance``, and ``route_point``.
Pedestrian wrappers supply compatible walk-network distance and point mapping.
"""

from dataclasses import replace
from random import Random
from typing import Dict, List, Sequence, Set, Tuple, TypeVar

from ..scenario import InvalidRoute, RouteSpace, SpeedPoint, RiderType, VehicleType, WalkerType
from ..scenario.replay import MAX_SPEED_POINTS, point_reference
from ..scenario.realism import draw_speed, speed_step, style_step
from .common import ReplayOperatorConfig, gauss_step

Replayed = TypeVar("Replayed")
"""An NPCVehicleAgent, BicycleAgent or PedestrianAgent."""

__all__: List[str] = [
    "clamp_speeds", "cx_drive_style", "cx_speed_profile", "mut_drive_style", "cx_start_time", "mut_speed", "mut_speed_point", "mut_start_t", "mut_stop",
    "with_profile",
]


def clamp_speeds(profile: Sequence[SpeedPoint], max_speed: float) -> Tuple[SpeedPoint, ...]:
    """
    Cap every target speed in the profile at ``max_speed``.

    :param Sequence[SpeedPoint] profile: Speed points whose target speeds are capped.
    :param float max_speed: Maximum target speed, in m/s.
    :returns: A tuple of updated speed points with all other point attributes preserved.
    :rtype: Tuple[SpeedPoint, ...]
    """
    return tuple(replace(point, speed=min(point.speed, max_speed)) for point in profile)


def kind_of(agent: Replayed) -> str:
    """
    Return the agent's vehicle, rider, or walker type name for selecting realism priors.

    Inspect ``vehicle_type``, ``rider_type``, then ``walker_type``, using the first
    attribute whose value is not ``None``.

    :param Replayed agent: Agent whose type name is requested.
    :returns: The name of the first available vehicle, rider, or walker type.
    :rtype: str
    :raises TypeError: If the agent has no non-``None`` vehicle, rider, or walker type.
    """
    for attribute in ("vehicle_type", "rider_type", "walker_type"):
        kind: VehicleType | RiderType | WalkerType | None = getattr(agent, attribute, None)
        if kind is not None:
            return kind.name
    raise TypeError(f"{type(agent).__name__} has no vehicle, rider or walker type")


def with_profile(agent: Replayed, profile: Sequence[SpeedPoint]) -> Replayed | None:
    """
    Put another speed profile on an agent, preserving its other attributes.

    :param Replayed agent: Agent receiving the replacement profile.
    :param Sequence[SpeedPoint] profile: Proposed speed points, converted to a tuple.
    :returns: The updated agent, or ``None`` if construction raises ``ValueError``.
    :rtype: Replayed | None
    """
    try:
        return replace(agent, speed_profile=tuple(profile))
    except ValueError:
        return None


def cx_speed_profile(first: Replayed, second: Replayed, rng: Random, space: RouteSpace | None = None,
                     second_space: RouteSpace | None = None, min_speed: float = 0.0) -> Replayed | None:
    """
    Splice the parents' speed profiles at a randomly selected point position.

    Keep ``first``'s points before the cut and ``second``'s points from it onwards,
    with target speeds capped at ``first``'s maximum.

    Without ``space``, only donor points on lanes of ``first``'s route are used,
    and points are ordered by lane and position. With ``space``, donor points on
    other lanes are mapped to the same fraction of driven distance on ``first``'s
    route. Their positive speeds are scaled by the destination/source reference
    speed ratio and floored at ``min_speed`` before the final speed cap.

    With ``space``, duplicate distances and trailing stops are removed. Excess
    points are removed randomly and missing points are drawn from unused parental
    points to preserve ``first``'s point count; a splice that cannot do so fails.

    :param Replayed first: Lead parent providing the route, speed limit, and profile prefix.
    :param Replayed second: Donor parent providing the profile suffix.
    :param Random rng: Random number generator used for cut selection and point-count repair.
    :param RouteSpace | None space: Lead parent's route space for driven-distance mapping;
        ``None`` uses lane order without distance mapping or point-count repair.
    :param RouteSpace | None second_space: Donor parent's route space; defaults to ``space``.
    :param float min_speed: Minimum positive speed for donor points mapped to other lanes, in m/s.
    :returns: The child, or ``None`` if route mapping fails, the required point count
        cannot be preserved, the profile is unchanged or empty, or construction rejects it.
    :rtype: Replayed | None
    """
    if space is None:
        order: Dict[str, int] = {lane_id: index for index, lane_id in enumerate(first.route.lanes)}

        def key(point: SpeedPoint) -> Tuple[int, float]:
            """
            Return a speed point's position in the lead parent's lane order.

            :param SpeedPoint point: Point on a lane present in the lead parent's route.
            :returns: The route lane index and position along that lane, in metres.
            :rtype: Tuple[int, float]
            :raises KeyError: If the point's lane is absent from the lead parent's route.
            """
            return order[point.lane_id], point.s

        matching: List[SpeedPoint] = [point for point in second.speed_profile if point.lane_id in order]
        cuts: List[Tuple[int, float]] = sorted({key(point) for point in first.speed_profile} | {key(point) for point in matching})
        lane_cut: Tuple[int, float] = rng.choice(cuts)
        profile: Tuple[SpeedPoint, ...] = (tuple(point for point in first.speed_profile if key(point) < lane_cut)
                   + clamp_speeds([point for point in matching if key(point) >= lane_cut], first.spec.max_speed))
        return None if profile == first.speed_profile else with_profile(first, profile)
    second_space = second_space or space
    try:
        total: float = space.length(first.route, lane_change_s=first.lane_change_s)
        other_total: float = second_space.length(second.route, lane_change_s=second.lane_change_s)
        mine: List[Tuple[float, SpeedPoint]] = [(first.route_distance(space, point.lane_id, point.s), point) for point in first.speed_profile]
        theirs: List[Tuple[float, SpeedPoint]] = []
        on_route: Set[str] = set(first.route.lanes)
        for point in second.speed_profile:
            if point.lane_id in on_route:
                distance: float = first.route_distance(space, point.lane_id, point.s)
                theirs.append((distance, point))
            else:
                share: float = second.route_distance(second_space, point.lane_id, point.s) / other_total if other_total > 0 else 0.0
                lane_id: str
                s: float
                lane_id, s = first.route_point(space, share * total)
                speed: float = point.speed
                if speed > 0:
                    # Keep the point's cruise factor: its speed relative to the lane it moves to.
                    kind: str
                    top: float
                    kind, top = kind_of(first), first.spec.max_speed
                    speed *= (point_reference(space, lane_id, kind, top)
                              / point_reference(second_space, point.lane_id, kind_of(second), second.spec.max_speed))
                    speed = max(speed, min_speed)  # a rescaled point stays a moving target
                theirs.append((first.route_distance(space, lane_id, s), replace(point, lane_id=lane_id, s=s, speed=speed)))
    except (InvalidRoute, KeyError):
        return None
    cut: float = rng.choice(sorted({d for d, _ in mine} | {d for d, _ in theirs}))
    placed: List[Tuple[float, SpeedPoint]] = [(d, p) for d, p in mine if d < cut] + [(d, p) for d, p in sorted(theirs, key=lambda item: item[0]) if d >= cut]
    spare: List[Tuple[float, SpeedPoint]] = [(d, p) for d, p in mine if d >= cut] + [(d, p) for d, p in theirs if d < cut]
    # The child keeps the lead parent's number of points: splices ran up to
    # eight points and were cut to four, so counts fell over generations
    # (-0.14 points per child). Excess points go at random (any of them: keeping
    # the first would leave a 1-point parent unchanged), missing ones come at
    # random from the parents' unused points.
    target: int = len(first.speed_profile)
    kept: List[Tuple[float, SpeedPoint]] = _distinct(placed)
    while len(kept) > target:
        del kept[rng.randrange(len(kept))]
    while kept and kept[-1][1].speed == 0:  # a trailing stop is dropped below
        kept.pop()
    while len(kept) < target and spare:
        candidate: Tuple[float, SpeedPoint] = spare.pop(rng.randrange(len(spare)))
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


def _distinct(placed: List[Tuple[float, SpeedPoint]]) -> List[Tuple[float, SpeedPoint]]:
    """
    Sort distance/point pairs in driving order and discard near-duplicate distances.

    Keep the first pair at each distance, treating distances within ``1e-6``
    metres of the last retained distance as duplicates. Equal-distance pairs
    retain their input order before filtering.

    :param List[Tuple[float, SpeedPoint]] placed: Pairs of driven distance in metres and the corresponding speed point.
    :returns: A new list of retained distance/point pairs in ascending distance order.
    :rtype: List[Tuple[float, SpeedPoint]]
    """
    kept: List[Tuple[float, SpeedPoint]] = []
    for distance, point in sorted(placed, key=lambda item: item[0]):
        if kept and distance <= kept[-1][0] + 1e-6:
            continue
        kept.append((distance, point))
    return kept


def cx_start_time(first: Replayed, second: Replayed) -> Replayed | None:
    """
    Give ``first`` the start time of ``second``.

    :param Replayed first: Parent receiving the start time.
    :param Replayed second: Parent providing the start time.
    :returns: The child, or ``None`` if the start times are equal.
    :rtype: Replayed | None
    """
    return None if first.start_t == second.start_t else replace(first, start_t=second.start_t)


def mut_speed(agent: Replayed, rng: Random, config: ReplayOperatorConfig,
              space: RouteSpace | None = None) -> Replayed | None:
    """
    Move one positive target speed, preserving its point's position and dwell.

    Without ``space``, apply a normal step using ``config.sigma_speed`` and clamp
    it between ``min(config.min_speed, max_speed)`` and the agent's maximum speed.
    With ``space``, use ``realism.speed_step`` under the point's lane cruise prior;
    a proposal made less likely by the prior may be rejected.

    :param Replayed agent: Agent whose target speed is mutated.
    :param Random rng: Random number generator used to select a moving point and sample its speed.
    :param ReplayOperatorConfig config: Configuration defining minimum speed and speed step size.
    :param RouteSpace | None space: Optional route space supplying lane-specific speed references.
    :returns: The mutant, or ``None`` if no point has positive speed, the speed step
        is rejected, or the replacement profile is malformed.
    :rtype: Replayed | None
    :raises TypeError: If lane-aware mutation cannot determine the agent's realism type.
    """
    profile: Tuple[SpeedPoint, ...] = agent.speed_profile
    moving: List[int] = [index for index, point in enumerate(profile) if point.speed > 0]
    if not moving:
        return None
    index: int = rng.choice(moving)
    max_speed: float = agent.spec.max_speed
    low: float = min(config.min_speed, max_speed)
    if space is not None:
        kind: str = kind_of(agent)
        reference: float = point_reference(space, profile[index].lane_id, kind, max_speed)
        speed: float | None = speed_step(rng, kind, profile[index].speed, reference, low, max_speed, config.sigma_speed)
        if speed is None:
            return None
    else:
        speed = min(max(profile[index].speed + rng.gauss(0, config.sigma_speed), low), max_speed)
    return with_profile(agent, profile[:index] + (replace(profile[index], speed=speed),) + profile[index + 1:])


SPEED_POINT_ACTIONS: Tuple[str, ...] = ("add", "remove", "move")
"""``mut_speed_point``'s actions, drawn with equal chance once per mutation event."""


def mut_speed_point(agent: Replayed, rng: Random, config: ReplayOperatorConfig,
                    space: RouteSpace | None = None, action: str | None = None) -> Replayed | None:
    """
    Add or remove a speed point, or move one by a normal position step.

    With ``space``, additions sample a driven distance along the route and draw a
    lane-aware speed. Moves use ``config.sigma_s`` metres and stay between adjacent
    points, possibly crossing route lane boundaries. Without ``space``, additions
    are placed after a random existing point on its lane, and moves stay on their
    lane; malformed profiles are rejected during agent construction.

    Draw ``action`` with equal probability once per mutation event and reuse it
    across retries. Adding at ``MAX_SPEED_POINTS`` or removing the only point
    falls back to moving a point. Choosing only applicable actions previously
    pushed point counts away from the generation distribution towards 2-3 points.

    :param Replayed agent: Agent whose speed-point count or positions are mutated.
    :param Random rng: Random number generator used for unspecified action selection,
        point selection, and position and speed sampling.
    :param ReplayOperatorConfig config: Configuration defining position step size and minimum speed.
    :param RouteSpace | None space: Optional route space used for driven-distance placement and speed priors.
    :param str | None action: Action from ``SPEED_POINT_ACTIONS``, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if initial distance mapping fails, a sampled
        position violates spacing or movement bounds, or the replacement profile is malformed.
    :rtype: Replayed | None
    :raises ValueError: If ``action`` is not in ``SPEED_POINT_ACTIONS``.
    :raises InvalidRoute: If later route-length or point-placement operations fail
        after the initial distance mapping.
    :raises TypeError: If lane-aware addition cannot determine the agent's realism type.
    """
    profile: List[SpeedPoint] = list(agent.speed_profile)
    action = rng.choice(SPEED_POINT_ACTIONS) if action is None else action
    if action not in SPEED_POINT_ACTIONS:
        raise ValueError(f"action must be one of {SPEED_POINT_ACTIONS}, got {action!r}")
    if (action == "add" and len(profile) >= MAX_SPEED_POINTS) or (action == "remove" and len(profile) <= 1):
        action = "move"
    max_speed: float = agent.spec.max_speed
    if action == "remove":
        del profile[rng.randrange(len(profile))]
        return with_profile(agent, profile)
    try:
        marks: List[float] | None = [agent.route_distance(space, point.lane_id, point.s) for point in profile] if space else None
    except InvalidRoute:
        return None
    if action == "add":
        speed: float = rng.uniform(min(config.min_speed, max_speed), max_speed)
        if marks is not None:
            distance: float = rng.uniform(0, space.length(agent.route, lane_change_s=agent.lane_change_s))
            if any(abs(distance - mark) < 1e-6 for mark in marks):
                return None
            lane_id: str
            s: float
            lane_id, s = agent.route_point(space, distance)
            kind: str = kind_of(agent)
            speed = draw_speed(rng, kind, point_reference(space, lane_id, kind, max_speed),
                               min(config.min_speed, max_speed), max_speed)
            index: int = sum(mark < distance for mark in marks)
        else:
            index = rng.randrange(len(profile))
            lane_id, s = profile[index].lane_id, profile[index].s + abs(rng.gauss(0, config.sigma_s))
            index += 1
        profile.insert(index, SpeedPoint(lane_id, s, speed))
        return with_profile(agent, profile)
    index = rng.randrange(len(profile))
    step: float = rng.gauss(0, config.sigma_s)
    if marks is not None:
        distance = marks[index] + step
        low: float = marks[index - 1] if index else -1.0
        high: float = marks[index + 1] if index < len(marks) - 1 else float("inf")
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
    Turn an inner moving point into a stop, resume a stop, or step its dwell.

    Choose uniformly among applicable actions. A new stop gets a uniform dwell
    in ``[0, config.max_dwell]``; a resumed stop gets a uniform speed between
    ``min(config.min_speed, max_speed)`` and the agent's maximum, with zero dwell.
    Dwell steps use ``config.sigma_t`` and are clamped to the dwell bounds.
    Only moving points between the first and last can become stops; all points
    keep their map positions.

    :param Replayed agent: Agent whose stops or dwell times are mutated.
    :param Random rng: Random number generator used to select an action, point, and mutation value.
    :param ReplayOperatorConfig config: Configuration defining speed bounds, dwell bounds, and dwell step size.
    :returns: The mutant, or ``None`` if there is no inner moving point or existing
        stop, or the replacement profile is malformed.
    :rtype: Replayed | None
    """
    profile: Tuple[SpeedPoint, ...] = agent.speed_profile
    stoppable: List[int] = [index for index, point in enumerate(profile[:-1]) if point.speed > 0 and index > 0]
    stops: List[int] = [index for index, point in enumerate(profile) if point.speed == 0]
    actions: List[str] = (["stop"] if stoppable else []) + (["go", "dwell"] if stops else [])
    if not actions:
        return None
    action: str = rng.choice(actions)
    if action == "stop":
        index: int = rng.choice(stoppable)
        point: SpeedPoint = replace(profile[index], speed=0.0, dwell=rng.uniform(0, config.max_dwell))
    else:
        index = rng.choice(stops)
        if action == "go":
            max_speed: float = agent.spec.max_speed
            point = replace(profile[index], speed=rng.uniform(min(config.min_speed, max_speed), max_speed),
                            dwell=0.0)
        else:
            point = replace(profile[index],
                            dwell=gauss_step(rng, profile[index].dwell, config.sigma_t, config.max_dwell))
    return with_profile(agent, profile[:index] + (point,) + profile[index + 1:])


def mut_start_t(agent: Replayed, rng: Random, config: ReplayOperatorConfig) -> Replayed:
    """
    Move the start time by a normal step, clamped to ``[0, config.max_start_t]``.

    :param Replayed agent: Agent whose start time is mutated.
    :param Random rng: Random number generator used to sample the time step.
    :param ReplayOperatorConfig config: Configuration defining ``sigma_t`` and the maximum start time, in seconds.
    :returns: The agent with the stepped start time and all other attributes preserved.
    :rtype: Replayed
    """
    return replace(agent, start_t=gauss_step(rng, agent.start_t, config.sigma_t, config.max_start_t))


def cx_drive_style(first: Replayed, second: Replayed) -> Replayed | None:
    """
    Give ``first`` the drive style of ``second``.

    :param Replayed first: Parent receiving the drive style.
    :param Replayed second: Parent providing the drive style.
    :returns: The child, or ``None`` if the drive styles are equal.
    :rtype: Replayed | None
    """
    return None if first.drive_style == second.drive_style else replace(first, drive_style=second.drive_style)


def mut_drive_style(agent: Replayed, rng: Random) -> Replayed:
    """
    Move the drive style by a reflected normal step within [0, 1].

    Use ``realism.style_step`` to preserve the uniform style prior.

    :param Replayed agent: Agent whose drive style is mutated.
    :param Random rng: Random number generator used to sample the style step.
    :returns: The agent with the updated drive style and all other attributes preserved.
    :rtype: Replayed
    """
    return replace(agent, drive_style=style_step(rng, agent.drive_style))
