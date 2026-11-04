"""
RSS trajectory sets of unstructured constellations, ported from ad-rss-lib
5.0.0 ``ad_rss/src/unstructured/TrajectoryCommon.cpp``,
``TrajectoryVehicle.cpp`` and ``TrajectoryPedestrian.cpp`` (digest §6.1-6.3).

Each road user gets a brake set (everywhere it may be when it brakes as RSS
requires) and a continue-forward set (everywhere it may be when it does not),
as shapely geometries. Where the library's polygon union fails because the
pieces do not form one polygon, this port keeps the multi-polygon: deciding
whether two sets collide is the same.
"""

from dataclasses import dataclass, field, replace
from math import cos, fabs, pi, sin
from typing import List, Tuple, Iterator
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ...utils import rectangle_corners
from .physics import (DISTANCE, DISTANCE_MAX, DURATION, SPEED, accelerated_limited_movement, feq, fge, fgt, flt,
                      fmax, speed_and_distance_offset, time_to_stop)
from .types import ObjectType, RelativeObjectState

__all__ = ["TrajectoryPoint", "pedestrian_sets", "sets", "vehicle_sets"]

_RATIO = 1e-6
"""Tolerance of the ratio loops, against float accumulation."""
_PEDESTRIAN_MAX_RADIUS = 1000.0
"""Metres: the widest circle a pedestrian walks on; a smaller turning ratio walks straight."""
_CIRCLE_STEP = 2 * pi / 20
"""Radians between the 20 vertices of a standing pedestrian's circle (``_circle``)."""


@dataclass(frozen=True)
class TrajectoryPoint:
    """
    A sampled unstructured trajectory pose and motion state.

    :param float x: Body centre, map X coordinate in metres.
    :param float y: Body centre, map Y coordinate in metres.
    :param float angle: Heading in radians.
    :param float speed: Speed in metres per second.
    :param float yaw_rate: Angular velocity in radians per second, counter-clockwise.
    """
    x: float
    """Body centre, map X coordinate in metres."""
    y: float
    """Body centre, map Y coordinate in metres."""
    angle: float
    """Heading in radians."""
    speed: float
    """Speed in metres per second."""
    yaw_rate: float = 0.0
    """Angular velocity in radians per second, counter-clockwise."""


@dataclass
class _Step:
    """
    Left, right and centre trajectory samples at one reachable-set calculation step.

    :param List[TrajectoryPoint] left: The samples at positive ratios, left of the centre, in ratio order: the
        last turns hardest.
    :param List[TrajectoryPoint] right: The samples at negative ratios, right of the centre, in ratio order: the
        first turns hardest.
    :param TrajectoryPoint | None center: The sample at ratio 0: a vehicle keeping its yaw rate, a pedestrian
        walking straight; None until set.
    """
    left: List[TrajectoryPoint] = field(default_factory=list)
    """The samples at positive ratios, left of the centre, in ratio order: the last turns hardest."""
    right: List[TrajectoryPoint] = field(default_factory=list)
    """The samples at negative ratios, right of the centre, in ratio order: the first turns hardest."""
    center: TrajectoryPoint | None = None
    """The sample at ratio 0: a vehicle keeping its yaw rate, a pedestrian walking straight; None until set."""


@dataclass(frozen=True)
class _Location:
    """
    The footprints a step leaves: its last left point, centre and first right point.

    :param Tuple[Tuple[float, float], ...] left: The body corners at the hardest left turn (``_corners``).
    :param Tuple[Tuple[float, float], ...] center: The body corners at the centre sample.
    :param Tuple[Tuple[float, float], ...] right: The body corners at the hardest right turn.
    """
    left: Tuple[Tuple[float, float], ...]
    """The body corners at the hardest left turn (``_corners``)."""
    center: Tuple[Tuple[float, float], ...]
    """The body corners at the centre sample."""
    right: Tuple[Tuple[float, float], ...]
    """The body corners at the hardest right turn."""


def _start(obj: RelativeObjectState, maximum: bool) -> TrajectoryPoint:
    """
    Create the initial trajectory point using the selected minimum or maximum object speed.

    :param RelativeObjectState obj: The road user.
    :param bool maximum: Whether to start at its ``speed_max``, else its ``speed_min``.
    :returns: Its body centre, heading, chosen speed and yaw rate now.
    :rtype: TrajectoryPoint
    """
    u = obj.unstructured
    return TrajectoryPoint(u.x, u.y, u.yaw, u.speed_max if maximum else u.speed_min, u.yaw_rate)


def _corners(point: TrajectoryPoint, obj: RelativeObjectState) -> Tuple[Tuple[float, float], ...]:
    """
    Front-left, front-right, back-left, back-right of the body at ``point`` (TrajectoryCommon.cpp:30-75).

    :param TrajectoryPoint point: The pose.
    :param RelativeObjectState obj: The road user, for its size.
    :returns: The four ``(x, y)`` corners in that order; the ego's front reaches its
        ``min_longitudinal_safety_distance`` further.
    :rtype: Tuple[Tuple[float, float], ...]
    """
    u = obj.unstructured
    front = u.length / 2
    if obj.object_type == ObjectType.EGO_VEHICLE:
        front += obj.dynamics.min_longitudinal_safety_distance
    corners = rectangle_corners(point.x, point.y, point.angle, front=front,
                                back=u.length / 2, left=u.width / 2, right=u.width / 2)
    # RSS indexes front-left, front-right, back-left, back-right, not a ring.
    return tuple(tuple(corners[index]) for index in (0, 3, 1, 2))


def _hull(points) -> BaseGeometry:
    """
    Return the convex hull of the supplied map-frame points.

    :param Iterable[Tuple[float, float]] points: The points.
    :returns: Their hull: a polygon, or a line or point where they do not span an area.
    :rtype: BaseGeometry
    """
    return MultiPoint(list(points)).convex_hull


def _union(a: BaseGeometry | None, b: BaseGeometry) -> BaseGeometry:
    """
    Union a new geometry with the accumulated set, accepting an empty or absent accumulator.

    :param BaseGeometry | None a: The set so far; None or empty for none yet.
    :param BaseGeometry b: The geometry to add.
    :returns: Their union; ``b`` itself without a set so far.
    :rtype: BaseGeometry
    """
    return b if a is None or a.is_empty else unary_union([a, b])


def _step_polygon(obj: RelativeObjectState, step: _Step) -> Tuple[BaseGeometry, _Location]:
    """
    Build a step's left/right hulls and return their union with the boundary footprint locations.

    :param RelativeObjectState obj: The road user, for its size.
    :param _Step step: The step, with at least one left and one right sample and its centre.
    :returns: ``(polygon, location)``: the hull of the left samples' corners with the centre's, united with
        that of the centre's with the right samples'; and the step's footprints.
    :rtype: Tuple[BaseGeometry, _Location]
    """
    left_points = [c for p in step.left for c in _corners(p, obj)]
    center = _corners(step.center, obj)
    right_points = [c for p in step.right for c in _corners(p, obj)]
    location = _Location(_corners(step.left[-1], obj), center, _corners(step.right[0], obj))
    return unary_union([_hull(left_points + list(center)), _hull(list(center) + right_points)]), location


def _between(polygon: BaseGeometry | None, previous: _Location, current: _Location) -> BaseGeometry | None:
    """
    Add swept hulls between consecutive footprint locations to the accumulated set.

    :param BaseGeometry | None polygon: The set so far; None for none yet.
    :param _Location previous: The earlier step's footprints.
    :param _Location current: The later step's.
    :returns: The set with the hulls joining their left and centre, and their right and centre, footprints;
        unchanged when the two are the same.
    :rtype: BaseGeometry | None
    """
    if previous == current:
        return polygon
    left = _hull(previous.left + previous.center + current.left + current.center)
    right = _hull(previous.right + previous.center + current.right + current.center)
    return _union(polygon, unary_union([right, left]))


def _front_and_sides(obj: RelativeObjectState, initial: _Location, sides: List[_Step], front: _Step,
                     polygon: BaseGeometry | None) -> Tuple[BaseGeometry, _Location]:
    """
    Accumulate side and front step polygons with connecting sweeps; return the set and final location.

    :param RelativeObjectState obj: The road user, for its size.
    :param _Location initial: The footprints the sweep starts from.
    :param List[_Step] sides: The intermediate steps, in order.
    :param _Step front: The last step.
    :param BaseGeometry | None polygon: The set so far; None for none yet.
    :returns: ``(set, location)``: the set with every step and the sweeps between them, and the front step's
        footprints.
    :rtype: Tuple[BaseGeometry, _Location]
    """
    previous = initial
    for side in sides:
        step_polygon, location = _step_polygon(obj, side)
        polygon = _between(polygon, previous, location)
        previous = location
        polygon = _union(polygon, step_polygon)
    front_polygon, front_location = _step_polygon(obj, front)
    polygon = _between(polygon, previous, front_location)
    return _union(polygon, front_polygon), front_location


def _response_time_polygon(obj: RelativeObjectState, initial: _Location, polygon: BaseGeometry) -> BaseGeometry:
    """
    Include the response-time hull connecting the minimum-speed start body to the initial location.

    :param RelativeObjectState obj: The road user.
    :param _Location initial: The footprints the set's sweep starts from.
    :param BaseGeometry polygon: The set so far.
    :returns: The set with the hull of the body now and ``initial``.
    :rtype: BaseGeometry
    """
    current = _corners(_start(obj, maximum=False), obj)
    return _union(polygon, _hull(initial.left + initial.center + initial.right + current))


def _ratios(diff: float, *, negative: bool, positive: bool, zero: bool) -> Iterator[float]:
    """
    The library's ratio loops: ``-1 .. <0`` step ``diff``; ``diff .. <=1``; and 0 when asked.

    :param float diff: The step between ratios.
    :param bool negative: Whether to yield ``-1`` up to below 0.
    :param bool positive: Whether to yield ``diff`` up to 1.
    :param bool zero: Whether to yield 0 between them.
    :returns: The ratios, in that order.
    :rtype: Iterator[float]
    """
    if negative:
        r = -1.0
        while r < -_RATIO:
            yield r
            r += diff
    if zero:
        yield 0.0
    if positive:
        r = diff
        while r <= 1.0 + _RATIO:
            yield r
            r += diff


# Vehicles (TrajectoryVehicle.cpp)

def _time_in_movement(speed: float, acceleration: float, time: float) -> float:
    """
    Limit a movement duration in seconds to the stopping time under negative acceleration.

    :param float speed: m/s.
    :param float acceleration: m/s^2.
    :param float time: Seconds.
    :returns: Seconds: ``time``, or less where braking stops it first; 0 standing while braking.
    :rtype: float
    """
    if acceleration < 0:
        if feq(speed, 0.0, SPEED):
            return 0.0
        stop = time_to_stop(speed, 0.0, speed, acceleration, acceleration)
        return min(time, stop if stop is not None else time)
    return time


def _on_circle(point: TrajectoryPoint, acceleration: float, duration: float, obj: RelativeObjectState
               ) -> TrajectoryPoint:
    """
    One move on the circle the yaw rate gives (TrajectoryVehicle.cpp:271-319).

    :param TrajectoryPoint point: The start.
    :param float acceleration: m/s^2 along the way.
    :param float duration: Seconds.
    :param RelativeObjectState obj: The road user, for its speed cap and ``vehicle_min_radius``.
    :returns: The point moved on the circle, its heading turned and its speed changed; ``point`` itself for
        no duration.
    :rtype: TrajectoryPoint
    """
    if feq(duration, 0.0, DURATION):
        return point
    d = obj.dynamics
    moved = accelerated_limited_movement(point.speed, d.max_speed_on_acceleration, acceleration, duration)
    final_speed, distance = moved if moved is not None else (point.speed, 0.0)
    radius = DISTANCE_MAX if feq(point.yaw_rate, 0.0, 1e-3) else point.speed / point.yaw_rate
    if fabs(radius) < d.unstructured.vehicle_min_radius:
        radius = d.unstructured.vehicle_min_radius if radius >= 0 else -d.unstructured.vehicle_min_radius
    start = point.angle - pi / 2
    ox, oy = point.x - radius * cos(start), point.y - radius * sin(start)
    turn = distance / radius
    dx, dy = point.x - ox, point.y - oy
    x, y = ox + dx * cos(turn) - dy * sin(turn), oy + dx * sin(turn) + dy * cos(turn)
    return replace(point, x=x, y=y, angle=point.angle + turn, speed=final_speed)


def _vehicle_point(start: TrajectoryPoint, obj: RelativeObjectState, duration: float, acceleration: float,
                   ratio: float) -> TrajectoryPoint:
    """
    Integrate with the yaw rate changing at ``ratio`` of its maximum (TrajectoryVehicle.cpp:237-269).

    :param TrajectoryPoint start: The start.
    :param RelativeObjectState obj: The road user, for its settings.
    :param float duration: Seconds.
    :param float acceleration: m/s^2 along the way.
    :param float ratio: Of ``vehicle_yaw_rate_change``, from -1 (hardest right) to 1 (hardest left).
    :returns: Where it is after ``duration``, in ``vehicle_trajectory_calculation_step`` steps.
    :rtype: TrajectoryPoint
    """
    settings = obj.dynamics.unstructured
    point, t = start, 0.0
    while flt(t, duration, DURATION):
        step = settings.vehicle_trajectory_calculation_step
        if fge(duration, t + step, DURATION):
            t += step
        else:
            step, t = duration - t, duration
        point = replace(point, yaw_rate=start.yaw_rate + settings.vehicle_yaw_rate_change * t * ratio)
        point = _on_circle(point, acceleration, step, obj)
    return point


def _vehicle_response_step(obj: RelativeObjectState, acceleration: float, steps: int, maximum: bool) -> _Step:
    """
    Sample vehicle steering ratios over its moving portion of response time at the given acceleration.

    :param RelativeObjectState obj: The vehicle.
    :param float acceleration: m/s^2 during the response time.
    :param int steps: How many ratios lie between the hardest turn and the centre on each side.
    :param bool maximum: Whether to start at its ``speed_max``, else its ``speed_min``.
    :returns: The step at the end of the response time, or where it stops first.
    :rtype: _Step
    """
    diff = 2.0 / (2.0 * steps + 2.0)
    start = _start(obj, maximum)
    duration = _time_in_movement(start.speed, acceleration, obj.dynamics.response_time)
    step = _Step()
    step.right = [_vehicle_point(start, obj, duration, acceleration, r)
                  for r in _ratios(diff, negative=True, positive=False, zero=False)]
    step.left = [_vehicle_point(start, obj, duration, acceleration, r)
                 for r in _ratios(diff, negative=False, positive=True, zero=False)]
    step.center = _vehicle_point(start, obj, duration, acceleration, 0.0)
    return step


def _move_step(step: _Step, obj: RelativeObjectState, duration: float, acceleration: float) -> _Step:
    """
    Advance all step samples along their turning circles for the given duration and acceleration.

    :param _Step step: The step.
    :param RelativeObjectState obj: The vehicle.
    :param float duration: Seconds.
    :param float acceleration: m/s^2.
    :returns: A new step, each sample moved (``_on_circle``).
    :rtype: _Step
    """
    return _Step([_on_circle(p, acceleration, duration, obj) for p in step.left],
                 [_on_circle(p, acceleration, duration, obj) for p in step.right],
                 _on_circle(step.center, acceleration, duration, obj))


def vehicle_sets(obj: RelativeObjectState) -> Tuple[BaseGeometry, BaseGeometry] | None:
    """
    The brake and continue-forward sets of a vehicle (TrajectoryVehicle.cpp:29-110).

    :param RelativeObjectState obj: A vehicle in an unstructured constellation.
    :returns: ``(brake, continue_forward)``, or None when the library's calculation fails.
    :rtype: Tuple[BaseGeometry, BaseGeometry] | None
    """
    d, u = obj.dynamics, obj.unstructured
    lon, settings = d.alpha_lon, d.unstructured
    stop_max = time_to_stop(u.speed_max, d.response_time, d.max_speed_on_acceleration, lon.accel_max, lon.brake_min)
    stop_min = time_to_stop(u.speed_min, d.response_time, d.max_speed_on_acceleration, lon.accel_max, lon.brake_min)
    if stop_max is None or stop_min is None:
        return None
    back = _vehicle_response_step(obj, lon.brake_max, settings.vehicle_back_intermediate_yaw_rate_change_ratio_steps,
                                  maximum=False)
    front = _vehicle_response_step(obj, lon.accel_max,
                                   settings.vehicle_front_intermediate_yaw_rate_change_ratio_steps, maximum=True)
    after_max = fmax(0.0, stop_max - d.response_time, DURATION)
    after_min = fmax(0.0, stop_min - d.response_time, DURATION)

    # Brake (TrajectoryVehicle.cpp:321-450).
    stop_brake_max = 0.0
    if fgt(back.center.speed, 0.0, SPEED):
        stop_brake_max = time_to_stop(back.center.speed, after_min, d.max_speed_on_acceleration, lon.brake_max,
                                      lon.brake_max)
        if stop_brake_max is None:
            return None
    brake, brake_max_location = _step_polygon(obj, _move_step(back, obj, stop_brake_max, lon.brake_max))
    brake_front = _move_step(front, obj, after_max, lon.brake_min)
    accelerations = []
    diff = (lon.brake_min - lon.brake_max) / (1.0 + settings.vehicle_brake_intermediate_acceleration_steps)
    a = lon.brake_max + diff
    while flt(a, lon.brake_min, 1e-4):
        accelerations.append(a)
        a += diff
    accelerations.append(lon.brake_min)
    sides = []
    for a in accelerations:
        t = _time_in_movement(front.center.speed, a, after_max)
        side = _Step([front.left[-1]], [front.right[0]], front.center)
        sides.append(_move_step(side, obj, t, a))
    brake, brake_min_location = _front_and_sides(obj, brake_max_location, sides, brake_front, brake)
    brake = _response_time_polygon(obj, brake_max_location, brake)

    # Continue forward (TrajectoryVehicle.cpp:452-646).
    diff = 2.0 / (2.0 * settings.vehicle_continue_forward_intermediate_yaw_rate_change_ratio_steps + 2.0)
    forward = _move_step(front, obj, after_max, lon.accel_max)
    forward.left = forward.left + [_vehicle_point(front.left[-1], obj, after_max, lon.accel_max, r)
                                   for r in _ratios(diff, negative=False, positive=True, zero=False)]
    forward.right = [_vehicle_point(front.right[0], obj, after_max, lon.accel_max, r)
                     for r in _ratios(diff, negative=True, positive=False, zero=False)] + forward.right
    accelerations, diff = [], (lon.accel_max - lon.brake_min) / (
        1.0 + settings.vehicle_continue_forward_intermediate_acceleration_steps)
    a = lon.brake_min + diff
    while flt(a, lon.accel_max, 1e-4):
        accelerations.append(a)
        a += diff
    accelerations.append(lon.accel_max)
    sides = []
    for a in accelerations:
        t = _time_in_movement(front.center.speed, a, after_max)
        sides.append(_Step([_vehicle_point(front.left[-1], obj, t, a, 1.0)],
                           [_vehicle_point(front.right[0], obj, t, a, -1.0)],
                           _on_circle(front.center, a, t, obj)))
    forward_set, _ = _front_and_sides(obj, brake_min_location, sides, forward, brake)
    forward_set = _response_time_polygon(obj, brake_min_location, forward_set)
    return brake, forward_set


# Pedestrians (TrajectoryPedestrian.cpp)

def _pedestrian_move(point: TrajectoryPoint, obj: RelativeObjectState, distance: float, ratio: float
                     ) -> TrajectoryPoint:
    """
    Walk ``distance`` on the circle ``ratio`` gives, or straight (TrajectoryPedestrian.cpp:556-585).

    :param TrajectoryPoint point: The start.
    :param RelativeObjectState obj: The pedestrian, for its ``pedestrian_turning_radius``.
    :param float distance: Metres.
    :param float ratio: From -1 (hardest right) to 1 (hardest left): the radius is the turning radius over it,
        straight where that is wider than ``_PEDESTRIAN_MAX_RADIUS``.
    :returns: The point after walking, its heading turned; its speed unchanged.
    :rtype: TrajectoryPoint
    """
    if feq(distance, 0.0, DISTANCE):
        return point
    turning = obj.dynamics.unstructured.pedestrian_turning_radius
    if fabs(ratio) > turning / _PEDESTRIAN_MAX_RADIUS:
        radius = turning / ratio
        start = point.angle - pi / 2
        ox, oy = point.x - radius * cos(start), point.y - radius * sin(start)
        turn = distance / radius
        return replace(point, x=ox + radius * cos(start + turn), y=oy + radius * sin(start + turn),
                       angle=point.angle + turn)
    return replace(point, x=point.x + cos(point.angle) * distance, y=point.y + sin(point.angle) * distance)


def _pedestrian_point(point: TrajectoryPoint, obj: RelativeObjectState, duration: float, acceleration: float,
                      ratio: float) -> TrajectoryPoint:
    """
    Advance a pedestrian with capped acceleration and a selected turning ratio.

    :param TrajectoryPoint point: The start.
    :param RelativeObjectState obj: The pedestrian.
    :param float duration: Seconds.
    :param float acceleration: m/s^2.
    :param float ratio: As for ``_pedestrian_move``.
    :returns: The point after ``duration``, with its new speed.
    :rtype: TrajectoryPoint
    """
    moved = accelerated_limited_movement(point.speed, obj.dynamics.max_speed_on_acceleration, acceleration, duration)
    speed, distance = moved if moved is not None else (point.speed, 0.0)
    return _pedestrian_move(replace(point, speed=speed), obj, distance, ratio)


def _straight(point: TrajectoryPoint, distance: float) -> TrajectoryPoint:
    """
    Translate a trajectory point along its heading for a positive distance in metres.

    :param TrajectoryPoint point: The point.
    :param float distance: Metres ahead.
    :returns: The point moved; ``point`` itself for a distance not above 0.
    :rtype: TrajectoryPoint
    """
    if fgt(distance, 0.0, DISTANCE):
        return replace(point, x=point.x + cos(point.angle) * distance, y=point.y + sin(point.angle) * distance)
    return point


def _straight_step(step: _Step, distance: float) -> _Step:
    """
    Translate every step sample straight ahead, leaving nonpositive distances unchanged.

    :param _Step step: The step.
    :param float distance: Metres ahead.
    :returns: A new step, each sample moved; ``step`` itself for a distance not above 0.
    :rtype: _Step
    """
    if not fgt(distance, 0.0, DISTANCE):
        return step
    return _Step([_straight(p, distance) for p in step.left], [_straight(p, distance) for p in step.right],
                 _straight(step.center, distance))


def _pedestrian_response_step(obj: RelativeObjectState, acceleration: float, steps: int, maximum: bool) -> _Step:
    """
    Sample pedestrian turning ratios through response time at the selected initial speed.

    :param RelativeObjectState obj: The pedestrian.
    :param float acceleration: m/s^2 during the response time.
    :param int steps: How many ratios lie between the hardest turn and the centre on each side.
    :param bool maximum: Whether to start at its ``speed_max``, else its ``speed_min``.
    :returns: The step at the end of the response time.
    :rtype: _Step
    """
    diff = 2.0 / (2.0 * steps + 2.0)
    step, r = _Step(), -1.0
    while r <= 1.0 + _RATIO:
        point = _pedestrian_point(_start(obj, maximum), obj, obj.dynamics.response_time, acceleration, r)
        if abs(r) < _RATIO:
            step.center = point
        elif r > 0:
            step.left.append(point)
        else:
            step.right.append(point)
        r += diff
    return step


def _circle(x: float, y: float, radius: float) -> Polygon:
    """
    Approximate a map-frame circle with a regular 20-vertex polygon.

    :param float x: The centre, map frame.
    :param float y: The centre.
    :param float radius: Metres.
    :returns: The polygon, its vertices on the circle.
    :rtype: Polygon
    """
    return Polygon([(x + radius * cos(k * _CIRCLE_STEP), y + radius * sin(k * _CIRCLE_STEP)) for k in range(20)])


def pedestrian_sets(obj: RelativeObjectState) -> Tuple[BaseGeometry, BaseGeometry] | None:
    """
    The brake and continue-forward sets of a pedestrian (TrajectoryPedestrian.cpp:34-374).

    :param RelativeObjectState obj: A pedestrian.
    :returns: ``(brake, continue_forward)``, circles around one standing; None when the library's calculation
        fails.
    :rtype: Tuple[BaseGeometry, BaseGeometry] | None
    """
    d, u = obj.dynamics, obj.unstructured
    lon, settings = d.alpha_lon, d.unstructured
    v_max = d.max_speed_on_acceleration
    stop_max = time_to_stop(u.speed_max, d.response_time, v_max, lon.accel_max, lon.brake_min)
    if stop_max is None:
        return None
    if feq(u.speed_max, 0.0, SPEED):
        brake_reach = speed_and_distance_offset(stop_max, u.speed_max, d.response_time, v_max, lon.accel_max,
                                                lon.brake_min)
        forward_reach = speed_and_distance_offset(stop_max, u.speed_max, d.response_time, v_max, lon.accel_max,
                                                  lon.accel_max)
        if brake_reach is None or forward_reach is None:
            return None
        return _circle(u.x, u.y, brake_reach[1]), _circle(u.x, u.y, forward_reach[1])
    stop_min = time_to_stop(u.speed_min, d.response_time, v_max, lon.accel_max, lon.brake_min)
    if stop_min is None:
        return None
    back = _pedestrian_response_step(obj, lon.brake_max,
                                     settings.pedestrian_back_intermediate_heading_change_ratio_steps, maximum=False)
    front = _pedestrian_response_step(obj, lon.accel_max,
                                      settings.pedestrian_front_intermediate_heading_change_ratio_steps, maximum=True)
    stop_brake_max = 0.0
    if fgt(back.center.speed, 0.0, SPEED):
        stop_brake_max = time_to_stop(back.center.speed, fmax(0.0, stop_min - d.response_time, DURATION), v_max,
                                      lon.brake_max, lon.brake_max)
        if stop_brake_max is None:
            return None
    after_max = fmax(0.0, stop_max - d.response_time, DURATION)
    brake_max_distance = accelerated_limited_movement(back.center.speed, v_max, lon.brake_max, stop_brake_max)[1]
    brake_min_distance = accelerated_limited_movement(front.center.speed, v_max, lon.brake_min, after_max)[1]
    accel_max_distance = accelerated_limited_movement(front.center.speed, v_max, lon.accel_max, after_max)[1]

    brake_front = _straight_step(front, brake_min_distance)
    reach_at_response = accelerated_limited_movement(u.speed_max, v_max, lon.accel_max, d.response_time)[1]
    width = reach_at_response / (1.0 + settings.pedestrian_brake_intermediate_acceleration_steps)
    sides, distance = [], width
    while width > 0 and flt(distance, reach_at_response, DISTANCE):
        start = _start(obj, maximum=True)
        sides.append(_Step([_pedestrian_move(start, obj, distance, 1.0)],
                           [_pedestrian_move(start, obj, distance, -1.0)],
                           _pedestrian_move(start, obj, distance, 0.0)))
        distance += width
    right = _straight(back.right[0], brake_max_distance)
    left = _straight(back.left[-1], brake_max_distance)
    center = _straight(back.center, brake_max_distance)
    location = _Location(_corners(left, obj), _corners(center, obj), _corners(right, obj))
    right_corners, left_corners = _corners(right, obj), _corners(left, obj)
    # backRight, frontRight of the right point; backLeft, frontLeft of the left point.
    brake = _hull([right_corners[3], right_corners[1], left_corners[2], left_corners[0]])
    brake, _ = _front_and_sides(obj, location, sides, brake_front, brake)
    brake = _response_time_polygon(obj, location, brake)

    forward = _straight_step(front, accel_max_distance)
    forward = _Step(list(forward.left), list(forward.right), forward.center)
    diff = 2.0 / (2.0 * settings.pedestrian_continue_forward_intermediate_heading_change_ratio_steps + 2.0)
    forward.left += [_pedestrian_move(front.left[-1], obj, accel_max_distance, r)
                     for r in _ratios(diff, negative=False, positive=True, zero=False)]
    forward.right += [_pedestrian_move(front.right[0], obj, accel_max_distance, r)
                      for r in _ratios(diff, negative=True, positive=False, zero=False)]
    width = (accel_max_distance - brake_max_distance) / (
        1.0 + settings.pedestrian_continue_forward_intermediate_acceleration_steps)
    sides, distance = [], brake_max_distance
    while distance < accel_max_distance + DISTANCE:
        sides.append(_Step([_pedestrian_move(front.left[-1], obj, distance, 1.0)],
                           [_pedestrian_move(front.right[0], obj, distance, -1.0)],
                           _pedestrian_move(front.center, obj, distance, 0.0)))
        if width <= 0:
            break  # The library's loop never ends here (digest §6.3); one step covers it.
        distance += width
    forward_set, _ = _front_and_sides(obj, location, sides, forward, brake)
    return brake, forward_set


def sets(obj: RelativeObjectState) -> Tuple[BaseGeometry, BaseGeometry] | None:
    """
    The brake and continue-forward sets of any road user, by its type (checker 49-56).

    :param RelativeObjectState obj: A road user in an unstructured constellation.
    :returns: ``(brake, continue_forward)`` (``pedestrian_sets`` or ``vehicle_sets``); None for an artificial
        or invalid object, or when the calculation fails.
    :rtype: Tuple[BaseGeometry, BaseGeometry] | None
    """
    if obj.object_type in (ObjectType.PEDESTRIAN, ObjectType.ARTIFICIAL_PEDESTRIAN):
        return pedestrian_sets(obj)
    if obj.object_type in (ObjectType.ARTIFICIAL_OBJECT, ObjectType.INVALID):
        return None
    return vehicle_sets(obj)
