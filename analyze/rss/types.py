"""
RSS data: dynamics and parameters, object states, constellations and the
checkers' results, after ad-rss-lib 5.0.0's generated types
(``ad_rss/generated/include/ad/rss``). Decelerations are negative; lateral
values grow to the right, as in the library.
"""

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Tuple

__all__ = ["CANONICAL", "ConstellationType", "DEFAULT_UNSTRUCTURED", "Dynamics", "Evaluator", "IntersectionState",
           "LatPosition", "LatResponse", "LatState", "LateralAccelerations", "LonPosition", "LonResponse", "LonState",
           "LongitudinalAccelerations", "ObjectType", "PEDESTRIAN", "RelativeConstellation", "RelativeObjectState",
           "RssParameters", "RssState", "StructuredState", "UnstructuredObjectState", "UnstructuredResponse",
           "UnstructuredResult", "UnstructuredSettings", "VEHICLE", "Velocity", "is_dangerous",
           "is_lateral_safe", "is_longitudinal_safe"]


class ConstellationType(IntEnum):
    """
    RSS pairwise traffic configurations, including priority-specific intersection cases.
    """
    NOT_RELEVANT = 0
    """The pair need not be checked."""
    SAME_DIRECTION = 1
    """Both drive the same way along related lanes."""
    OPPOSITE_DIRECTION = 2
    """They drive towards each other along related lanes."""
    INTERSECTION_EGO_HAS_PRIORITY = 3
    """Their routes cross in a junction; the ego has priority."""
    INTERSECTION_OBJECT_HAS_PRIORITY = 4
    """Their routes cross in a junction; the other has priority."""
    INTERSECTION_SAME_PRIORITY = 5
    """Their routes cross in a junction; neither has priority."""
    UNSTRUCTURED = 6
    """Judged in the plane, by trajectory sets instead of lanes."""

    @property
    def is_intersection(self) -> bool:
        """
        Whether this is one of the three intersection types.

        :returns: True for ``INTERSECTION_EGO_HAS_PRIORITY``, ``INTERSECTION_OBJECT_HAS_PRIORITY`` and
            ``INTERSECTION_SAME_PRIORITY``.
        :rtype: bool
        """
        return self in (ConstellationType.INTERSECTION_EGO_HAS_PRIORITY,
                        ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY,
                        ConstellationType.INTERSECTION_SAME_PRIORITY)


class ObjectType(IntEnum):
    """
    RSS road-user categories used to select dynamics and response rules.
    """
    INVALID = 0
    """No valid road user: it has no trajectory sets."""
    EGO_VEHICLE = 1
    """The party in the ego's role: the first of a pair."""
    OTHER_VEHICLE = 2
    """Another vehicle."""
    PEDESTRIAN = 3
    """A pedestrian."""
    ARTIFICIAL_OBJECT = 4
    """An artificial object, as of a virtual obstacle: it has no trajectory sets."""
    ARTIFICIAL_VEHICLE = 5
    """An artificial vehicle; its trajectory sets are a vehicle's."""
    ARTIFICIAL_PEDESTRIAN = 6
    """An artificial pedestrian; its trajectory sets are a pedestrian's."""
    BICYCLE = 7
    """A bicycle."""
    OTHER_OBJECT = 8
    """Any other road user, with other vehicles' dynamics."""


class LonPosition(IntEnum):
    """
    Where the ego (object a) is along the road relative to the other (object b).
    """
    IN_FRONT = 0
    """Wholly ahead of the other; at an intersection, nearer the conflict area."""
    OVERLAP_FRONT = 1
    """Overlapping it, reaching further ahead."""
    OVERLAP = 2
    """Overlapping it, neither reaching further both ways."""
    OVERLAP_BACK = 3
    """Overlapping it, the other reaching further ahead."""
    AT_BACK = 4
    """Wholly behind the other."""


class LatPosition(IntEnum):
    """
    Where the ego is across the road relative to the other.
    """
    AT_LEFT = 0
    """Wholly left of the other."""
    OVERLAP_LEFT = 1
    """Overlapping it, reaching further left."""
    OVERLAP = 2
    """Overlapping it, neither reaching further both ways."""
    OVERLAP_RIGHT = 3
    """Overlapping it, reaching further right."""
    AT_RIGHT = 4
    """Wholly right of the other."""


class LonResponse(IntEnum):
    """
    Longitudinal proper responses, distinguishing correct-lane braking from ordinary minimum braking.
    """
    NONE = 0
    """No longitudinal restriction."""
    BRAKE_MIN_CORRECT = 1
    """Brake at least as hard as ``brake_min_correct``: in its correct lane, facing oncoming traffic."""
    BRAKE_MIN = 2
    """Brake at least as hard as ``brake_min``."""


class LatResponse(IntEnum):
    """
    Lateral proper responses: no restriction or minimum lateral braking.
    """
    NONE = 0
    """No lateral restriction."""
    BRAKE_MIN = 1
    """Stop moving towards the other, braking sideways at least as hard as the lateral ``brake_min``."""


class UnstructuredResponse(IntEnum):
    """
    Unstructured proper responses for reachable-set interactions.
    """
    NONE = 0
    """No response given."""
    CONTINUE_FORWARD = 1
    """Safe: it may go on as it drives."""
    DRIVE_AWAY = 2
    """It may move only with a heading in ``heading_range``, away from the other."""
    BRAKE = 3
    """It must brake."""


class Evaluator(IntEnum):
    """
    Identifier of the distance or intersection rule that produced an RSS result.
    """
    NONE = 0
    """No rule evaluated, as on the lateral side away from the other."""
    LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION_EGO_CORRECT_LANE = 1
    """Opposite directions, the ego in its correct lane braking at ``brake_min_correct``."""
    LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION = 2
    """Opposite directions, the other in its correct lane."""
    LONGITUDINAL_DISTANCE_SAME_DIRECTION_EGO_FRONT = 3
    """Same direction, the ego in front."""
    LONGITUDINAL_DISTANCE_SAME_DIRECTION_OTHER_IN_FRONT = 4
    """Same direction, the other in front."""
    LATERAL_DISTANCE = 5
    """The lateral safe distance; 0 where they overlap across the road."""
    INTERSECTION_OTHER_PRIORITY_EGO_ABLE_TO_STOP = 6
    """The ego without priority: whether it can stop in front of the conflict area."""
    INTERSECTION_EGO_PRIORITY_OTHER_ABLE_TO_STOP = 7
    """The other without priority: whether it can stop in front of the conflict area."""
    INTERSECTION_EGO_IN_FRONT = 8
    """The ego nearer the conflict area: the same-direction safe distance, the ego leading."""
    INTERSECTION_OTHER_IN_FRONT = 9
    """The other nearer the conflict area: the same-direction safe distance, the other leading."""
    INTERSECTION_OVERLAP = 10
    """Whether their times in the conflict area overlap."""


class IntersectionState(Enum):
    """
    Which sub-check last found an intersection safe (RssIntersectionConstellationChecker).
    """
    NON_PRIO_ABLE_TO_BREAK = 1
    """The one without priority can stop in front of the conflict area (the library's spelling)."""
    SAFE_LONGITUDINAL_DISTANCE = 2
    """They keep the safe distance of one following the other."""
    NO_TIME_OVERLAP = 3
    """One leaves the conflict area before the other can reach it."""


@dataclass(frozen=True)
class LongitudinalAccelerations:
    """
    Longitudinal acceleration limits in metres per second squared; braking values are negative,
    ``brake_max <= brake_min <= brake_min_correct < 0``.

    :param float accel_max: The most it may accelerate during its response time.
    :param float brake_max: The hardest it can brake.
    :param float brake_min: The least braking it owes as a proper response.
    :param float brake_min_correct: The least braking owed in its correct lane facing oncoming traffic.
    """
    accel_max: float
    """The most it may accelerate during its response time."""
    brake_max: float
    """The hardest it can brake."""
    brake_min: float
    """The least braking it owes as a proper response."""
    brake_min_correct: float
    """The least braking owed in its correct lane facing oncoming traffic."""


@dataclass(frozen=True)
class LateralAccelerations:
    """
    Lateral acceleration limits in metres per second squared; braking is negative.

    :param float accel_max: The most it may accelerate sideways during its response time.
    :param float brake_min: The least lateral braking it owes as a proper response.
    """
    accel_max: float
    """The most it may accelerate sideways during its response time."""
    brake_min: float
    """The least lateral braking it owes as a proper response."""


@dataclass(frozen=True)
class UnstructuredSettings:
    """
    Turning bounds and trajectory sampling counts for unstructured reachable sets. A count of ratio steps is
    how many ratios lie between the hardest turn and the centre on each side.

    :param float pedestrian_turning_radius: Metres: a pedestrian's tightest turn.
    :param float drive_away_max_angle: Radians either side of straight away from the other that driving away
        may head.
    :param float vehicle_yaw_rate_change: rad/s^2: how fast a vehicle's yaw rate may change.
    :param float vehicle_min_radius: Metres: a vehicle's tightest turn.
    :param float vehicle_trajectory_calculation_step: Seconds per integration step of a vehicle's trajectory.
    :param int vehicle_front_intermediate_yaw_rate_change_ratio_steps: Ratio steps of a vehicle accelerating
        in its response time.
    :param int vehicle_back_intermediate_yaw_rate_change_ratio_steps: Ratio steps of a vehicle braking hardest
        in its response time.
    :param int vehicle_brake_intermediate_acceleration_steps: Accelerations sampled between ``brake_max`` and
        ``brake_min`` for a vehicle's brake set.
    :param int vehicle_continue_forward_intermediate_acceleration_steps: Accelerations sampled between
        ``brake_min`` and ``accel_max`` for a vehicle's continue-forward set.
    :param int vehicle_continue_forward_intermediate_yaw_rate_change_ratio_steps: Ratio steps after the
        response time for a vehicle's continue-forward set.
    :param int pedestrian_continue_forward_intermediate_heading_change_ratio_steps: Ratio steps after the
        response time for a pedestrian's continue-forward set.
    :param int pedestrian_continue_forward_intermediate_acceleration_steps: Distances sampled between braking
        hardest and accelerating for a pedestrian's continue-forward set.
    :param int pedestrian_brake_intermediate_acceleration_steps: Distances sampled within a pedestrian's reach
        in its response time for its brake set.
    :param int pedestrian_front_intermediate_heading_change_ratio_steps: Ratio steps of a pedestrian
        accelerating in its response time.
    :param int pedestrian_back_intermediate_heading_change_ratio_steps: Ratio steps of a pedestrian braking
        hardest in its response time.
    """
    pedestrian_turning_radius: float = 2.0
    """Metres: a pedestrian's tightest turn."""
    drive_away_max_angle: float = 2.4
    """Radians either side of straight away from the other that driving away may head."""
    vehicle_yaw_rate_change: float = 0.3
    """rad/s^2: how fast a vehicle's yaw rate may change."""
    vehicle_min_radius: float = 3.5
    """Metres: a vehicle's tightest turn."""
    vehicle_trajectory_calculation_step: float = 0.2
    """Seconds per integration step of a vehicle's trajectory."""
    vehicle_front_intermediate_yaw_rate_change_ratio_steps: int = 3
    """Ratio steps of a vehicle accelerating in its response time."""
    vehicle_back_intermediate_yaw_rate_change_ratio_steps: int = 0
    """Ratio steps of a vehicle braking hardest in its response time."""
    vehicle_brake_intermediate_acceleration_steps: int = 1
    """Accelerations sampled between ``brake_max`` and ``brake_min`` for a vehicle's brake set."""
    vehicle_continue_forward_intermediate_acceleration_steps: int = 0
    """Accelerations sampled between ``brake_min`` and ``accel_max`` for a vehicle's continue-forward set."""
    vehicle_continue_forward_intermediate_yaw_rate_change_ratio_steps: int = 3
    """Ratio steps after the response time for a vehicle's continue-forward set."""
    pedestrian_continue_forward_intermediate_heading_change_ratio_steps: int = 3
    """Ratio steps after the response time for a pedestrian's continue-forward set."""
    pedestrian_continue_forward_intermediate_acceleration_steps: int = 0
    """Distances sampled between braking hardest and accelerating for a pedestrian's continue-forward set."""
    pedestrian_brake_intermediate_acceleration_steps: int = 3
    """Distances sampled within a pedestrian's reach in its response time for its brake set."""
    pedestrian_front_intermediate_heading_change_ratio_steps: int = 4
    """Ratio steps of a pedestrian accelerating in its response time."""
    pedestrian_back_intermediate_heading_change_ratio_steps: int = 0
    """Ratio steps of a pedestrian braking hardest in its response time."""


DEFAULT_UNSTRUCTURED = UnstructuredSettings()
"""ad-rss-lib's test-support settings (TestSupport.cpp:92-107)."""


@dataclass(frozen=True)
class Dynamics:
    """
    One road user's RSS dynamics (``world::RssDynamics``).

    :param LongitudinalAccelerations alpha_lon: Its longitudinal limits.
    :param LateralAccelerations alpha_lat: Its lateral limits.
    :param float response_time: Seconds before it applies a proper response.
    :param float lateral_fluctuation_margin: Metres it may sway sideways; half of each party's adds to the
        lateral safe distance.
    :param float max_speed_on_acceleration: m/s: the speed it may reach accelerating in its response time.
    :param float min_longitudinal_safety_distance: Metres added to its longitudinal safe distance, and to the
        ego's front in its trajectory sets.
    :param UnstructuredSettings unstructured: How its unstructured trajectory sets are sampled.
    """
    alpha_lon: LongitudinalAccelerations
    """Its longitudinal limits."""
    alpha_lat: LateralAccelerations
    """Its lateral limits."""
    response_time: float
    """Seconds before it applies a proper response."""
    lateral_fluctuation_margin: float = 0.0
    """Metres it may sway sideways; half of each party's adds to the lateral safe distance."""
    max_speed_on_acceleration: float = 100.0
    """m/s: the speed it may reach accelerating in its response time."""
    min_longitudinal_safety_distance: float = 0.0
    """Metres added to its longitudinal safe distance, and to the ego's front in its trajectory sets."""
    unstructured: UnstructuredSettings = DEFAULT_UNSTRUCTURED
    """How its unstructured trajectory sets are sampled."""


VEHICLE = Dynamics(LongitudinalAccelerations(3.5, -8.0, -4.0, -3.0), LateralAccelerations(0.2, -0.8),
                   response_time=2.0, lateral_fluctuation_margin=0.1)
"""The canonical set (ad-rss-lib tests and docs, CARLA, Apollo's rss_decider) with the others' 2 s response."""
PEDESTRIAN = Dynamics(LongitudinalAccelerations(2.0, -4.0, -2.0, -2.0), LateralAccelerations(0.001, -0.001),
                      response_time=0.5, lateral_fluctuation_margin=0.1, max_speed_on_acceleration=10.0)
"""CARLA's pedestrian set (LibCarla RssCheck.cpp:107-125), the only one shipped."""


@dataclass(frozen=True)
class RssParameters:
    """
    The dynamics of each kind of road user.

    :param Dynamics ego: The ego.
    :param Dynamics vehicle: Other vehicles.
    :param Dynamics bicycle: Bicycles.
    :param Dynamics pedestrian: Pedestrians.
    :param float speed_limit_factor: An object's ``max_speed_on_acceleration`` is its lanes' speed limit times
        this (CARLA's IncreasedSpeedLimit10); 0 keeps the dynamics' value.
    """
    ego: Dynamics = replace(VEHICLE, response_time=1.0)
    """The ego."""
    vehicle: Dynamics = VEHICLE
    """Other vehicles."""
    bicycle: Dynamics = VEHICLE
    """Bicycles."""
    pedestrian: Dynamics = PEDESTRIAN
    """Pedestrians."""
    speed_limit_factor: float = 1.1
    """
    An object's ``max_speed_on_acceleration`` is its lanes' speed limit times this (CARLA's
    IncreasedSpeedLimit10); 0 keeps the dynamics' value.
    """


CANONICAL = RssParameters()
"""The spec's parameters: ego response 1 s, other vehicles and bicycles 2 s, pedestrians CARLA's set."""


@dataclass(frozen=True)
class Velocity:
    """
    Speed ranges along and across the constellation's road; lateral speed is positive to the right.

    :param float speed_lon_min: m/s along the road, the least; not negative for valid input.
    :param float speed_lon_max: m/s along the road, the most.
    :param float speed_lat_min: m/s across the road, the least.
    :param float speed_lat_max: m/s across the road, the most.
    """
    speed_lon_min: float = 0.0
    """m/s along the road, the least; not negative for valid input."""
    speed_lon_max: float = 100.0
    """m/s along the road, the most."""
    speed_lat_min: float = -100.0
    """m/s across the road, the least."""
    speed_lat_max: float = 100.0
    """m/s across the road, the most."""


@dataclass(frozen=True)
class StructuredState:
    """
    Lane-relative velocity, priority and intersection entry/exit distances in metres.

    :param Velocity velocity: Its speed ranges in the constellation's lane coordinates.
    :param bool has_priority: At an intersection, whether it has priority.
    :param bool is_in_correct_lane: Whether it drives in a lane of its own direction.
    :param float distance_to_enter_intersection: Metres until its front enters the conflict area; 0 inside.
    :param float distance_to_leave_intersection: Metres until its rear has left the conflict area.
    """
    velocity: Velocity = Velocity()
    """Its speed ranges in the constellation's lane coordinates."""
    has_priority: bool = False
    """At an intersection, whether it has priority."""
    is_in_correct_lane: bool = True
    """Whether it drives in a lane of its own direction."""
    distance_to_enter_intersection: float = 0.0
    """Metres until its front enters the conflict area; 0 inside."""
    distance_to_leave_intersection: float = 1000.0
    """Metres until its rear has left the conflict area."""


@dataclass(frozen=True)
class UnstructuredObjectState:
    """
    A road user in the plane (``world::ObjectState``).

    :param float x: Body centre, map frame.
    :param float y: Body centre.
    :param float yaw: Heading, radians.
    :param float length: Metres, along the heading.
    :param float width: Metres, across it.
    :param float speed_min: m/s, the least.
    :param float speed_max: m/s, the most.
    :param float yaw_rate: rad/s, counter-clockwise.
    :param float steering_angle: Radians it steers off its heading; the record has none, so 0.
    """
    x: float
    """Body centre, map frame."""
    y: float
    """Body centre."""
    yaw: float
    """Heading, radians."""
    length: float
    """Metres, along the heading."""
    width: float
    """Metres, across it."""
    speed_min: float
    """m/s, the least."""
    speed_max: float
    """m/s, the most."""
    yaw_rate: float = 0.0
    """rad/s, counter-clockwise."""
    steering_angle: float = 0.0
    """Radians it steers off its heading; the record has none, so 0."""


@dataclass(frozen=True)
class RelativeObjectState:
    """
    One road user's type, dynamics and structured or unstructured state for a relative constellation.

    :param ObjectType object_type: Its type; ``EGO_VEHICLE`` for the party in the ego's role.
    :param Dynamics dynamics: Its dynamics.
    :param StructuredState structured: Its state in lane coordinates, for structured constellations.
    :param UnstructuredObjectState | None unstructured: Its state in the plane, for unstructured
        constellations; None otherwise.
    """
    object_type: ObjectType
    """Its type; ``EGO_VEHICLE`` for the party in the ego's role."""
    dynamics: Dynamics
    """Its dynamics."""
    structured: StructuredState = StructuredState()
    """Its state in lane coordinates, for structured constellations."""
    unstructured: UnstructuredObjectState | None = None
    """Its state in the plane, for unstructured constellations; None otherwise."""


@dataclass(frozen=True)
class RelativeConstellation:
    """
    The ego (object a) and another road user (object b) in one constellation.

    :param int constellation_id: The id the checkers keep its history by.
    :param ConstellationType constellation_type: Its type.
    :param RelativeObjectState ego: The party in the ego's role.
    :param RelativeObjectState other: The other.
    :param LonPosition lon_position: Where the ego is along the road relative to the other.
    :param float lon_distance: Metres between them along the road; 0 overlapping.
    :param LatPosition lat_position: Where the ego is across the road relative to the other.
    :param float lat_distance: Metres between them across the road; 0 overlapping.
    :param int ego_id: The ego's road-user id; -1 for the recorded ego.
    :param int object_id: The other's road-user id.
    """
    constellation_id: int
    """The id the checkers keep its history by."""
    constellation_type: ConstellationType
    """Its type."""
    ego: RelativeObjectState
    """The party in the ego's role."""
    other: RelativeObjectState
    """The other."""
    lon_position: LonPosition = LonPosition.OVERLAP
    """Where the ego is along the road relative to the other."""
    lon_distance: float = 0.0
    """Metres between them along the road; 0 overlapping."""
    lat_position: LatPosition = LatPosition.OVERLAP
    """Where the ego is across the road relative to the other."""
    lat_distance: float = 0.0
    """Metres between them across the road; 0 overlapping."""
    ego_id: int = 0
    """The ego's road-user id; -1 for the recorded ego."""
    object_id: int = 1
    """The other's road-user id."""


@dataclass
class LonState:
    """
    Longitudinal safety, required response, evaluator and current/safe distances in metres.

    :param bool is_safe: Whether the longitudinal distance is safe.
    :param LonResponse response: The proper response owed.
    :param Evaluator evaluator: The rule that decided.
    :param float current_distance: Metres, the distance the rule measured.
    :param float safe_distance: Metres, the safe distance the rule asks for.
    """
    is_safe: bool = False
    """Whether the longitudinal distance is safe."""
    response: LonResponse = LonResponse.BRAKE_MIN
    """The proper response owed."""
    evaluator: Evaluator = Evaluator.NONE
    """The rule that decided."""
    current_distance: float = 1e9
    """Metres, the distance the rule measured."""
    safe_distance: float = 1e9
    """Metres, the safe distance the rule asks for."""


@dataclass
class LatState:
    """
    Lateral safety, required response, evaluator and current/safe distances in metres.

    :param bool is_safe: Whether the lateral distance on this side is safe.
    :param LatResponse response: The proper response owed.
    :param Evaluator evaluator: The rule that decided.
    :param float current_distance: Metres, the distance the rule measured.
    :param float safe_distance: Metres, the safe distance the rule asks for.
    """
    is_safe: bool = False
    """Whether the lateral distance on this side is safe."""
    response: LatResponse = LatResponse.BRAKE_MIN
    """The proper response owed."""
    evaluator: Evaluator = Evaluator.NONE
    """The rule that decided."""
    current_distance: float = 1e9
    """Metres, the distance the rule measured."""
    safe_distance: float = 1e9
    """Metres, the safe distance the rule asks for."""


@dataclass
class UnstructuredResult:
    """
    Unstructured safety, required response and allowed heading interval in radians.

    :param bool is_safe: Whether it is safe: its response is ``CONTINUE_FORWARD``.
    :param UnstructuredResponse response: The proper response owed.
    :param Tuple[float, float] heading_range: ``(begin, end)``, the headings driving away may take, wrapping
        through pi when ``end < begin``; the library's default ``(0, 2 pi)``, unread, unless driving away.
    """
    is_safe: bool = False
    """Whether it is safe: its response is ``CONTINUE_FORWARD``."""
    response: UnstructuredResponse = UnstructuredResponse.BRAKE
    """The proper response owed."""
    heading_range: Tuple[float, float] = (0.0, 6.283185307179586)
    """
    ``(begin, end)``, the headings driving away may take, wrapping through pi when ``end < begin``; the
    library's default ``(0, 2 pi)``, unread, unless driving away.
    """


@dataclass
class RssState:
    """
    One checker's result for one constellation and frame (``state::RssState``).

    :param int constellation_id: The constellation's id.
    :param ConstellationType constellation_type: Its type.
    :param LonState lon: The longitudinal result, for structured constellations.
    :param LatState lat_left: The lateral result towards the ego's left.
    :param LatState lat_right: The lateral result towards its right.
    :param UnstructuredResult unstructured: The result for unstructured constellations.
    """
    constellation_id: int
    """The constellation's id."""
    constellation_type: ConstellationType
    """Its type."""
    lon: LonState = field(default_factory=LonState)
    """The longitudinal result, for structured constellations."""
    lat_left: LatState = field(default_factory=LatState)
    """The lateral result towards the ego's left."""
    lat_right: LatState = field(default_factory=LatState)
    """The lateral result towards its right."""
    unstructured: UnstructuredResult = field(default_factory=UnstructuredResult)
    """The result for unstructured constellations."""


def is_longitudinal_safe(state: RssState) -> bool:
    """
    Return whether the longitudinal component of the RSS state is safe.

    :param RssState state: A checker's result.
    :returns: True if ``state.lon`` is safe.
    :rtype: bool
    """
    return state.lon.is_safe


def is_lateral_safe(state: RssState) -> bool:
    """
    Return whether both lateral components of the RSS state are safe.

    :param RssState state: A checker's result.
    :returns: True if ``state.lat_left`` and ``state.lat_right`` are both safe.
    :rtype: bool
    """
    return state.lat_left.is_safe and state.lat_right.is_safe


def is_dangerous(state: RssState) -> bool:
    """
    Unstructured: not safe; structured: longitudinally and laterally unsafe (RssStateOperation.hpp:73-82).

    :param RssState state: A checker's result.
    :returns: True if dangerous: an unsafe unstructured result, or a structured one unsafe both ways.
    :rtype: bool
    """
    if state.constellation_type == ConstellationType.UNSTRUCTURED:
        return not state.unstructured.is_safe
    return not is_longitudinal_safe(state) and not is_lateral_safe(state)
