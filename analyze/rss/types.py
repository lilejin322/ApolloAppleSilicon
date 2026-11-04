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
    """RSS pairwise traffic configurations, including priority-specific intersection cases."""
    NOT_RELEVANT = 0
    SAME_DIRECTION = 1
    OPPOSITE_DIRECTION = 2
    INTERSECTION_EGO_HAS_PRIORITY = 3
    INTERSECTION_OBJECT_HAS_PRIORITY = 4
    INTERSECTION_SAME_PRIORITY = 5
    UNSTRUCTURED = 6

    @property
    def is_intersection(self) -> bool:
        """Whether this is one of the three intersection types."""
        return self in (ConstellationType.INTERSECTION_EGO_HAS_PRIORITY,
                        ConstellationType.INTERSECTION_OBJECT_HAS_PRIORITY,
                        ConstellationType.INTERSECTION_SAME_PRIORITY)


class ObjectType(IntEnum):
    """RSS road-user categories used to select dynamics and response rules."""
    INVALID = 0
    EGO_VEHICLE = 1
    OTHER_VEHICLE = 2
    PEDESTRIAN = 3
    ARTIFICIAL_OBJECT = 4
    ARTIFICIAL_VEHICLE = 5
    ARTIFICIAL_PEDESTRIAN = 6
    BICYCLE = 7
    OTHER_OBJECT = 8


class LonPosition(IntEnum):
    """Where the ego (object a) is along the road relative to the other (object b)."""

    IN_FRONT = 0
    OVERLAP_FRONT = 1
    OVERLAP = 2
    OVERLAP_BACK = 3
    AT_BACK = 4


class LatPosition(IntEnum):
    """Where the ego is across the road relative to the other."""

    AT_LEFT = 0
    OVERLAP_LEFT = 1
    OVERLAP = 2
    OVERLAP_RIGHT = 3
    AT_RIGHT = 4


class LonResponse(IntEnum):
    """Longitudinal proper responses, distinguishing correct-lane braking from ordinary minimum braking."""
    NONE = 0
    BRAKE_MIN_CORRECT = 1
    BRAKE_MIN = 2


class LatResponse(IntEnum):
    """Lateral proper responses: no restriction or minimum lateral braking."""
    NONE = 0
    BRAKE_MIN = 1


class UnstructuredResponse(IntEnum):
    """Unstructured proper responses for reachable-set interactions."""
    NONE = 0
    CONTINUE_FORWARD = 1
    DRIVE_AWAY = 2
    BRAKE = 3


class Evaluator(IntEnum):
    """Identifier of the distance or intersection rule that produced an RSS result."""
    NONE = 0
    LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION_EGO_CORRECT_LANE = 1
    LONGITUDINAL_DISTANCE_OPPOSITE_DIRECTION = 2
    LONGITUDINAL_DISTANCE_SAME_DIRECTION_EGO_FRONT = 3
    LONGITUDINAL_DISTANCE_SAME_DIRECTION_OTHER_IN_FRONT = 4
    LATERAL_DISTANCE = 5
    INTERSECTION_OTHER_PRIORITY_EGO_ABLE_TO_STOP = 6
    INTERSECTION_EGO_PRIORITY_OTHER_ABLE_TO_STOP = 7
    INTERSECTION_EGO_IN_FRONT = 8
    INTERSECTION_OTHER_IN_FRONT = 9
    INTERSECTION_OVERLAP = 10


class IntersectionState(Enum):
    """Which sub-check last found an intersection safe (RssIntersectionConstellationChecker)."""

    NON_PRIO_ABLE_TO_BREAK = 1
    SAFE_LONGITUDINAL_DISTANCE = 2
    NO_TIME_OVERLAP = 3


@dataclass(frozen=True)
class LongitudinalAccelerations:
    """
    Longitudinal acceleration limits and positive braking magnitudes, in metres per second squared.

    :param float accel_max: Maximum acceleration during response time.
    :param float brake_max: Maximum braking magnitude.
    :param float brake_min: Minimum required braking magnitude.
    :param float brake_min_correct: Minimum braking when travelling in the correct lane.
    """
    accel_max: float
    brake_max: float
    brake_min: float
    brake_min_correct: float


@dataclass(frozen=True)
class LateralAccelerations:
    """
    Lateral acceleration and braking magnitudes, in metres per second squared.

    :param float accel_max: Maximum lateral acceleration during response time.
    :param float brake_min: Minimum required lateral braking magnitude.
    """
    accel_max: float
    brake_min: float


@dataclass(frozen=True)
class UnstructuredSettings:
    """Turning bounds and trajectory sampling counts for unstructured reachable sets."""
    pedestrian_turning_radius: float = 2.0
    drive_away_max_angle: float = 2.4
    vehicle_yaw_rate_change: float = 0.3
    vehicle_min_radius: float = 3.5
    vehicle_trajectory_calculation_step: float = 0.2
    vehicle_front_intermediate_yaw_rate_change_ratio_steps: int = 3
    vehicle_back_intermediate_yaw_rate_change_ratio_steps: int = 0
    vehicle_brake_intermediate_acceleration_steps: int = 1
    vehicle_continue_forward_intermediate_acceleration_steps: int = 0
    vehicle_continue_forward_intermediate_yaw_rate_change_ratio_steps: int = 3
    pedestrian_continue_forward_intermediate_heading_change_ratio_steps: int = 3
    pedestrian_continue_forward_intermediate_acceleration_steps: int = 0
    pedestrian_brake_intermediate_acceleration_steps: int = 3
    pedestrian_front_intermediate_heading_change_ratio_steps: int = 4
    pedestrian_back_intermediate_heading_change_ratio_steps: int = 0


DEFAULT_UNSTRUCTURED = UnstructuredSettings()
"""ad-rss-lib's test-support settings (TestSupport.cpp:92-107)."""


@dataclass(frozen=True)
class Dynamics:
    """One road user's RSS dynamics (``world::RssDynamics``)."""

    alpha_lon: LongitudinalAccelerations
    alpha_lat: LateralAccelerations
    response_time: float
    lateral_fluctuation_margin: float = 0.0
    max_speed_on_acceleration: float = 100.0
    min_longitudinal_safety_distance: float = 0.0
    unstructured: UnstructuredSettings = DEFAULT_UNSTRUCTURED

    def with_(self, **changes) -> "Dynamics":
        """A copy with ``changes``."""
        return replace(self, **changes)


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

    ego: Dynamics = VEHICLE.with_(response_time=1.0)
    vehicle: Dynamics = VEHICLE
    bicycle: Dynamics = VEHICLE
    pedestrian: Dynamics = PEDESTRIAN
    speed_limit_factor: float = 1.1


CANONICAL = RssParameters()
"""The spec's parameters: ego response 1 s, other vehicles and bicycles 2 s, pedestrians CARLA's set."""


@dataclass(frozen=True)
class Velocity:
    """Speed ranges along and across the constellation's road; lateral speed is positive to the right."""

    speed_lon_min: float = 0.0
    speed_lon_max: float = 100.0
    speed_lat_min: float = -100.0
    speed_lat_max: float = 100.0


@dataclass(frozen=True)
class StructuredState:
    """Lane-relative velocity, priority and intersection entry/exit distances in metres."""
    velocity: Velocity = Velocity()
    has_priority: bool = False
    is_in_correct_lane: bool = True
    distance_to_enter_intersection: float = 0.0
    distance_to_leave_intersection: float = 1000.0


@dataclass(frozen=True)
class UnstructuredObjectState:
    """A road user in the plane (``world::ObjectState``)."""

    x: float
    y: float
    yaw: float
    length: float
    width: float
    speed_min: float
    speed_max: float
    yaw_rate: float = 0.0
    steering_angle: float = 0.0


@dataclass(frozen=True)
class RelativeObjectState:
    """One road user's type, dynamics and structured or unstructured state for a relative constellation."""
    object_type: ObjectType
    dynamics: Dynamics
    structured: StructuredState = StructuredState()
    unstructured: UnstructuredObjectState | None = None


@dataclass(frozen=True)
class RelativeConstellation:
    """The ego (object a) and another road user (object b) in one constellation."""

    constellation_id: int
    constellation_type: ConstellationType
    ego: RelativeObjectState
    other: RelativeObjectState
    lon_position: LonPosition = LonPosition.OVERLAP
    lon_distance: float = 0.0
    lat_position: LatPosition = LatPosition.OVERLAP
    lat_distance: float = 0.0
    ego_id: int = 0
    object_id: int = 1


@dataclass
class LonState:
    """Longitudinal safety, required response, evaluator and current/safe distances in metres."""
    is_safe: bool = False
    response: LonResponse = LonResponse.BRAKE_MIN
    evaluator: Evaluator = Evaluator.NONE
    current_distance: float = 1e9
    safe_distance: float = 1e9


@dataclass
class LatState:
    """Lateral safety, required response, evaluator and current/safe distances in metres."""
    is_safe: bool = False
    response: LatResponse = LatResponse.BRAKE_MIN
    evaluator: Evaluator = Evaluator.NONE
    current_distance: float = 1e9
    safe_distance: float = 1e9


@dataclass
class UnstructuredResult:
    """Unstructured safety, required response and allowed heading interval in radians."""
    is_safe: bool = False
    response: UnstructuredResponse = UnstructuredResponse.BRAKE
    heading_range: Tuple[float, float] = (0.0, 6.283185307179586)


@dataclass
class RssState:
    """One checker's result for one constellation and frame (``state::RssState``)."""

    constellation_id: int
    constellation_type: ConstellationType
    lon: LonState = field(default_factory=LonState)
    lat_left: LatState = field(default_factory=LatState)
    lat_right: LatState = field(default_factory=LatState)
    unstructured: UnstructuredResult = field(default_factory=UnstructuredResult)


def is_longitudinal_safe(state: RssState) -> bool:
    """Return whether the longitudinal component of the RSS state is safe. :rtype: bool"""
    return state.lon.is_safe


def is_lateral_safe(state: RssState) -> bool:
    """Return whether both lateral components of the RSS state are safe. :rtype: bool"""
    return state.lat_left.is_safe and state.lat_right.is_safe


def is_dangerous(state: RssState) -> bool:
    """Unstructured: not safe; structured: longitudinally and laterally unsafe (RssStateOperation.hpp:73-82)."""
    if state.constellation_type == ConstellationType.UNSTRUCTURED:
        return not state.unstructured.is_safe
    return not is_longitudinal_safe(state) and not is_lateral_safe(state)
