"""
Ego (ADS) chromosomes, after META-V2V's ``ADAgent``.

An ``ADAgent`` is plain data: the lane route to drive and when to start. It
uses ``scenario.routing`` to build its routing request, which
``manager.routing`` sends to Apollo.
"""

from dataclasses import dataclass
from random import Random

from .routing import LaneRoute, RoutePoses, RouteRules, RouteSpace
from ..utils import require_nonnegative


@dataclass(frozen=True)
class ADAgent:
    """
    Genetic representation of one ego ADS instance.

    The instance starts at ``(route.lanes[0], route.start_s)`` and, ``start_t``
    seconds into the scenario, drives along ``route`` to
    ``(route.lanes[-1], route.end_s)``.

    :param LaneRoute route: Lanes to drive, with start and end positions.
    :param float start_t: Seconds after scenario start to send the route.
    """

    route: LaneRoute
    start_t: float

    def __post_init__(self) -> None:
        require_nonnegative(start_t=self.start_t)

    @staticmethod
    def rules(*, min_turn_radius: float) -> RouteRules:
        """
        Route rules for an ego vehicle: city lanes, no U-turn tighter than it can drive.

        :param float min_turn_radius: The vehicle's minimum turning radius in metres,
            e.g. ``utils.get_vehicle_params(...).min_turn_radius``.
        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"CITY_DRIVING"}), min_turn_radius=min_turn_radius)

    @property
    def routing_str(self) -> str:
        """
        The route's lanes joined by arrows.

        :rtype: str
        """
        return "->".join(self.route.lanes)

    def routing_request(self, space: RouteSpace) -> RoutePoses:
        """
        Build the routing request data for ``manager.routing``.

        :param RouteSpace space: The ego's route space, e.g. built with ``ADAgent.rules``.
        :raises InvalidRoute: If the route is illegal or cannot be sent unambiguously.
        :rtype: RoutePoses
        """
        return space.poses(self.route)

    @classmethod
    def random(cls, space: RouteSpace, rng: Random, *, max_start_t: float = 10.0,
               min_length: float = 50.0, max_length: float = 500.0,
               lane_change_probability: float = 0.2) -> "ADAgent":
        """
        Draw a random ego with a valid route and a start time in ``[0, max_start_t]``.

        :param RouteSpace space: The ego's route space.
        :param Random rng: Source of randomness; equal seeds give equal agents.
        :param float max_start_t: Latest start time in seconds.
        :param float min_length: Minimum driven distance in metres.
        :param float max_length: Maximum driven distance in metres.
        :param float lane_change_probability: Chance of a lane change where one is possible.
        :rtype: ADAgent
        """
        route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                   lane_change_probability=lane_change_probability)
        return cls(route, rng.uniform(0, max_start_t))

    def to_dict(self) -> dict:
        """
        Return a JSON-compatible representation.

        :rtype: dict
        """
        return {"route": self.route.to_dict(), "start_t": self.start_t}

    @classmethod
    def from_dict(cls, data: dict) -> "ADAgent":
        """
        Build an agent from ``to_dict`` output.

        :rtype: ADAgent
        """
        return cls(LaneRoute.from_dict(data["route"]), data["start_t"])
