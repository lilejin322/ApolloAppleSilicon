"""
Ego (ADS) chromosomes.

An ``ADAgent`` is plain data: the lane route to drive and when to start. It
uses ``scenario.routing`` to build its routing request, which
``manager.routing`` sends to Apollo.
"""

from dataclasses import dataclass
from random import Random

from .routing import ENDPOINT_JUNCTION_MARGIN, LaneRoute, RoutePoses, RouteRules, RouteSpace
from ..utils import require_nonnegative
from typing import Dict


@dataclass(frozen=True)
class ADAgent:
    """
    Genetic representation of one ego ADS instance.

    The route starts at ``(route.lanes[0], route.start_s)`` and ends at
    ``(route.lanes[-1], route.end_s)``. The routing request is sent
    ``start_t`` seconds after the scenario starts.

    :param LaneRoute route: Lanes to drive, with start and end positions.
    :param float start_t: Seconds after scenario start to send the route.
    """

    route: LaneRoute
    """Lanes to drive, with start and end positions."""
    start_t: float
    """Seconds after scenario start to send the route."""

    def __post_init__(self) -> None:
        """
        Validate the field values and consistency of the constructed object.

        :returns: ``None`` if validation succeeds.
        :rtype: None
        """
        require_nonnegative(start_t=self.start_t)

    @staticmethod
    def rules(*, min_turn_radius: float) -> RouteRules:
        """
        Route rules for an ego vehicle: city lanes, no U-turn tighter than it
        can drive, starts and ends away from junctions (``ENDPOINT_JUNCTION_MARGIN``).

        :param float min_turn_radius: The vehicle's minimum turning radius in metres,
            e.g. ``utils.get_vehicle_params(...).min_turn_radius``.
        :returns: Immutable route rules for this agent category.
        :rtype: RouteRules
        """
        return RouteRules(lane_types=frozenset({"CITY_DRIVING"}), min_turn_radius=min_turn_radius,
                          endpoint_junction_margin=ENDPOINT_JUNCTION_MARGIN)

    @property
    def routing_str(self) -> str:
        """
        The route's lanes joined by arrows.

        :returns: The route's lanes joined by arrows.
        :rtype: str
        """
        return "->".join(self.route.lanes)

    def routing_request(self, space: RouteSpace) -> RoutePoses:
        """
        Build the routing request data for ``manager.routing``.

        :param RouteSpace space: The ego's route space, e.g. built with ``ADAgent.rules``.
        :returns: Route poses suitable for sending as the ego routing request.
        :rtype: RoutePoses
        :raises InvalidRoute: If the route is illegal or cannot be sent unambiguously.
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
        :returns: A randomly generated valid agent satisfying the requested constraints.
        :rtype: ADAgent
        :raises ValueError: If the sampled start time violates the agent's field constraints.
        """
        route = space.random_route(rng, min_length=min_length, max_length=max_length,
                                   lane_change_probability=lane_change_probability)
        return cls(route, rng.uniform(0, max_start_t))

    def to_dict(self) -> Dict:
        """
        Return a JSON-compatible representation.

        :returns: A new JSON-compatible mapping of the object fields.
        :rtype: Dict
        """
        return {"route": self.route.to_dict(), "start_t": self.start_t}

    @classmethod
    def from_dict(cls, data: Dict) -> "ADAgent":
        """
        Build an agent from ``to_dict`` output.

        :param Dict data: Serialized mapping previously produced by ``to_dict``.
        :returns: The reconstructed object.
        :rtype: ADAgent
        :raises KeyError: If a required serialized field is absent.
        """
        return cls(LaneRoute.from_dict(data["route"]), data["start_t"])
