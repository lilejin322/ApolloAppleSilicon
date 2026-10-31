"""
Shared pieces of route-aware ADAgent crossover and mutation.

Operators change a route's lanes, transitions, and positions through the
agent's ``RouteSpace``. ``on_route`` constructs proposals; ``acceptable``
checks route validity, driven-length bounds, and the start-time limit.
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar, Mapping, Tuple

from ...scenario import ADAgent, LaneRoute, RouteSpace
from ..common import LaneRouteConfig


@dataclass(frozen=True)
class ADOperatorConfig(LaneRouteConfig):
    """
    Bounds, weights, and step sizes for ADAgent crossover and mutation.

    Bounds, step sizes, and lane-route search limits are inherited from ``LaneRouteConfig``. Weight keys
    must belong to ``CROSSOVERS`` or ``MUTATIONS``; omitted or zero-weight
    operators are disabled. Positive weights control selection order, with
    up to ``attempts`` tries per selected operator.

    :param float min_length: Minimum accepted driven route length in metres; defaults to 50.
    :param float max_length: Maximum accepted driven route length and lane-path search
        distance in metres; defaults to 500. Must be finite and at least ``min_length``.
    :param float max_start_t: Latest accepted start time in scenario seconds; defaults to 10.
    :param int attempts: Maximum tries per selected operator; defaults to 5, minimum 1.
    :param Mapping[str, float] crossover_weights: Relative selection weights for
        ``shared_lane``, ``bridge``, and ``start_time``; defaults to 3, 2, and 1.
        Omitted or zero-weight operators are disabled.
    :param Mapping[str, float] mutation_weights: Relative selection weights for
        ``reroute``, ``extend_or_truncate``, ``lane_change``, ``s``, and ``start_t``;
        defaults to 3, 2, 2, 2, and 1. Omitted or zero-weight operators are disabled.
    :param float sigma_s: Gaussian endpoint-step standard deviation in metres;
        defaults to 10. Steps are clamped to the endpoint lane.
    :param float sigma_t: Gaussian start-time-step standard deviation in seconds;
        defaults to 1. Steps are clamped to ``[0, max_start_t]``.
    :param int bridge_candidates: Number of cheapest lane paths considered per bridge
        search; defaults to 5, minimum 1. Rerouting requests one extra path to
        allow for excluding the current stretch.
    :param float max_detour: Maximum relative extra length of a rerouted stretch
        over the shortest of the original stretch and its alternatives;
        defaults to 0.3. Nonnegative; ``inf`` disables this restriction.
    :param float lane_change_odds: Acceptance factor in ``[0, 1]`` for additional
        lane changes; defaults to 0.25. A proposal adding ``k`` lane changes is
        retained with probability ``lane_change_odds ** k``.
    :param bool bridge_within_parents: Restrict bridged route lengths to the parents'
        length range widened by ``route_operators.BRIDGE_BOUND_SLACK`` times
        the longer parent's length; defaults to False.
    :raises ValueError: If inherited validation rejects length bounds, nonfinite
        or negative time/step values, retry/search counts below 1, negative
        detour limits, lane-change odds outside ``[0, 1]``, unknown operator
        names, or nonfinite or negative weights.
    """

    CROSSOVERS: ClassVar[Tuple[str, ...]] = ("shared_lane", "bridge", "start_time")
    """Supported crossover names used as keys in ``crossover_weights``."""
    MUTATIONS: ClassVar[Tuple[str, ...]] = ("reroute", "extend_or_truncate", "lane_change", "s", "start_t")
    """Supported mutation names used as keys in ``mutation_weights``."""

    crossover_weights: Mapping[str, float] = field(default_factory=lambda: {
        "shared_lane": 3.0, "bridge": 2.0, "start_time": 1.0,
    })
    """Relative crossover weights; omitted or zero-weight operators are disabled."""
    mutation_weights: Mapping[str, float] = field(default_factory=lambda: {
        "reroute": 3.0, "extend_or_truncate": 2.0, "lane_change": 2.0, "s": 2.0, "start_t": 1.0,
    })
    """Relative mutation weights; omitted or zero-weight operators are disabled."""


def on_route(agent: ADAgent, route: LaneRoute | None) -> ADAgent | None:
    """
    Return a copy of ``agent`` with ``route``, or None if no route was proposed.

    The start time is preserved; route validity and operator bounds are not
    checked here. Use ``acceptable`` to validate the candidate.

    :param ADAgent agent: Source agent whose start time is preserved.
    :param LaneRoute | None route: Proposed lane route, or None when the route operator produced no candidate.
    :returns: A copy with the proposed route, or None if ``route`` is None.
    :rtype: ADAgent | None
    """
    return None if route is None else replace(agent, route=route)


def acceptable(space: RouteSpace, agent: ADAgent, config: ADOperatorConfig) -> bool:
    """
    Check route validity, inclusive driven-length bounds, and the start-time limit.

    Lengths are in metres and ``max_start_t`` is in seconds. ``ADAgent``
    already enforces a finite, nonnegative start time at construction.

    :param RouteSpace space: Ego route space providing lane connectivity, geometry, and validity rules.
    :param ADAgent agent: Candidate whose route, driven length, and start time are checked.
    :param ADOperatorConfig config: Inclusive minimum/maximum driven length and maximum start time.
    :returns: True if all route, length, and time checks pass; otherwise False.
    :rtype: bool
    """
    return (agent.start_t <= config.max_start_t and space.is_valid(agent.route)
            and config.min_length <= space.length(agent.route) <= config.max_length)
