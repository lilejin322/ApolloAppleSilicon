"""
LaneRoute crossover and mutation shared by ego and NPC vehicles.

These pure functions propose routes; they do not construct agents or accept
children. The caller supplies a RouteSpace with the vehicle's rules and must
validate the complete child. AD wrappers use their usual route/time bounds;
NPC callers repair lane-change positions with with_route() and check the
resulting trajectory and its length. Pass lane_change_s for searches that
must preserve an NPC prefix's explicit lane-change positions; omission uses
the usual earliest-change entries for AD proposals.
"""

from dataclasses import replace
from random import Random
from typing import Iterable

from ..scenario.routing import LaneRoute, RouteSpace, Transition, lane_changes
from .common import OperatorConfig, gauss_step

__all__ = [
    "cx_shared_lane", "cx_bridge", "mut_reroute", "mut_extend_or_truncate", "mut_lane_change", "mut_s",
]

_EXTRA_LANES = 2
"""Lanes an extension may add beyond its draw to end on a lane where routes may end."""

Path = tuple[tuple[str, ...], tuple[Transition, ...]]
"""Lanes and the transitions between them, as yielded by ``RouteSpace.lane_paths``."""


def cx_shared_lane(first: LaneRoute, second: LaneRoute, space: RouteSpace, rng: Random) -> LaneRoute | None:
    """
    Join the first route's prefix with the second route's suffix at a shared lane.

    The candidate keeps the first start s and the second end s.

    :returns: The child, or None if the routes share no lane that joins without repeats.
    """
    a, b = first, second
    positions = {lane: index for index, lane in enumerate(b.lanes)}
    options = [(i, positions[lane]) for i, lane in enumerate(a.lanes)
               if lane in positions and _disjoint(a.lanes[:i], b.lanes[positions[lane]:])]
    if not options:
        return None
    i, j = rng.choice(options)
    return _route(a.lanes[:i] + b.lanes[j:], a.transitions[:i] + b.transitions[j:], a.start_s, b.end_s)


def cx_bridge(first: LaneRoute, second: LaneRoute, space: RouteSpace, rng: Random,
              config: OperatorConfig = OperatorConfig(), *,
              lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    A lane of ``first`` and a lane of ``second`` reachable from it within
    ``config.max_length`` are chosen; one of the cheapest lane paths between
    them becomes the bridge. Routes need not meet.

    ``lane_change_s`` describes the first route's lane changes.

    :returns: The child, or None if no lane pair is reachable or the bridge
        would repeat a lane.
    """
    a, b = first, second
    targets = {lane: index for index, lane in enumerate(b.lanes)}
    entries = _entries(space, a, lane_change_s)
    pairs = sorted((i, targets[lane]) for i, source in enumerate(a.lanes)
                   for lane in space.reachable(source, config.max_length, entry_s=entries[i])
                   if lane in targets)
    if not pairs:
        return None
    i, j = rng.choice(pairs)
    paths = _paths(space, config, a.lanes[i], b.lanes[j], entries[i], a.lanes[:i], b.lanes[j + 1:])
    if not paths:
        return None
    return _splice(a, i, rng.choice(paths), b, j)


def mut_reroute(route: LaneRoute, space: RouteSpace, rng: Random,
                config: OperatorConfig = OperatorConfig(), *,
                lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Replace the stretch between two lanes of the route with another lane path.

    :returns: The mutant, or None if the route has one lane or no alternative fits.
    """
    lanes = route.lanes
    if len(lanes) < 2:
        return None
    i = rng.randrange(len(lanes) - 1)
    j = rng.randrange(i + 1, len(lanes))
    current = lanes[i:j + 1]
    # One extra candidate, since the current stretch may be among them.
    paths = [path for path in _paths(space, config, lanes[i], lanes[j], _entries(space, route, lane_change_s)[i],
                                     lanes[:i], lanes[j + 1:], k=config.bridge_candidates + 1)
             if path[0] != current]
    if not paths:
        return None
    return _splice(route, i, rng.choice(paths), route, j)


def mut_extend_or_truncate(route: LaneRoute, space: RouteSpace, rng: Random,
                           config: OperatorConfig = OperatorConfig()) -> LaneRoute | None:
    """
    Extend or truncate the route at its start or end.

    Extensions add one to three lanes after the end, or one or two before
    the start, along any legal step; truncations drop lanes from either end.
    The new end lane gets a random s.

    :returns: The mutant, or None if no change applies.
    """
    lanes, transitions = list(route.lanes), list(route.transitions)
    graph = space.graph
    actions = []
    if len(lanes) > 1:
        actions += ["truncate_end", "truncate_start"]
    if any(lane not in lanes for lane in graph.successors(lanes[-1])):
        actions.append("extend_end")
    if any(lane not in lanes for lane in graph.predecessors(lanes[0])):
        actions.append("extend_start")
    if not actions:
        return None

    start_s, end_s = route.start_s, route.end_s
    action = rng.choice(actions)
    if action == "truncate_end":
        k = _endpoint_index(space, rng, lanes[:-1])
        if k is None:
            return None
        lanes, transitions = lanes[:k + 1], transitions[:k]
        end_s = space.random_endpoint(lanes[-1], rng)
    elif action == "truncate_start":
        k = _endpoint_index(space, rng, lanes[1:])
        if k is None:
            return None
        k += 1
        lanes, transitions = lanes[k:], transitions[k:]
        start_s = space.random_endpoint(lanes[0], rng)
    elif action == "extend_end":
        count = rng.randint(1, 3)
        added = 0
        # With endpoint restrictions, a few more lanes may be added to reach one an end may lie on.
        while added < count or (added < count + _EXTRA_LANES and not space.endpoint_ranges(lanes[-1])):
            options = sorted(lane for lane in graph.successors(lanes[-1]) if lane not in lanes)
            if not options:
                break
            lane = rng.choice(options)
            transitions.append(space.transition(lanes[-1], lane))
            lanes.append(lane)
            added += 1
        end_s = space.random_endpoint(lanes[-1], rng)
    else:
        count = rng.randint(1, 2)
        added = 0
        while added < count or (added < count + _EXTRA_LANES and not space.endpoint_ranges(lanes[0])):
            options = sorted(lane for lane in graph.predecessors(lanes[0]) if lane not in lanes)
            if not options:
                break
            lane = rng.choice(options)
            transitions.insert(0, space.transition(lane, lanes[0]))
            lanes.insert(0, lane)
            added += 1
        start_s = space.random_endpoint(lanes[0], rng)
    if start_s is None or end_s is None:
        return None
    return _route(lanes, transitions, start_s, end_s)


def mut_lane_change(route: LaneRoute, space: RouteSpace, rng: Random,
                    config: OperatorConfig = OperatorConfig(), *,
                    lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Insert a lane change into the route, or remove one.

    An insertion moves to a neighbouring lane and rejoins the route later
    through a lane path, or ends on the neighbouring lane. A removal replaces
    a stretch containing a lane change with a path that has fewer, or ends
    the route before the lane change.

    :returns: The mutant, or None if the route can neither gain nor lose a lane change.
    """
    lanes = route.lanes
    inserts = [(i, neighbor) for i, lane in enumerate(lanes)
               for neighbor in sorted(space.graph.successors(lane))
               if neighbor not in lanes and space.transition(lane, neighbor) is not Transition.FORWARD]
    removals = route.lane_change_indices
    actions = (["insert"] if inserts else []) + (["remove"] if removals else [])
    if not actions:
        return None
    if rng.choice(actions) == "insert":
        return _insert_lane_change(route, space, rng, config, *rng.choice(inserts),
                                   lane_change_s=lane_change_s)
    return _remove_lane_change(route, space, rng, config, rng.choice(removals),
                               lane_change_s=lane_change_s)


def _insert_lane_change(route: LaneRoute, space: RouteSpace, rng: Random, config: OperatorConfig,
                        i: int, neighbor: str, *,
                        lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    lanes, transitions = route.lanes, route.transitions
    step = space.transition(lanes[i], neighbor)
    entry = _entries(space, route, lane_change_s)[i]
    if entry + space.rules.min_lane_change_distance > space.lane_length(lanes[i]):
        return None
    if i < len(lanes) - 1:
        j = rng.randrange(i + 1, len(lanes))
        paths = _paths(space, config, neighbor, lanes[j], space.entry_after_change(lanes[i], entry, neighbor),
                       lanes[:i + 1], lanes[j + 1:])
        if paths:
            path, steps = rng.choice(paths)
            return _route(lanes[:i + 1] + path + lanes[j + 1:],
                          transitions[:i] + (step,) + steps + transitions[j:], route.start_s, route.end_s)
    # End on the neighbouring lane instead of rejoining the route.
    end_s = space.random_endpoint(neighbor, rng)
    if end_s is None:
        return None
    return _route(lanes[:i + 1] + (neighbor,), transitions[:i] + (step,), route.start_s, end_s)


def _remove_lane_change(route: LaneRoute, space: RouteSpace, rng: Random, config: OperatorConfig,
                        i: int, *, lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    lanes, transitions = route.lanes, route.transitions
    j = rng.randrange(i + 1, len(lanes))
    current = lane_changes(transitions[i:j])
    paths = [path for path in _paths(space, config, lanes[i], lanes[j], _entries(space, route, lane_change_s)[i],
                                     lanes[:i], lanes[j + 1:])
             if lane_changes(path[1]) < current]
    if paths:
        return _splice(route, i, rng.choice(paths), route, j)
    # No way around it: end the route before the lane change.
    end_s = space.random_endpoint(lanes[i], rng)
    if end_s is None:
        return None
    return _route(lanes[:i + 1], transitions[:i], route.start_s, end_s)


def mut_s(route: LaneRoute, space: RouteSpace, rng: Random,
          config: OperatorConfig = OperatorConfig()) -> LaneRoute | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    Where the space keeps endpoints away from junctions, a step that lands too
    close to one moves to the nearest allowed position on the lane.

    :returns: The mutant, or None if an endpoint's lane has no allowed position.
    :rtype: LaneRoute | None
    """
    start_s, end_s = route.start_s, route.end_s
    which = rng.choice(("start", "end", "both"))
    if which in ("start", "both"):
        start_s = gauss_step(rng, start_s, config.sigma_s, space.lane_length(route.lanes[0]))
        if space.restricts_endpoints:
            start_s = space.nearest_endpoint(route.lanes[0], start_s)
    if which in ("end", "both"):
        end_s = gauss_step(rng, end_s, config.sigma_s, space.lane_length(route.lanes[-1]))
        if space.restricts_endpoints:
            end_s = space.nearest_endpoint(route.lanes[-1], end_s)
    if start_s is None or end_s is None:
        return None
    return replace(route, start_s=start_s, end_s=end_s)


def _endpoint_index(space: RouteSpace, rng: Random, lanes) -> int | None:
    """A random index of ``lanes``; with endpoint restrictions, only of lanes an endpoint may lie on."""
    if not space.restricts_endpoints:
        return rng.randrange(len(lanes))
    options = [index for index, lane in enumerate(lanes) if space.endpoint_ranges(lane)]
    return rng.choice(options) if options else None


def _disjoint(lanes: Iterable[str], *others: Iterable[str]) -> bool:
    lanes = set(lanes)
    return all(lanes.isdisjoint(other) for other in others)


def _route(lanes, transitions, start_s: float, end_s: float) -> LaneRoute | None:
    """Build a route, or None if it would use a lane twice."""
    lanes, transitions = tuple(lanes), tuple(transitions)
    if len(set(lanes)) != len(lanes):
        return None
    return LaneRoute(lanes, transitions, start_s, end_s)


def _entries(space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...] | None) -> list[float]:
    """Where the route enters each of its lanes."""
    return [leg.start_s for leg in space.legs(route, lane_change_s=lane_change_s)]


def _paths(space: RouteSpace, config: OperatorConfig, source: str, target: str, entry_s: float,
           *avoid: Iterable[str], k: int | None = None) -> list[Path]:
    """The ``k`` (default ``config.bridge_candidates``) cheapest lane paths from
    ``source``, entered at ``entry_s``, to ``target`` within ``config.max_length``,
    keeping those that use none of the ``avoid`` lanes."""
    candidates = space.lane_paths(source, target, config.bridge_candidates if k is None else k,
                                  max_cost=config.max_length, entry_s=entry_s)
    return [(lanes, steps) for lanes, steps in candidates if _disjoint(lanes, *avoid)]


def _splice(head: LaneRoute, i: int, path: Path, tail: LaneRoute, j: int) -> LaneRoute | None:
    """``head`` before its lane ``i``, then ``path`` from that lane to ``tail``'s
    lane ``j``, then ``tail`` after it; ``head``'s start s and ``tail``'s end s."""
    lanes, steps = path
    return _route(head.lanes[:i] + lanes + tail.lanes[j + 1:], head.transitions[:i] + steps + tail.transitions[j:],
                  head.start_s, tail.end_s)
