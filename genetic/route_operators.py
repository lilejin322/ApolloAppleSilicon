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
from math import inf, isfinite
from random import Random
from typing import Iterable

from ..scenario.routing import InvalidRoute, LaneRoute, RouteSpace, Transition, lane_changes
from .common import OperatorConfig, gauss_step

BRIDGE_LENGTH_SLACK = 0.1
"""Share of the longer parent by which a bridged child's target length may leave the parents' range."""
BRIDGE_BOUND_SLACK = 0.25
"""With ``bridge_within_parents``: share of the longer parent by which a bridged child's
length may leave the parents' range. 0.1 shrank sunnyvale_loop bicycles (median 173 ->
143 m in 30 generations); unbounded, shalun bicycles grew 150 -> 190 m."""
BRIDGE_TRIALS = 6
"""Lane pairs tried per bridge, nearest estimated length first."""

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

    The candidate keeps the first start s and the second end s. Of the shared
    lanes, the join whose child length is nearest to a target drawn uniformly
    between the parents' lengths is used, so children blend their parents'
    lengths. A uniform choice made children grow (borregas_ave NPCs: +20 m per
    accepted child). Long parents share more lanes, and children shorter than
    ``min_length`` were rejected more often than ones longer than ``max_length``.

    :returns: The child, or None if the routes share no lane that joins without repeats.
    """
    a, b = first, second
    positions = {lane: index for index, lane in enumerate(b.lanes)}
    options = [(i, positions[lane]) for i, lane in enumerate(a.lanes)
               if lane in positions and _disjoint(a.lanes[:i], b.lanes[positions[lane]:])]
    if not options:
        return None
    head, tail = _cumulative(space, a, None), _cumulative(space, b, None)
    low, high = sorted((head[-1], tail[-1]))
    target = rng.uniform(low, high)
    i, j = min(options, key=lambda option: (abs(head[option[0]] + tail[-1] - tail[option[1]] - target), rng.random()))
    return _route(a.lanes[:i] + b.lanes[j:], a.transitions[:i] + b.transitions[j:], a.start_s, b.end_s)


def cx_bridge(first: LaneRoute, second: LaneRoute, space: RouteSpace, rng: Random,
              config: OperatorConfig = OperatorConfig(), *,
              lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Join ``first``'s head to ``second``'s tail through a lane path between them.

    A lane of ``first`` and a lane of ``second`` reachable from it within
    ``config.max_length`` are joined by one of the cheapest lane paths between
    them with the fewest lane changes. Routes need not meet. The child's length
    is a blend of the parents': a target is drawn uniformly between their lengths
    (widened by ``BRIDGE_LENGTH_SLACK``) and, of up to ``BRIDGE_TRIALS`` lane
    pairs whose estimated child length is nearest to it, the child actually
    nearest is kept (only within the parents' range widened by
    ``BRIDGE_BOUND_SLACK`` if ``config.bridge_within_parents``). Uniform pairs made bridged routes grow (shalun ego: +129 m
    per child), so lengths crept towards ``max_length`` over generations.

    ``lane_change_s`` describes the first route's lane changes.

    :returns: The child, or None if no lane pair is reachable or every bridge
        would repeat a lane.
    """
    a, b = first, second
    targets = {lane: index for index, lane in enumerate(b.lanes)}
    entries = _entries(space, a, lane_change_s)
    costs = {}
    for i, source in enumerate(a.lanes):
        for lane, cost in space.reachable(source, config.max_length, entry_s=entries[i]).items():
            if lane in targets:
                costs[i, targets[lane]] = cost
    if not costs:
        return None
    head, tail = _cumulative(space, a, lane_change_s), _cumulative(space, b, None)
    low, high = sorted((head[-1], tail[-1]))
    spread = BRIDGE_LENGTH_SLACK * high
    bound = BRIDGE_BOUND_SLACK * high
    target = rng.uniform(max(low - spread, 0.0), high + spread)
    estimates = {(i, j): head[i] + cost + tail[-1] - tail[j] for (i, j), cost in costs.items()}
    pairs = sorted(estimates, key=lambda pair: (abs(estimates[pair] - target), rng.random()))
    best, best_gap = None, inf
    for i, j in pairs[:BRIDGE_TRIALS]:
        paths = _paths(space, config, a.lanes[i], b.lanes[j], entries[i], a.lanes[:i], b.lanes[j + 1:])
        if not paths:
            continue
        child = _splice(a, i, rng.choice(_fewest_extra_changes(paths, 0)), b, j)
        if child is None:
            continue
        try:
            gap = abs(space.length(child) - target)
        except InvalidRoute:
            continue
        if config.bridge_within_parents and not max(low - bound, 0.0) <= space.length(child) <= high + bound:
            continue
        if gap < best_gap:
            best, best_gap = child, gap
    return best


def mut_reroute(route: LaneRoute, space: RouteSpace, rng: Random,
                config: OperatorConfig = OperatorConfig(), *,
                lane_change_s: tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Replace the stretch between two lanes of the route with another lane path
    that has as many lane changes as the stretch it replaces and is at most
    ``config.max_detour`` longer than the shortest of the stretch and its
    alternatives (real routes are near-shortest; unbounded detours made routes
    loop and grow over generations).

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
    # Keep the stretch's number of lane changes: lane changes are
    # mut_lane_change's gene, and alternatives between two lanes of one road
    # usually zigzag (out and back), which piled up lane changes over generations.
    current_changes = lane_changes(route.transitions[i:j])
    paths = [path for path in paths if lane_changes(path[1]) == current_changes]
    if not paths:
        return None
    if isfinite(config.max_detour):
        legs = space.legs(route, lane_change_s=lane_change_s)
        stretch = (sum(leg.end_s - leg.start_s for leg in legs[i:j + 1])
                   + current_changes * space.rules.min_lane_change_distance)
        base = space.length(route, lane_change_s=lane_change_s) - stretch
        lengths = []
        for path in paths:
            child = _splice(route, i, path, route, j)
            try:
                lengths.append(space.length(child) - base if child is not None else inf)
            except InvalidRoute:
                lengths.append(inf)
        shortest = min([stretch, *lengths])
        paths = [path for path, length in zip(paths, lengths) if length <= (1 + config.max_detour) * shortest + 1e-6]
        if not paths:
            return None
    return _splice(route, i, rng.choice(paths), route, j)


def mut_extend_or_truncate(route: LaneRoute, space: RouteSpace, rng: Random,
                           config: OperatorConfig = OperatorConfig(), *,
                           direction: str | None = None) -> LaneRoute | None:
    """
    Extend or truncate the route at its start or end.

    Extensions add one to three lanes after the end, or one or two before
    the start, along any legal step; truncations drop as many (one to three
    lanes at the end, one or two at the start), so neither direction is
    favoured.
    The new end lane gets a random s.

    ``direction`` ("extend" or "truncate") is drawn 50/50 here if None. Callers
    that retry until a child changes must draw it once per mutation event and
    pass it to every retry. Extensions fail far more often (map edges, endpoint
    rules: san_mateo ego 30% accepted vs truncations 73%), and choosing anew on
    each retry shrank routes (243 -> 158 m in 10 neutral generations).

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
    if direction is None:
        direction = rng.choice(("extend", "truncate"))
    elif direction not in ("extend", "truncate"):
        raise ValueError(f"direction must be 'extend' or 'truncate', not {direction!r}")
    actions = [action for action in actions if action.startswith(direction)]
    if not actions:
        return None

    start_s, end_s = route.start_s, route.end_s
    action = rng.choice(actions)
    # Truncations drop as many lanes as extensions add (1-3 at the end, 1-2 at
    # the start). Keeping a uniformly random prefix dropped half the route on
    # average: most such children fell below min_length and were rejected,
    # while extensions were accepted, so routes grew over generations.
    if action == "truncate_end":
        k = _truncation_index(space, rng, lanes[:-1], from_end=True, most=3)
        if k is None:
            return None
        lanes, transitions = lanes[:k + 1], transitions[:k]
        end_s = space.random_endpoint(lanes[-1], rng)
    elif action == "truncate_start":
        k = _truncation_index(space, rng, lanes[1:], from_end=False, most=2)
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
                    lane_change_s: tuple[float, ...] | None = None,
                    acceptance: float | None = None) -> LaneRoute | None:
    """
    Insert a lane change into the route, or remove one.

    An insertion moves to a neighbouring lane and rejoins the route later
    through a lane path, or else drives on forward from the neighbouring lane.
    A removal replaces a stretch containing a lane change with a path that has
    fewer, or else drives on forward from before the lane change. Driving on
    covers about the distance the route had left: ending right there silently
    truncated routes (up to ~100 m per child).

    A child with ``k`` more lane changes is kept only if ``acceptance`` (a
    uniform draw, made once per mutation event by the caller so that retries
    cannot redraw it; drawn here if None) is below ``config.lane_change_odds ** k``.
    Insertions apply to almost every route and removals only to routes with a
    lane change, and callers retry until a child changes. Without this
    acceptance, lane changes piled up (shalun NPCs: 37% -> 76% of routes with
    one in 30 neutral generations).

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
    if acceptance is None:
        acceptance = rng.random()
    if rng.choice(actions) == "insert":
        child = _insert_lane_change(route, space, rng, config, *rng.choice(inserts),
                                    lane_change_s=lane_change_s)
    else:
        child = _remove_lane_change(route, space, rng, config, rng.choice(removals),
                                    lane_change_s=lane_change_s)
    if child is not None:
        added = child.lane_changes - route.lane_changes
        if added > 0 and acceptance >= config.lane_change_odds ** added:
            return None
    return child


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
    # No way back: stay on the neighbouring side and drive on forward about as
    # far as the route did (ending right on the neighbour truncated it).
    remaining = _cumulative(space, route, lane_change_s)[-1] - _cumulative(space, route, lane_change_s)[i]
    tail, steps, end_s = _drive_on(space, rng, lanes[:i + 1] + (neighbor,), remaining)
    if end_s is None:
        return None
    return _route(tail, transitions[:i] + (step,) + steps, route.start_s, end_s)


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
    # No way around it: drive on forward from before the lane change about as
    # far as the route did (ending before it truncated the route).
    remaining = _cumulative(space, route, lane_change_s)[-1] - _cumulative(space, route, lane_change_s)[i]
    tail, steps, end_s = _drive_on(space, rng, lanes[:i + 1], remaining)
    if end_s is None:
        return None
    return _route(tail, transitions[:i] + steps, route.start_s, end_s)


def _drive_on(space: RouteSpace, rng: Random, lanes, remaining: float):
    """
    ``lanes`` continued along forward successors until about ``remaining``
    metres from the start of their last lane, the added steps, and an allowed
    end s near that distance (None if the final lane has none).
    """
    lanes, steps = list(lanes), []
    while remaining > space.lane_length(lanes[-1]):
        options = sorted(lane for lane in space.graph.successors(lanes[-1])
                         if lane not in lanes and space.transition(lanes[-1], lane) is Transition.FORWARD)
        if not options:
            break
        remaining -= space.lane_length(lanes[-1])
        lanes.append(rng.choice(options))
        steps.append(Transition.FORWARD)
    s = min(max(remaining, 0.0), space.lane_length(lanes[-1]))
    end_s = space.nearest_endpoint(lanes[-1], s) if space.restricts_endpoints else s
    return tuple(lanes), tuple(steps), end_s


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


def _truncation_index(space: RouteSpace, rng: Random, lanes, *, from_end: bool, most: int) -> int | None:
    """
    Index of the lane a truncated route ends (``from_end``) or starts on, among
    ``lanes`` (the route without its last or first lane): within ``most`` lanes of
    the cut end. With endpoint restrictions, only lanes an endpoint may lie on;
    if none is that close, the closest one, as extensions add lanes to reach one.
    """
    allowed = [index for index, lane in enumerate(lanes)
               if not space.restricts_endpoints or space.endpoint_ranges(lane)]
    if not allowed:
        return None
    reach = range(max(len(lanes) - most, 0), len(lanes)) if from_end else range(min(most, len(lanes)))
    near = [index for index in allowed if index in reach]
    if near:
        return rng.choice(near)
    return max(allowed) if from_end else min(allowed)


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


def _cumulative(space: RouteSpace, route: LaneRoute, lane_change_s: tuple[float, ...] | None) -> list[float]:
    """
    Driven distance to the entry of each lane of the route, then its total
    (``RouteSpace.length``). Where the lane-change positions are unknown or do
    not fit, whole lanes are counted, as an estimate.
    """
    try:
        stretches = [leg.end_s - leg.start_s for leg in space.legs(route, lane_change_s=lane_change_s)]
    except InvalidRoute:
        stretches = [space.service.get_lane_by_id(lane).length for lane in route.lanes]
        stretches[-1] = route.end_s
        stretches[0] -= route.start_s
    marks, distance = [], 0.0
    for index, stretch in enumerate(stretches):
        marks.append(distance)
        distance += stretch
        if index < len(route.transitions) and route.transitions[index] is not Transition.FORWARD:
            distance += space.rules.min_lane_change_distance
    return marks + [distance]


def _fewest_extra_changes(paths: list[Path], changes: int) -> list[Path]:
    """The paths whose lane-change count is closest to ``changes``."""
    gaps = [abs(lane_changes(steps) - changes) for _, steps in paths]
    return [path for path, gap in zip(paths, gaps) if gap == min(gaps)]


def _splice(head: LaneRoute, i: int, path: Path, tail: LaneRoute, j: int) -> LaneRoute | None:
    """``head`` before its lane ``i``, then ``path`` from that lane to ``tail``'s
    lane ``j``, then ``tail`` after it; ``head``'s start s and ``tail``'s end s."""
    lanes, steps = path
    return _route(head.lanes[:i] + lanes + tail.lanes[j + 1:], head.transitions[:i] + steps + tail.transitions[j:],
                  head.start_s, tail.end_s)
