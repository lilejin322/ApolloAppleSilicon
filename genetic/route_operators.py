"""
LaneRoute crossover and mutation shared by ego vehicles, NPC vehicles, and bicycles.

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
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple, TypeAlias
from networkx import DiGraph

from ..scenario.routing import InvalidRoute, LaneRoute, RouteSpace, Transition, lane_changes, RouteLeg
from .common import LaneRouteConfig, OperatorConfig, gauss_step

BRIDGE_LENGTH_SLACK: float = 0.1
"""Share of the longer parent by which a bridged child's target length may leave the parents' range."""
BRIDGE_BOUND_SLACK: float = 0.25
"""With ``bridge_within_parents``: share of the longer parent by which a bridged child's
length may leave the parents' range. 0.1 shrank sunnyvale_loop bicycles (median 173 ->
143 m in 30 generations); unbounded, shalun bicycles grew 150 -> 190 m."""
BRIDGE_TRIALS: int = 6
"""Lane pairs tried per bridge, nearest estimated length first."""

__all__: List[str] = [
    "cx_shared_lane", "cx_bridge", "mut_reroute", "mut_extend_or_truncate", "mut_lane_change", "mut_s",
]

_EXTRA_LANES: int = 2
"""Lanes an extension may add beyond its draw to end on a lane where routes may end."""

Path: TypeAlias = Tuple[Tuple[str, ...], Tuple[Transition, ...]]
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

    :param LaneRoute first: Parent providing the route prefix and start position.
    :param LaneRoute second: Parent providing the route suffix and end position.
    :param RouteSpace space: Route space used to estimate parent and candidate lengths.
    :param Random rng: Random number generator used to draw the target length and break ties.
    :returns: The child, or ``None`` if no shared lane allows joining without
        repeated lanes or the joined route cannot be constructed.
    :rtype: LaneRoute | None
    """
    a: LaneRoute
    b: LaneRoute
    a, b = first, second
    positions: Dict[str, int] = {lane: index for index, lane in enumerate(b.lanes)}
    options: List[Tuple[int, int]] = [(i, positions[lane]) for i, lane in enumerate(a.lanes)
               if lane in positions and _disjoint(a.lanes[:i], b.lanes[positions[lane]:])]
    if not options:
        return None
    head: List[float]
    tail: List[float]
    head, tail = _cumulative(space, a, None), _cumulative(space, b, None)
    low: float
    high: float
    low, high = sorted((head[-1], tail[-1]))
    target: float = rng.uniform(low, high)
    i: int
    j: int
    i, j = min(options, key=lambda option: (abs(head[option[0]] + tail[-1] - tail[option[1]] - target), rng.random()))
    return _route(a.lanes[:i] + b.lanes[j:], a.transitions[:i] + b.transitions[j:], a.start_s, b.end_s)


def cx_bridge(first: LaneRoute, second: LaneRoute, space: RouteSpace, rng: Random,
              config: LaneRouteConfig = LaneRouteConfig(), *,
              lane_change_s: Tuple[float, ...] | None = None) -> LaneRoute | None:
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

    :param LaneRoute first: Parent providing the route head and start position.
    :param LaneRoute second: Parent providing the route tail and end position.
    :param RouteSpace space: Route space used to find connecting paths and evaluate lengths.
    :param Random rng: Random number generator used for target length, tie-breaking, and path selection.
    :param LaneRouteConfig config: Configuration defining reachability distance,
        candidate path count, and optional parental length bounds.
    :param Tuple[float, ...] | None lane_change_s: Lane-change start positions on
        ``first``'s source lanes, in metres, used to estimate lane entries and
        route lengths; ``None`` uses the route space's default placement.
    :returns: The child nearest the target among the evaluated candidates, or
        ``None`` if no lane pair is reachable or no evaluated bridge can be
        constructed, measured, and kept within the enabled parental length bounds.
    :rtype: LaneRoute | None
    """
    a: LaneRoute
    b: LaneRoute
    a, b = first, second
    targets: Dict[str, int] = {lane: index for index, lane in enumerate(b.lanes)}
    entries: List[float] = _entries(space, a, lane_change_s)
    costs: Dict[Tuple[int, int], float] = {}
    for i, source in enumerate(a.lanes):
        for lane, cost in space.reachable(source, config.max_length, entry_s=entries[i]).items():
            if lane in targets:
                costs[i, targets[lane]] = cost
    if not costs:
        return None
    head: List[float]
    tail: List[float]
    head, tail = _cumulative(space, a, lane_change_s), _cumulative(space, b, None)
    low: float
    high: float
    low, high = sorted((head[-1], tail[-1]))
    spread: float = BRIDGE_LENGTH_SLACK * high
    bound: float = BRIDGE_BOUND_SLACK * high
    target: float = rng.uniform(max(low - spread, 0.0), high + spread)
    estimates: Dict[Tuple[int, int], float] = {(i, j): head[i] + cost + tail[-1] - tail[j] for (i, j), cost in costs.items()}
    pairs: List[Tuple[int, int]] = sorted(estimates, key=lambda pair: (abs(estimates[pair] - target), rng.random()))
    best: LaneRoute | None
    best_gap: float
    best, best_gap = None, inf
    for i, j in pairs[:BRIDGE_TRIALS]:
        paths: List[Path] = _paths(space, config, a.lanes[i], b.lanes[j], entries[i], a.lanes[:i], b.lanes[j + 1:])
        if not paths:
            continue
        child: LaneRoute | None = _splice(a, i, rng.choice(_fewest_extra_changes(paths, 0)), b, j)
        if child is None:
            continue
        try:
            gap: float = abs(space.length(child) - target)
        except InvalidRoute:
            continue
        if config.bridge_within_parents and not max(low - bound, 0.0) <= space.length(child) <= high + bound:
            continue
        if gap < best_gap:
            best, best_gap = child, gap
    return best


def mut_reroute(route: LaneRoute, space: RouteSpace, rng: Random,
                config: LaneRouteConfig = LaneRouteConfig(), *,
                lane_change_s: Tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Replace the stretch between two lanes of the route with another lane path
    that has as many lane changes as the stretch it replaces and is at most
    ``config.max_detour`` longer than the shortest of the stretch and its
    alternatives (real routes are near-shortest; unbounded detours made routes
    loop and grow over generations).

    :param LaneRoute route: Route whose randomly selected stretch is replaced.
    :param RouteSpace space: Route space used to find alternatives and measure stretch lengths.
    :param Random rng: Random number generator used to select the stretch and a replacement path.
    :param LaneRouteConfig config: Configuration defining path search limits,
        candidate path count, and the maximum relative detour.
    :param Tuple[float, ...] | None lane_change_s: Lane-change start positions on
        the route's source lanes, in metres, used to determine lane entries and
        stretch lengths; ``None`` uses the route space's default placement.
    :returns: The mutant, or ``None`` if the route has fewer than two lanes,
        no alternative satisfies the lane-change count and detour bounds,
        or the selected replacement cannot be spliced into the route.
    :rtype: LaneRoute | None
    """
    lanes: Tuple[str, ...] = route.lanes
    if len(lanes) < 2:
        return None
    i: int = rng.randrange(len(lanes) - 1)
    j: int = rng.randrange(i + 1, len(lanes))
    current: Tuple[str, ...] = lanes[i:j + 1]
    # One extra candidate, since the current stretch may be among them.
    paths: List[Path] = [path for path in _paths(space, config, lanes[i], lanes[j], _entries(space, route, lane_change_s)[i],
                                     lanes[:i], lanes[j + 1:], k=config.bridge_candidates + 1)
             if path[0] != current]
    if not paths:
        return None
    # Keep the stretch's number of lane changes: lane changes are
    # mut_lane_change's gene, and alternatives between two lanes of one road
    # usually zigzag (out and back), which piled up lane changes over generations.
    current_changes: int = lane_changes(route.transitions[i:j])
    paths = [path for path in paths if lane_changes(path[1]) == current_changes]
    if not paths:
        return None
    if isfinite(config.max_detour):
        legs: List[RouteLeg] = space.legs(route, lane_change_s=lane_change_s)
        stretch: float = (sum(leg.end_s - leg.start_s for leg in legs[i:j + 1])
                   + current_changes * space.rules.min_lane_change_distance)
        base: float = space.length(route, lane_change_s=lane_change_s) - stretch
        lengths: List[float] = []
        for path in paths:
            child: LaneRoute | None = _splice(route, i, path, route, j)
            try:
                lengths.append(space.length(child) - base if child is not None else inf)
            except InvalidRoute:
                lengths.append(inf)
        shortest: float = min([stretch, *lengths])
        paths = [path for path, length in zip(paths, lengths) if length <= (1 + config.max_detour) * shortest + 1e-6]
        if not paths:
            return None
    return _splice(route, i, rng.choice(paths), route, j)


def mut_extend_or_truncate(route: LaneRoute, space: RouteSpace, rng: Random,
                           config: OperatorConfig = OperatorConfig(), *,
                           direction: str | None = None) -> LaneRoute | None:
    """
    Extend or truncate the route at its start or end.

    Draw one to three lanes at the end or one to two at the start for either
    extension or truncation. Endpoint restrictions may require an extension of
    up to ``_EXTRA_LANES`` more lanes or a truncation to a farther eligible lane.
    Sample a new position on the changed endpoint's lane.

    Draw ``direction`` with equal probability once per mutation event and reuse
    it across retries. Redrawing it on retries previously favoured truncation
    because extensions failed more often (san_mateo ego: 30% versus 73% accepted;
    route lengths fell from 243 to 158 m in 10 neutral generations).

    :param LaneRoute route: Route whose start or end is extended or truncated.
    :param RouteSpace space: Route space supplying legal graph steps and endpoint ranges.
    :param Random rng: Random number generator used for direction, endpoint, lane-count, and lane selection.
    :param OperatorConfig config: Shared operator configuration, retained for interface consistency;
        this function does not use its fields or enforce its length bounds.
    :param str | None direction: ``"extend"`` or ``"truncate"``, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if no action applies, no endpoint is available,
        or the proposed route repeats a lane.
    :rtype: LaneRoute | None
    :raises ValueError: If ``direction`` is unsupported or route construction rejects the proposal.
    """
    lanes: List[str]
    transitions: List[Transition]
    lanes, transitions = list(route.lanes), list(route.transitions)
    graph: DiGraph = space.graph
    actions: List[str] = []
    if len(lanes) > 1:
        actions += ["truncate_end", "truncate_start"]
    if any(lane not in lanes for lane in graph.successors(lanes[-1])):
        actions.append("extend_end")
    if any(lane not in lanes for lane in graph.predecessors(lanes[0])):
        actions.append("extend_start")
    if direction is None:
        direction: str = rng.choice(("extend", "truncate"))
    elif direction not in ("extend", "truncate"):
        raise ValueError(f"direction must be 'extend' or 'truncate', not {direction!r}")
    actions = [action for action in actions if action.startswith(direction)]
    if not actions:
        return None

    start_s: float | None
    end_s: float | None
    start_s, end_s = route.start_s, route.end_s
    action: str = rng.choice(actions)
    # Truncations drop as many lanes as extensions add (1-3 at the end, 1-2 at
    # the start). Keeping a uniformly random prefix dropped half the route on
    # average: most such children fell below min_length and were rejected,
    # while extensions were accepted, so routes grew over generations.
    if action == "truncate_end":
        k: int | None = _truncation_index(space, rng, lanes[:-1], from_end=True, most=3)
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
        count: int = rng.randint(1, 3)
        added: int = 0
        # With endpoint restrictions, a few more lanes may be added to reach one an end may lie on.
        while added < count or (added < count + _EXTRA_LANES and not space.endpoint_ranges(lanes[-1])):
            options: List[str] = sorted(lane for lane in graph.successors(lanes[-1]) if lane not in lanes)
            if not options:
                break
            lane: str = rng.choice(options)
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
                    config: LaneRouteConfig = LaneRouteConfig(), *,
                    lane_change_s: Tuple[float, ...] | None = None,
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

    :param LaneRoute route: Route whose lane-change count is mutated.
    :param RouteSpace space: Route space supplying neighbouring lanes, paths, and endpoint constraints.
    :param Random rng: Random number generator used for action, lane, path, and continuation selection.
    :param LaneRouteConfig config: Configuration defining connecting-path limits and lane-change acceptance odds.
    :param Tuple[float, ...] | None lane_change_s: Existing lane-change start positions
        on source lanes, in metres, or ``None`` for default placement.
    :param float | None acceptance: Uniform draw in [0, 1) reused across mutation
        retries, or ``None`` to draw one here.
    :returns: The mutant, or ``None`` if no action applies, route construction
        fails, no continuation endpoint is available, or added lane changes are rejected.
    :rtype: LaneRoute | None
    """
    lanes: Tuple[str, ...] = route.lanes
    inserts: List[Tuple[int, str]] = [(i, neighbor) for i, lane in enumerate(lanes)
               for neighbor in sorted(space.graph.successors(lane))
               if neighbor not in lanes and space.transition(lane, neighbor) is not Transition.FORWARD]
    removals: Tuple[int, ...] = route.lane_change_indices
    actions: List[str] = (["insert"] if inserts else []) + (["remove"] if removals else [])
    if not actions:
        return None
    if acceptance is None:
        acceptance: float = rng.random()
    if rng.choice(actions) == "insert":
        child: LaneRoute | None = _insert_lane_change(route, space, rng, config, *rng.choice(inserts),
                                    lane_change_s=lane_change_s)
    else:
        child = _remove_lane_change(route, space, rng, config, rng.choice(removals),
                                    lane_change_s=lane_change_s)
    if child is not None:
        added: int = child.lane_changes - route.lane_changes
        if added > 0 and acceptance >= config.lane_change_odds ** added:
            return None
    return child


def _insert_lane_change(route: LaneRoute, space: RouteSpace, rng: Random, config: LaneRouteConfig,
                        i: int, neighbor: str, *,
                        lane_change_s: Tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Insert a change to a neighbouring lane, then rejoin the route or continue forward.

    Try a connecting path to a randomly selected later route lane. If none fits,
    continue forward from the neighbour for approximately the old remaining distance.

    :param LaneRoute route: Route receiving the lane change.
    :param RouteSpace space: Route space supplying transitions, paths, and endpoint constraints.
    :param Random rng: Random number generator used for rejoin and continuation choices.
    :param LaneRouteConfig config: Configuration defining connecting-path search limits and candidate count.
    :param int i: Index of the source lane in ``route``.
    :param str neighbor: Neighbouring lane reached by the inserted lane change.
    :param Tuple[float, ...] | None lane_change_s: Existing lane-change start positions
        in metres, or ``None`` for default placement.
    :returns: The proposed route, or ``None`` if the source has insufficient lane-change
        distance, no continuation endpoint is available, or construction repeats a lane.
    :rtype: LaneRoute | None
    """
    lanes: Tuple[str, ...]
    transitions: Tuple[Transition, ...]
    lanes, transitions = route.lanes, route.transitions
    step: Transition = space.transition(lanes[i], neighbor)
    entry: float = _entries(space, route, lane_change_s)[i]
    if entry + space.rules.min_lane_change_distance > space.lane_length(lanes[i]):
        return None
    if i < len(lanes) - 1:
        j: int = rng.randrange(i + 1, len(lanes))
        paths: List[Path] = _paths(space, config, neighbor, lanes[j], space.entry_after_change(lanes[i], entry, neighbor),
                       lanes[:i + 1], lanes[j + 1:])
        if paths:
            path: Tuple[str, ...]
            steps: Tuple[Transition, ...]
            path, steps = rng.choice(paths)
            return _route(lanes[:i + 1] + path + lanes[j + 1:],
                          transitions[:i] + (step,) + steps + transitions[j:], route.start_s, route.end_s)
    # No way back: stay on the neighbouring side and drive on forward about as
    # far as the route did (ending right on the neighbour truncated it).
    remaining: float = _cumulative(space, route, lane_change_s)[-1] - _cumulative(space, route, lane_change_s)[i]
    tail: Tuple[str, ...]
    end_s: float
    tail, steps, end_s = _drive_on(space, rng, lanes[:i + 1] + (neighbor,), remaining)
    return _ended(space, _route(tail, transitions[:i] + (step,) + steps, route.start_s, end_s), end_s)


def _remove_lane_change(route: LaneRoute, space: RouteSpace, rng: Random, config: LaneRouteConfig,
                        i: int, *, lane_change_s: Tuple[float, ...] | None = None) -> LaneRoute | None:
    """
    Replace a stretch with fewer lane changes, or continue forward before the change.

    Try paths to a randomly selected later lane. If none reduces the lane-change
    count, continue forward for approximately the old remaining distance.

    :param LaneRoute route: Route whose lane-change stretch is replaced.
    :param RouteSpace space: Route space supplying connecting paths and forward continuation.
    :param Random rng: Random number generator used for rejoin, path, and continuation choices.
    :param LaneRouteConfig config: Configuration defining connecting-path search limits and candidate count.
    :param int i: Source-lane index of the lane change to remove.
    :param Tuple[float, ...] | None lane_change_s: Existing lane-change start positions
        in metres, or ``None`` for default placement.
    :returns: The proposed route, or ``None`` if no continuation endpoint is available
        or the selected splice repeats a lane.
    :rtype: LaneRoute | None
    """
    lanes: Tuple[str, ...]
    transitions: Tuple[Transition, ...]
    lanes, transitions = route.lanes, route.transitions
    j: int = rng.randrange(i + 1, len(lanes))
    current: int = lane_changes(transitions[i:j])
    paths: List[Path] = [path for path in _paths(space, config, lanes[i], lanes[j], _entries(space, route, lane_change_s)[i],
                                     lanes[:i], lanes[j + 1:])
             if lane_changes(path[1]) < current]
    if paths:
        return _splice(route, i, rng.choice(paths), route, j)
    # No way around it: drive on forward from before the lane change about as
    # far as the route did (ending before it truncated the route).
    remaining: float = _cumulative(space, route, lane_change_s)[-1] - _cumulative(space, route, lane_change_s)[i]
    tail: Tuple[str, ...]
    steps: Tuple[Transition, ...]
    end_s: float
    tail, steps, end_s = _drive_on(space, rng, lanes[:i + 1], remaining)
    return _ended(space, _route(tail, transitions[:i] + steps, route.start_s, end_s), end_s)


def _drive_on(space: RouteSpace, rng: Random, lanes: Sequence[str], remaining: float,
              ) -> Tuple[Tuple[str, ...], Tuple[Transition, ...], float]:
    """
    Continue a lane sequence through forward successors for approximately the requested distance.

    Measure ``remaining`` from the start of the input sequence's last lane.
    Stop at that distance or when no unvisited forward successor is available.
    The returned position is the raw one: callers repair it on the whole route (``_ended``).

    :param RouteSpace space: Route space supplying lane lengths, forward edges, and endpoint ranges.
    :param Random rng: Random number generator used to select forward successors.
    :param Sequence[str] lanes: Nonempty lane sequence to continue without repeating lanes.
    :param float remaining: Desired distance from the start of the last input lane, in metres.
    :returns: The complete lane Tuple, transitions added after the input sequence,
        and the final lane position in metres, clamped to the lane.
    :rtype: Tuple[Tuple[str, ...], Tuple[Transition, ...], float]
    """
    steps: List[Transition]
    lanes, steps = list(lanes), []
    while remaining > space.lane_length(lanes[-1]):
        options: List[str] = sorted(lane for lane in space.graph.successors(lanes[-1])
                         if lane not in lanes and space.transition(lanes[-1], lane) is Transition.FORWARD)
        if not options:
            break
        remaining -= space.lane_length(lanes[-1])
        lanes.append(rng.choice(options))
        steps.append(Transition.FORWARD)
    return tuple(lanes), tuple(steps), min(max(remaining, 0.0), space.lane_length(lanes[-1]))


def _ended(space: RouteSpace, route: LaneRoute | None, s: float) -> LaneRoute | None:
    """
    ``route`` with its end at the allowed position nearest to ``s`` (``RouteSpace.nearest_end``).

    :param RouteSpace space: Route space supplying the allowed ends.
    :param LaneRoute | None route: Route to end, or ``None``.
    :param float s: Desired end position on its last lane, in metres.
    :returns: The ended route, or ``None`` if ``route`` is ``None`` or its last lane allows no end.
    :rtype: LaneRoute | None
    """
    end_s: float | None = None if route is None else space.nearest_end(route, s)
    return None if end_s is None else replace(route, end_s=end_s)


def mut_s(route: LaneRoute, space: RouteSpace, rng: Random,
          config: OperatorConfig = OperatorConfig()) -> LaneRoute | None:
    """
    Move the start s, the end s, or both by a normal step, staying on their lanes.

    Where the space keeps endpoints away from junctions, a step that lands too
    close to one moves to the nearest allowed position on the lane. The end then
    moves, as little as it must, to stay an allowed end of the mutated route
    (``RouteSpace.nearest_end``): beyond where it enters its last lane, which a
    later start can delay.

    :param LaneRoute route: Route whose endpoint positions are mutated.
    :param RouteSpace space: Route space supplying lane lengths and allowed endpoint positions.
    :param Random rng: Random number generator used to select endpoints and sample normal steps.
    :param OperatorConfig config: Configuration defining the position step standard deviation ``sigma_s``.
    :returns: The mutant, or ``None`` if a moved endpoint's lane has no allowed position.
    :rtype: LaneRoute | None
    """
    start_s: float | None
    end_s: float | None
    start_s, end_s = route.start_s, route.end_s
    which: str = rng.choice(("start", "end", "both"))
    if which in ("start", "both"):
        start_s = gauss_step(rng, start_s, config.sigma_s, space.lane_length(route.lanes[0]))
        if space.restricts_endpoints:
            start_s = space.nearest_endpoint(route.lanes[0], start_s)
    if start_s is None:
        return None
    if which in ("end", "both"):
        end_s = gauss_step(rng, end_s, config.sigma_s, space.lane_length(route.lanes[-1]))
    # A later start delays the earliest lane change, which can move the last lane's entry past the end.
    end_s = space.nearest_end(replace(route, start_s=start_s), end_s)
    if end_s is None:
        return None
    return replace(route, start_s=start_s, end_s=end_s)


def _truncation_index(space: RouteSpace, rng: Random, lanes: Sequence[str], *, from_end: bool, most: int) -> int | None:
    """
    Select an allowed endpoint lane near the truncated end of a lane sequence.

    Prefer lanes within ``most`` positions of the cut end. If endpoint restrictions
    leave none in that range, choose the closest eligible lane farther away.

    :param RouteSpace space: Route space defining allowed endpoint ranges.
    :param Random rng: Random number generator used to choose among nearby eligible lanes.
    :param Sequence[str] lanes: Candidate lanes, excluding the route's original last
        lane for end truncation or original first lane for start truncation.
    :param bool from_end: Whether to select near the end rather than the start of ``lanes``.
    :param int most: Number of lane positions near the cut end considered first.
    :returns: An index into ``lanes``, or ``None`` if no lane permits an endpoint.
    :rtype: int | None
    """
    allowed: List[int] = [index for index, lane in enumerate(lanes)
               if not space.restricts_endpoints or space.endpoint_ranges(lane)]
    if not allowed:
        return None
    reach: range = range(max(len(lanes) - most, 0), len(lanes)) if from_end else range(min(most, len(lanes)))
    near: List[int] = [index for index in allowed if index in reach]
    if near:
        return rng.choice(near)
    return max(allowed) if from_end else min(allowed)


def _disjoint(lanes: Iterable[str], *others: Iterable[str]) -> bool:
    """
    Check whether a lane collection shares no lane with any other collection.

    :param Iterable[str] lanes: Lane IDs checked against each other collection.
    :param Iterable[str] others: Additional lane-ID collections; overlaps among these are allowed.
    :returns: ``True`` if ``lanes`` is disjoint from every collection in ``others``.
    :rtype: bool
    """
    lanes: Set[str] = set(lanes)
    return all(lanes.isdisjoint(other) for other in others)


def _route(lanes: Iterable[str], transitions: Iterable[Transition],
           start_s: float, end_s: float) -> LaneRoute | None:
    """
    Construct a lane route after rejecting repeated lane IDs.

    :param Iterable[str] lanes: Lane IDs in driving order, converted to a tuple.
    :param Iterable[Transition] transitions: Transitions between consecutive lanes, converted to a tuple.
    :param float start_s: Position along the first lane, in metres.
    :param float end_s: Position along the last lane, in metres.
    :returns: The constructed route, or ``None`` if any lane ID is repeated.
    :rtype: LaneRoute | None
    :raises ValueError: If the route is empty, transition count is inconsistent,
        or either endpoint position is nonfinite.
    """
    lanes, transitions = tuple(lanes), tuple(transitions)
    if len(set(lanes)) != len(lanes):
        return None
    return LaneRoute(lanes, transitions, start_s, end_s)


def _entries(space: RouteSpace, route: LaneRoute, lane_change_s: Tuple[float, ...] | None) -> List[float]:
    """
    Return the position at which the route enters each lane.

    :param RouteSpace space: Route space used to compute route legs.
    :param LaneRoute route: Route whose lane entries are requested.
    :param Tuple[float, ...] | None lane_change_s: Lane-change start positions in metres,
        or ``None`` for default placement.
    :returns: Entry positions in metres along each lane, in route order.
    :rtype: List[float]
    :raises InvalidRoute: If route legs cannot be placed with the supplied lane-change positions.
    """
    return [leg.start_s for leg in space.legs(route, lane_change_s=lane_change_s)]


def _paths(space: RouteSpace, config: LaneRouteConfig, source: str, target: str, entry_s: float,
           *avoid: Iterable[str], k: int | None = None) -> List[Path]:
    """
    Find cheap connecting paths that avoid all supplied lane collections.

    Request up to ``k`` cheapest paths within ``config.max_length``, then discard
    paths intersecting ``avoid``. Discarded paths are not replaced by further searches.

    :param RouteSpace space: Route space used to search for lane paths.
    :param LaneRouteConfig config: Configuration defining default candidate count and maximum path cost.
    :param str source: Starting lane ID.
    :param str target: Destination lane ID.
    :param float entry_s: Entry position on the source lane, in metres.
    :param Iterable[str] avoid: Lane-ID collections that a returned path must not intersect.
    :param int | None k: Candidate count requested before filtering, or ``None`` to use
        ``config.bridge_candidates``.
    :returns: Retained paths, each containing lane IDs and their connecting transitions.
    :rtype: List[Path]
    """
    candidates: Iterator[Path] = space.lane_paths(source, target, config.bridge_candidates if k is None else k,
                                  max_cost=config.max_length, entry_s=entry_s)
    return [(lanes, steps) for lanes, steps in candidates if _disjoint(lanes, *avoid)]


def _cumulative(space: RouteSpace, route: LaneRoute, lane_change_s: Tuple[float, ...] | None) -> List[float]:
    """
    Estimate driven distance to each lane entry, followed by the total route length.

    Use placed route legs when possible. If leg placement raises ``InvalidRoute``,
    use full lane lengths adjusted by route endpoints. In both cases, add
    ``space.rules.min_lane_change_distance`` for each non-forward transition.

    :param RouteSpace space: Route space supplying route legs, lane lengths, and lane-change distance.
    :param LaneRoute route: Route whose cumulative distances are computed.
    :param Tuple[float, ...] | None lane_change_s: Lane-change start positions in metres,
        or ``None`` for default placement.
    :returns: Lane-entry distances followed by the total estimate, in metres;
        the list contains one more item than the route has lanes.
    :rtype: List[float]
    """
    try:
        stretches: List[float] = [leg.end_s - leg.start_s for leg in space.legs(route, lane_change_s=lane_change_s)]
    except InvalidRoute:
        stretches = [space.service.get_lane_by_id(lane).length for lane in route.lanes]
        stretches[-1] = route.end_s
        stretches[0] -= route.start_s
    marks: List[float]
    distance: float
    marks, distance = [], 0.0
    for index, stretch in enumerate(stretches):
        marks.append(distance)
        distance += stretch
        if index < len(route.transitions) and route.transitions[index] is not Transition.FORWARD:
            distance += space.rules.min_lane_change_distance
    return marks + [distance]


def _fewest_extra_changes(paths: List[Path], changes: int) -> List[Path]:
    """
    Keep paths whose lane-change count is closest to the requested count.

    :param List[Path] paths: Candidate paths with lane IDs and transitions.
    :param int changes: Target lane-change count.
    :returns: All paths with the smallest absolute count difference, in input order;
        an empty input yields an empty list.
    :rtype: List[Path]
    """
    gaps: List[int] = [abs(lane_changes(steps) - changes) for _, steps in paths]
    return [path for path, gap in zip(paths, gaps) if gap == min(gaps)]


def _splice(head: LaneRoute, i: int, path: Path, tail: LaneRoute, j: int) -> LaneRoute | None:
    """
    Join a route prefix, a connecting path, and a route suffix.

    The path includes ``head``'s lane at ``i`` and ``tail``'s lane at ``j``.
    Retain the head's start position and the tail's end position.

    :param LaneRoute head: Route providing lanes before index ``i`` and the start position.
    :param int i: Source-lane index in ``head`` included in the connecting path.
    :param Path path: Connecting lane IDs and transitions, including both join lanes.
    :param LaneRoute tail: Route providing lanes after index ``j`` and the end position.
    :param int j: Destination-lane index in ``tail`` included in the connecting path.
    :returns: The joined route, or ``None`` if the combined lanes contain repeats.
    :rtype: LaneRoute | None
    :raises ValueError: If the combined route fails structural or endpoint-finiteness validation.
    """
    lanes: Tuple[str, ...]
    steps: Tuple[Transition, ...]
    lanes, steps = path
    return _route(head.lanes[:i] + lanes + tail.lanes[j + 1:], head.transitions[:i] + steps + tail.transitions[j:],
                  head.start_s, tail.end_s)
