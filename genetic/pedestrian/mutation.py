"""
Mutation of PedestrianAgent chromosomes.

Each operator returns a mutant that changes one gene, or None when it does
not apply. ``mut_pedestrian`` mutates each gene with its rate. Operators that
draw (a kind, an add-or-remove action, a walker type) draw once per mutation
event and keep the draw over retries, so whatever succeeds more often is not
favoured and the generation mixes are kept.
"""

from dataclasses import replace
from random import Random

from shapely.geometry import LineString

from ...scenario.pedestrian import (
    CURB_STAND_BACK, CURB_WAIT_PROBABILITY, GUTTER_RANGE, GUTTER_SHARE, HESITATION, HESITATION_PROBABILITY,
    ISLAND_WAIT, ISLAND_WAIT_PROBABILITY, MAX_CURB_WAIT, RESUME_GAP, STOP_SPACING, PedestrianAgent, WalkerType,
)
from ...scenario.realism import draw_type, style_step
from ...scenario.replay import MAX_SPEED_POINTS, SpeedPoint, ordered_points
from ...scenario.walkways import (
    MAX_EXCURSIONS, MIN_TURN_BACK, MIN_WALK, TURN_BACK_SHARE, Excursion, ExcursionKind, WalkNetwork,
)
from ..common import gauss_step, vary
from ..profile_operators import SPEED_POINT_ACTIONS, mut_speed, mut_speed_point, mut_start_t, with_profile
from .common import PedestrianOperatorConfig, acceptable, on_route, rehomed, retyped

COUNT_ACTIONS = ("add", "remove")
"""``mut_count``'s actions, drawn with equal chance once per mutation event."""


def _with_excursion(agent: PedestrianAgent, index: int, excursion: Excursion | None, network: WalkNetwork,
                    rng: Random, config: PedestrianOperatorConfig) -> PedestrianAgent | None:
    """``agent`` with excursion ``index`` replaced (or removed if None), its end rehomed."""
    excursions = list(agent.route.excursions)
    if excursion is None:
        del excursions[index]
    else:
        excursions[index] = excursion
    route = rehomed(replace(agent.route, excursions=tuple(excursions)), network, rng, config, start=True)
    return on_route(agent, route, network)


def mut_shift(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
              config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Move one point of the walk along its sidewalk by a normal step: the start
    or end (``config.sigma_s``), or an excursion's entry or exit
    (``config.sigma_curb``), which must keep its excursion's kind.

    A start or end stays within ``config.max_approach`` of the entry or arrival
    it is drawn around at generation (a walk without excursions: its end
    ``MIN_WALK`` to twice that from its start); a step of any point that takes
    it beyond is rejected. The
    walk lengths then keep their generation distribution instead of diffusing
    outward (san_mateo, 20 neutral generations: median 52 -> 78 m).

    :returns: The mutant, or None if the moved walk is invalid or out of range.
    """
    route = agent.route
    choices = ["start", "end"] + [(index, side) for index in range(len(route.excursions)) for side in ("entry", "exit")]
    choice = rng.choice(choices)
    if choice in ("start", "end"):
        point = getattr(route, choice)
        moved = network.wrapped(point.sidewalk, point.s + rng.gauss(0, config.sigma_s))
        candidate = replace(route, **{choice: moved})
        if not _within_approach(candidate, network, config):
            return None
        return on_route(agent, candidate, network)
    index, side = choice
    excursion = route.excursions[index]
    point = getattr(excursion, side)
    moved = network.wrapped(point.sidewalk, point.s + rng.gauss(0, config.sigma_curb))
    excursions = route.excursions[:index] + (replace(excursion, **{side: moved}),) + route.excursions[index + 1:]
    candidate = replace(route, excursions=excursions)
    if not _within_approach(candidate, network, config):
        return None
    return on_route(agent, candidate, network)


def _within_approach(route, network: WalkNetwork, config: PedestrianOperatorConfig) -> bool:
    """Whether the walk's start and end lie where generation draws them."""
    if not route.excursions:
        return MIN_WALK <= network.gap(route.start, route.end) <= max(MIN_WALK, 2 * config.max_approach)
    return (network.gap(route.start, route.excursions[0].entry) <= config.max_approach
            and network.gap(route.end, route.excursions[-1].arrival) <= config.max_approach)


def mut_redraw(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
               config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Redraw one excursion of the same kind, entering within ``config.redraw_reach``
    of its old entry: another crosswalk at the corner, another spot mid-block.

    :returns: The mutant, or None if the walk has no excursion or nothing fits.
    """
    if not agent.route.excursions:
        return None
    index = rng.randrange(len(agent.route.excursions))
    excursion = agent.route.excursions[index]
    pair = network.propose(excursion.kind, rng, excursion.entry, config.redraw_reach)
    if pair is None:
        return None
    return _with_excursion(agent, index, replace(excursion, entry=pair[0], exit=pair[1]), network, rng, config)


def mut_kind(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
             config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
             drawn: ExcursionKind | None = None) -> PedestrianAgent | None:
    """
    Redraw one excursion's kind from ``config.kind_weights`` and draw an
    excursion of that kind entering within ``config.redraw_reach`` of the old
    entry. A mutation event draws the kind once (``drawn``); a draw of the
    current kind is a no-op.

    :returns: The mutant, or None if the walk has no excursion, the draw is the
        current kind, or nothing of the kind fits.
    """
    if not agent.route.excursions:
        return None
    if drawn is None:
        drawn = _draw_kind(network, rng, config)
        if drawn is None:
            return None
    index = rng.randrange(len(agent.route.excursions))
    excursion = agent.route.excursions[index]
    if drawn is excursion.kind:
        return None
    pair = network.propose(drawn, rng, excursion.entry, config.redraw_reach)
    if pair is None:
        return None
    return _with_excursion(agent, index, replace(excursion, kind=drawn, entry=pair[0], exit=pair[1]), network, rng,
                           config)


def mut_count(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
              config: PedestrianOperatorConfig = PedestrianOperatorConfig(), action: str | None = None,
              drawn: ExcursionKind | None = None, accept: float | None = None) -> PedestrianAgent | None:
    """
    Add an excursion after the last, entering within ``config.max_link`` of the
    walk's end (anywhere, for a walk without excursions), or remove the last
    one, the walk then ending near its entry.
    ``action`` (one of ``COUNT_ACTIONS``) and the added kind ``drawn`` are drawn
    once per mutation event.

    The change is a Metropolis step under ``config.count_weights``: going from
    ``n`` to ``m`` excursions is accepted with probability
    ``min(1, weight(m) / weight(n))``. Accepting every add and remove drove the
    counts towards a uniform mix (san_mateo, 15 generations of mutation:
    one-excursion walks 67% -> 47%). The Metropolis uniform ``accept`` is drawn
    once per mutation event too: drawn per retry, a rejected step was retried
    until accepted, which made leaving the common one-excursion walks about
    twice as likely as returning (shalun: 66% -> 56% in 17 generations).

    :returns: The mutant, or None if the action does not apply, is rejected, or nothing fits.
    """
    action = rng.choice(COUNT_ACTIONS) if action is None else action
    if action not in COUNT_ACTIONS:
        raise ValueError(f"action must be one of {COUNT_ACTIONS}, got {action!r}")
    route = agent.route
    count = len(route.excursions)
    target = count + (1 if action == "add" else -1)
    weight = {int(key): value for key, value in config.count_weights.items()}
    if not 0 <= target <= MAX_EXCURSIONS or weight.get(target, 0.0) <= 0:
        return None
    if weight.get(count, 0.0) > 0 and (rng.random() if accept is None else accept) >= weight[target] / weight[count]:
        return None
    if action == "remove":
        if not route.excursions:
            return None
        last = route.excursions[-1]
        end = network.wrapped(last.entry.sidewalk, last.entry.s + rng.uniform(-config.max_approach, config.max_approach))
        return on_route(agent, replace(route, excursions=route.excursions[:-1], end=end), network)
    if len(route.excursions) >= MAX_EXCURSIONS or (route.excursions and route.excursions[-1].turn_back is not None):
        return None
    drawn = _draw_kind(network, rng, config) if drawn is None else drawn
    if drawn is None:
        return None
    if not route.excursions:
        # A walk on the sidewalk only was drawn anywhere; so is its first excursion,
        # the walk then starting near the entry as at generation. Anchored to the
        # walk's end, adds failed where no crossing was near, and such walks piled up
        # (san_mateo, 15 generations of mutation: 14% -> 23%).
        pair = network.propose(drawn, rng)
        if pair is None:
            return None
        start = network.wrapped(pair[0].sidewalk, pair[0].s + rng.uniform(-config.max_approach, config.max_approach))
    else:
        pair = network.propose(drawn, rng, route.end, config.max_link)
        if pair is None or pair[0].sidewalk != route.end.sidewalk:
            return None
        start = route.start
    end = network.wrapped(pair[1].sidewalk, pair[1].s + rng.uniform(-config.max_approach, config.max_approach))
    added = replace(route, start=start, excursions=route.excursions + (Excursion(drawn, *pair),), end=end)
    return on_route(agent, added, network)


STOP_ACTIONS = ("toggle", "dwell")
"""``mut_walk_stop``'s actions, drawn with equal chance once per mutation event."""

_WAITS = ("curb", "island", "hesitation")


def _wait_chance(kind: str) -> float:
    return {"curb": CURB_WAIT_PROBABILITY, "island": ISLAND_WAIT_PROBABILITY,
            "hesitation": HESITATION_PROBABILITY}[kind]


def mut_walk_stop(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  action: str | None = None, choice: tuple[float, str, float] | None = None) -> PedestrianAgent | None:
    """
    Add or remove one wait of the kinds generation draws, or step a wait's dwell.

    ``"toggle"``: pick an excursion and a wait kind (at the curb, on a median
    or island it crosses, a hesitation mid-road). If the excursion has such a
    wait (a stop and the resume point ``RESUME_GAP`` after it), remove both;
    else add one where generation puts it. Adding is accepted with
    probability ``min(1, q / (1 - q))`` and removing with ``min(1, (1 - q) / q)``,
    ``q`` the kind's chance at generation, so each excursion keeps waits at
    the generation rates. ``"dwell"``: step one stop's dwell (``config.sigma_t``).
    ``choice`` (a uniform picking the excursion, the wait kind, a uniform for
    the acceptance) is drawn once per mutation event with the action: drawn per
    retry, whichever of adding or removing succeeds more often would win.

    The shared ``profile_operators.mut_stop`` chose among the actions that
    apply; walkers' profiles (start, curb wait, resume) seldom have an inner
    moving point to stop at, so it mostly removed waits (borregas: 0.62 ->
    0.30 waits per walker in 30 generations), and resumed at up to the type's
    top speed (3.5 m/s for adults).

    :returns: The mutant, or None if nothing applies or the step is rejected.
    """
    action = rng.choice(STOP_ACTIONS) if action is None else action
    if action not in STOP_ACTIONS:
        raise ValueError(f"action must be one of {STOP_ACTIONS}, got {action!r}")
    profile = agent.speed_profile
    if action == "dwell":
        stops = [index for index, point in enumerate(profile) if point.speed == 0]
        if not stops:
            return None
        index = rng.choice(stops)
        point = replace(profile[index], dwell=gauss_step(rng, profile[index].dwell, config.sigma_t, config.max_dwell))
        return with_profile(agent, profile[:index] + (point,) + profile[index + 1:])
    route = agent.route
    if not route.excursions:
        return None
    if choice is None:
        choice = (rng.random(), rng.choice(_WAITS), rng.random())
    pick, kind, accept = choice
    index = min(int(pick * len(route.excursions)), len(route.excursions) - 1)
    name = f"x{index}"
    frame = network.frame(route)
    element = frame.elements[frame.index[name]]
    line = LineString(element.coords)
    curb = min(network.curb_distance(line), element.length)
    room = element.length - RESUME_GAP - 0.1
    islands = network.island_spans(line)

    def in_kind(s: float) -> bool:
        if kind == "curb":
            return s <= curb + GUTTER_RANGE[1] + 1e-6
        if kind == "island":
            return any(low - 1e-6 <= s <= high + 1e-6 for low, high in islands)
        return curb + GUTTER_RANGE[1] < s and not any(low - 1e-6 <= s <= high + 1e-6 for low, high in islands)

    present = [i for i, point in enumerate(profile) if point.speed == 0 and point.lane_id == name and in_kind(point.s)]
    q = _wait_chance(kind)
    if present:
        if accept >= min(1.0, (1 - q) / q):
            return None
        i = present[0]
        drop = {i} | ({i + 1} if i + 1 < len(profile) and profile[i + 1].lane_id == name
                      and profile[i + 1].speed > 0 and abs(profile[i + 1].s - profile[i].s - RESUME_GAP) < 1e-6 else set())
        kept = tuple(point for j, point in enumerate(profile) if j not in drop)
        if not kept or kept[-1].speed == 0:
            return None
        return with_profile(agent, kept)
    if accept >= min(1.0, q / (1 - q)) or len(profile) + 2 > MAX_SPEED_POINTS:
        return None
    if kind == "curb":
        at = curb + rng.uniform(*GUTTER_RANGE) if rng.random() < GUTTER_SHARE else max(curb - CURB_STAND_BACK, 0.0)
        dwell = rng.uniform(0, MAX_CURB_WAIT)
    elif kind == "island":
        if not islands:
            return None
        low, high = rng.choice(islands)
        at, dwell = (low + high) / 2, rng.uniform(*ISLAND_WAIT)
    else:
        if room - 1 <= curb + 1:
            return None
        at, dwell = rng.uniform(curb + 1, room - 1), rng.uniform(*HESITATION)
    if not 0 < at <= room:
        return None
    position = frame.distance(name, at)
    if any(abs(frame.distance(point.lane_id, point.s) - position) < STOP_SPACING + RESUME_GAP for point in profile):
        return None
    moving = [point.speed for point in profile if point.speed > 0]
    after = [point for point in profile if frame.distance(point.lane_id, point.s) > position and point.speed > 0]
    resume = after[0].speed if after else moving[-1]
    added = [SpeedPoint(name, at, 0.0, dwell), SpeedPoint(name, at + RESUME_GAP, resume)]
    placed = [(frame.distance(point.lane_id, point.s), rank, point) for rank, point in enumerate(list(profile) + added)]
    result = ordered_points(placed, profile[0])
    if len(result) != len(profile) + 2:
        return None
    return with_profile(agent, result)


TURN_BACK_ACTIONS = ("step", "redraw")
"""``mut_turn_back``'s actions, drawn with equal chance once per mutation event."""


def mut_turn_back(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  action: str | None = None, draw: float | None = None) -> PedestrianAgent | None:
    """
    Step the last excursion's turning point by ``config.sigma_turn_back``
    (``"step"``; a no-op for a walker that does not turn back), or redraw
    whether the walker turns back (``"redraw"``): with
    ``config.turn_back_probability`` at a point drawn as at generation, else
    not at all. The action is drawn once per mutation event with equal chance,
    so walkers start and stop turning back at rates that keep that probability:
    stepping instead of redrawing whenever possible kept twice as many.

    ``draw`` (uniform in [0, 1)) decides a redraw and is drawn once per mutation
    event: drawn per retry, a redraw that left the walk as it was (most of
    them) was retried until a turning point came up, which doubled the share
    of walkers who turn back in 30 generations (borregas 4.8% -> 8.5%).

    :returns: The mutant, or None if the walk has no excursion or nothing changes.
    """
    if not agent.route.excursions:
        return None
    action = rng.choice(TURN_BACK_ACTIONS) if action is None else action
    if action not in TURN_BACK_ACTIONS:
        raise ValueError(f"action must be one of {TURN_BACK_ACTIONS}, got {action!r}")
    last = agent.route.excursions[-1]
    length = network.excursion_line(last.kind, last.entry, last.exit).length
    high = length - MIN_TURN_BACK
    if high < MIN_TURN_BACK:
        return None
    if action == "step":
        if last.turn_back is None:
            return None
        turn_back = min(max(last.turn_back + rng.gauss(0, config.sigma_turn_back), MIN_TURN_BACK), high)
    elif (rng.random() if draw is None else draw) < config.turn_back_probability:
        turn_back = rng.uniform(MIN_TURN_BACK, min(max(TURN_BACK_SHARE * length, MIN_TURN_BACK), high))
    else:
        turn_back = None
    if turn_back == last.turn_back:
        # A redraw that keeps the walk as it is is still a redraw: returning None
        # here made the event retry until a turning point was drawn, which doubled
        # the share of walkers who turn back in 30 generations (borregas 4.8% -> 8.5%).
        return None
    return _with_excursion(agent, len(agent.route.excursions) - 1, replace(last, turn_back=turn_back), network, rng,
                           config)


def mut_walker_type(agent: PedestrianAgent, rng: Random,
                    config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                    drawn: WalkerType | None = None) -> PedestrianAgent | None:
    """
    Resample the walker type from ``config.walker_type_weights``, speeds clamped
    to its maximum. A mutation event draws once (``drawn``); a draw of the current
    type is a no-op, since redrawing until it changes would flatten the mix.

    :returns: The mutant, or None if the draw is the current type.
    """
    walker_type = drawn if drawn is not None else WalkerType[draw_type(rng, config.walker_type_weights)]
    return None if walker_type is agent.walker_type else retyped(agent, walker_type)


def mut_walk_style(agent: PedestrianAgent, rng: Random) -> PedestrianAgent:
    """
    Step the walk style by a reflected normal step within [0, 1]
    (``realism.style_step``), which keeps its uniform prior.

    :rtype: PedestrianAgent
    """
    return replace(agent, walk_style=style_step(rng, agent.walk_style))


def _draw_kind(network: WalkNetwork, rng: Random, config: PedestrianOperatorConfig) -> ExcursionKind | None:
    available = {kind.name for kind in network.kinds()}
    weights = {name: weight for name, weight in config.kind_weights.items() if name in available and weight > 0}
    return ExcursionKind[draw_type(rng, weights)] if weights else None


def _same_stops(parent: PedestrianAgent, child: PedestrianAgent | None) -> PedestrianAgent | None:
    """``child`` if it has as many stops as ``parent`` and still starts moving, else None: waits change only
    through ``mut_walk_stop``, which keeps their generation rates (speed-point
    adds, removes and moves lost 0.09 waits per application on borregas)."""
    if child is None:
        return None
    count = lambda agent: sum(point.speed == 0 for point in agent.speed_profile)
    if child.speed_profile[0].speed == 0 and parent.speed_profile[0].speed > 0:
        # Removing the walking speed before a curb wait left profiles that start
        # with the wait (shalun: 19% of adults after 30 generations).
        return None
    return child if count(child) == count(parent) else None


def mut_pedestrian(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                   config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent:
    """
    Return a mutant: each gene mutates only with its own
    ``config.mutation_rates`` chance, through an operator drawn by
    ``config.mutation_weights``.

    Every step keeps the mutant acceptable. When no gene mutated, ``agent``
    is returned unchanged.

    :rtype: PedestrianAgent
    """
    # One draw of each per mutation event, kept over retries.
    kind = _draw_kind(network, rng, config)
    added = _draw_kind(network, rng, config)
    count_action = rng.choice(COUNT_ACTIONS)
    count_accept = rng.random()
    turn_back_action = rng.choice(TURN_BACK_ACTIONS)
    turn_back_draw = rng.random()
    point_action = rng.choice(SPEED_POINT_ACTIONS)
    stop_action = rng.choice(STOP_ACTIONS)
    stop_choice = (rng.random(), rng.choice(_WAITS), rng.random())
    walker_type = WalkerType[draw_type(rng, config.walker_type_weights)]
    operators = {
        "shift": lambda child: mut_shift(child, network, rng, config),
        "redraw": lambda child: mut_redraw(child, network, rng, config),
        "kind": lambda child: None if kind is None else mut_kind(child, network, rng, config, kind),
        "count": lambda child: mut_count(child, network, rng, config, count_action, added, count_accept),
        "turn_back": lambda child: mut_turn_back(child, network, rng, config, turn_back_action, turn_back_draw),
        "walker_type": lambda child: mut_walker_type(child, rng, config, walker_type),
        "speed": lambda child: mut_speed(child, rng, config, network),
        "speed_point": lambda child: _same_stops(child, mut_speed_point(child, rng, config, network, point_action)),
        "stop": lambda child: mut_walk_stop(child, network, rng, config, stop_action, stop_choice),
        "walk_style": lambda child: mut_walk_style(child, rng),
        "start_t": lambda child: mut_start_t(child, rng, config),
    }
    return vary(agent, config.GENES, operators, config.MUTATION_GENES, config.mutation_weights,
                config.mutation_rates, rng, config.attempts, lambda candidate: acceptable(network, candidate, config))
