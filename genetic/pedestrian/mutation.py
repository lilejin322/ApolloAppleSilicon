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
from typing import Callable, Dict, List, Set, Tuple
from shapely.geometry import LineString

from ...scenario.pedestrian import (
    CURB_STAND_BACK, CURB_WAIT_PROBABILITY, GUTTER_RANGE, GUTTER_SHARE, HESITATION, HESITATION_PROBABILITY,
    ISLAND_WAIT, ISLAND_WAIT_PROBABILITY, MAX_CURB_WAIT, RESUME_GAP, STOP_SPACING, PedestrianAgent, WalkerType,
)
from ...scenario.realism import draw_type, style_step
from ...scenario.replay import MAX_SPEED_POINTS, SpeedPoint, ordered_points
from ...scenario.walkways import (
    MAX_EXCURSIONS, MIN_TURN_BACK, MIN_WALK, TURN_BACK_SHARE, Excursion, ExcursionKind, WalkNetwork,
    WalkRoute, CurbPoint, WalkElement, WalkFrame,
)
from ..common import gauss_step, vary
from ..profile_operators import SPEED_POINT_ACTIONS, mut_speed, mut_speed_point, mut_start_t, with_profile
from .common import PedestrianOperatorConfig, acceptable, on_route, rehomed, retyped

COUNT_ACTIONS: Tuple[str, ...] = ("add", "remove")
"""``mut_count``'s actions, drawn with equal chance once per mutation event."""


def _with_excursion(agent: PedestrianAgent, index: int, excursion: Excursion | None, network: WalkNetwork,
                    rng: Random, config: PedestrianOperatorConfig) -> PedestrianAgent | None:
    """
    Replace or remove an excursion, adjusting the walk's endpoints when needed.

    :param PedestrianAgent agent: Agent whose excursion is changed.
    :param int index: Index of the excursion to replace or remove.
    :param Excursion | None excursion: Replacement excursion, or ``None`` to remove it.
    :param WalkNetwork network: Walk network used to adjust endpoints and apply the walk.
    :param Random rng: Random number generator used to redraw endpoints.
    :param PedestrianOperatorConfig config: Configuration defining the maximum approach distance.
    :returns: The updated agent, or ``None`` if the modified walk cannot be applied.
    :rtype: PedestrianAgent | None
    """
    excursions: List[Excursion] = list(agent.route.excursions)
    if excursion is None:
        del excursions[index]
    else:
        excursions[index] = excursion
    route: WalkRoute = rehomed(replace(agent.route, excursions=tuple(excursions)), network, rng, config, start=True)
    return on_route(agent, route, network)


def mut_shift(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
              config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Move one walk point along its sidewalk by a normal step.

    Start and end steps use ``config.sigma_s``; excursion entry and exit steps
    use ``config.sigma_curb`` and must preserve the excursion's kind.

    Endpoints must satisfy ``_within_approach``: with excursions, they stay
    within ``config.max_approach`` of the first entry and last arrival.
    Without excursions, their gap stays within generation's bounds.
    Steps outside these bounds are rejected to prevent walk lengths from
    drifting outward (san_mateo, 20 neutral generations: median 52 -> 78 m).

    :param PedestrianAgent agent: Agent whose walk point is mutated.
    :param WalkNetwork network: Walk network used to wrap positions, check bounds, and apply the walk.
    :param Random rng: Random number generator used to select a point and sample its step.
    :param PedestrianOperatorConfig config: Configuration defining step sizes and endpoint distance bounds.
    :returns: The mutant, or ``None`` if the modified walk is invalid or out of range.
    :rtype: PedestrianAgent | None
    """
    route: WalkRoute = agent.route
    choices: List[str | Tuple[int, str]] = ["start", "end"] + [(index, side) for index in range(len(route.excursions)) for side in ("entry", "exit")]
    choice: str | Tuple[int, str] = rng.choice(choices)
    if choice in ("start", "end"):
        point: CurbPoint = getattr(route, choice)
        moved: CurbPoint = network.wrapped(point.sidewalk, point.s + rng.gauss(0, config.sigma_s))
        candidate: WalkRoute = replace(route, **{choice: moved})
        if not _within_approach(candidate, network, config):
            return None
        return on_route(agent, candidate, network)
    index: int
    side: str
    index, side = choice
    excursion: Excursion = route.excursions[index]
    point = getattr(excursion, side)
    moved = network.wrapped(point.sidewalk, point.s + rng.gauss(0, config.sigma_curb))
    excursions: Tuple[Excursion, ...] = route.excursions[:index] + (replace(excursion, **{side: moved}),) + route.excursions[index + 1:]
    candidate = replace(route, excursions=excursions)
    if not _within_approach(candidate, network, config):
        return None
    return on_route(agent, candidate, network)


def _within_approach(route: WalkRoute, network: WalkNetwork,
                     config: PedestrianOperatorConfig) -> bool:
    """
    Check whether the walk's endpoints satisfy generation distance bounds.

    With excursions, endpoints must lie within ``config.max_approach`` of the
    first entry and last arrival. Without excursions, their gap must lie between
    ``MIN_WALK`` and ``max(MIN_WALK, 2 * config.max_approach)``.

    :param WalkRoute route: Walk whose endpoints are checked.
    :param WalkNetwork network: Walk network used to measure endpoint gaps.
    :param PedestrianOperatorConfig config: Configuration defining the maximum approach distance.
    :returns: ``True`` if both endpoints satisfy the generation bounds.
    :rtype: bool
    """
    if not route.excursions:
        return MIN_WALK <= network.gap(route.start, route.end) <= max(MIN_WALK, 2 * config.max_approach)
    return (network.gap(route.start, route.excursions[0].entry) <= config.max_approach
            and network.gap(route.end, route.excursions[-1].arrival) <= config.max_approach)


def mut_redraw(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
               config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent | None:
    """
    Redraw one excursion of the same kind, with its new entry within
    ``config.redraw_reach`` of the old entry along the sidewalk.

    :param PedestrianAgent agent: Agent whose excursion is redrawn.
    :param WalkNetwork network: Walk network used to propose an excursion and apply the modified walk.
    :param Random rng: Random number generator used to select and redraw an excursion.
    :param PedestrianOperatorConfig config: Configuration defining redraw reach and endpoint adjustment bounds.
    :returns: The mutant, or ``None`` if there is no excursion, no replacement fits,
        or the modified walk cannot be applied.
    :rtype: PedestrianAgent | None
    """
    if not agent.route.excursions:
        return None
    index: int = rng.randrange(len(agent.route.excursions))
    excursion: Excursion = agent.route.excursions[index]
    pair: Tuple[CurbPoint, CurbPoint] | None = network.propose(excursion.kind, rng, excursion.entry, config.redraw_reach)
    if pair is None:
        return None
    return _with_excursion(agent, index, replace(excursion, entry=pair[0], exit=pair[1]), network, rng, config)


def mut_kind(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
             config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
             drawn: ExcursionKind | None = None) -> PedestrianAgent | None:
    """
    Change one excursion's kind and redraw its entry and exit.

    Draw the kind from ``config.kind_weights`` and propose an excursion entering
    within ``config.redraw_reach`` of the old entry. A mutation event draws the
    kind once (``drawn``); drawing the selected excursion's current kind is a no-op.

    :param PedestrianAgent agent: Agent whose excursion kind is mutated.
    :param WalkNetwork network: Walk network used to propose an excursion and apply the modified walk.
    :param Random rng: Random number generator used to select an excursion, sample a kind, and propose endpoints.
    :param PedestrianOperatorConfig config: Configuration defining kind weights, redraw reach, and endpoint adjustment bounds.
    :param ExcursionKind | None drawn: Kind already drawn for this mutation event, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if there is no excursion, no kind is available,
        the selected kind is unchanged, no proposal fits, or the modified walk cannot be applied.
    :rtype: PedestrianAgent | None
    """
    if not agent.route.excursions:
        return None
    if drawn is None:
        drawn = _draw_kind(network, rng, config)
        if drawn is None:
            return None
    index: int = rng.randrange(len(agent.route.excursions))
    excursion: Excursion = agent.route.excursions[index]
    if drawn is excursion.kind:
        return None
    pair: Tuple[CurbPoint, CurbPoint] | None = network.propose(drawn, rng, excursion.entry, config.redraw_reach)
    if pair is None:
        return None
    return _with_excursion(agent, index, replace(excursion, kind=drawn, entry=pair[0], exit=pair[1]), network, rng,
                           config)


def mut_count(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
              config: PedestrianOperatorConfig = PedestrianOperatorConfig(), action: str | None = None,
              drawn: ExcursionKind | None = None, accept: float | None = None) -> PedestrianAgent | None:
    """
    Add an excursion after the last, or remove the last excursion.

    An added excursion enters within ``config.max_link`` of the walk's end.
    For a walk without excursions, it is proposed anywhere and the start is
    redrawn near its entry. The end is redrawn near the added excursion's exit
    or the removed excursion's entry.

    The count change is a Metropolis step under ``config.count_weights``:
    moving from ``n`` to ``m`` excursions is accepted with probability
    ``min(1, weight(m) / weight(n))`` when the current weight is positive.
    Target counts outside the allowed range or with nonpositive weight are rejected.

    Draw ``action``, the added kind ``drawn``, and the uniform ``accept`` once
    per mutation event and reuse them across retries. Redrawing acceptance on
    retries previously increased departures from common one-excursion walks
    (shalun: 66% -> 56% in 17 generations).

    :param PedestrianAgent agent: Agent whose excursion count is mutated.
    :param WalkNetwork network: Walk network used to propose excursions and apply the modified walk.
    :param Random rng: Random number generator used for unspecified event draws, excursion proposals, and endpoint sampling.
    :param PedestrianOperatorConfig config: Configuration defining count and kind weights, link reach, and endpoint bounds.
    :param str | None action: Action from ``COUNT_ACTIONS``, or ``None`` to sample one.
    :param ExcursionKind | None drawn: Kind for an added excursion, or ``None`` to sample one; unused for removal.
    :param float | None accept: Uniform draw in [0, 1) for Metropolis acceptance, or ``None`` to sample one when needed.
    :returns: The mutant, or ``None`` if the action cannot apply, the count change
        is rejected, no proposal fits, or the modified walk cannot be applied.
    :rtype: PedestrianAgent | None
    :raises ValueError: If ``action`` is not in ``COUNT_ACTIONS``.
    """
    action = rng.choice(COUNT_ACTIONS) if action is None else action
    if action not in COUNT_ACTIONS:
        raise ValueError(f"action must be one of {COUNT_ACTIONS}, got {action!r}")
    route: WalkRoute = agent.route
    count: int = len(route.excursions)
    target: int = count + (1 if action == "add" else -1)
    weight: Dict[int, float] = {int(key): value for key, value in config.count_weights.items()}
    if not 0 <= target <= MAX_EXCURSIONS or weight.get(target, 0.0) <= 0:
        return None
    if weight.get(count, 0.0) > 0 and (rng.random() if accept is None else accept) >= weight[target] / weight[count]:
        return None
    if action == "remove":
        if not route.excursions:
            return None
        last: Excursion = route.excursions[-1]
        end: CurbPoint = network.wrapped(last.entry.sidewalk, last.entry.s + rng.uniform(-config.max_approach, config.max_approach))
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
        pair: Tuple[CurbPoint, CurbPoint] | None = network.propose(drawn, rng)
        if pair is None:
            return None
        start: CurbPoint = network.wrapped(pair[0].sidewalk, pair[0].s + rng.uniform(-config.max_approach, config.max_approach))
    else:
        pair = network.propose(drawn, rng, route.end, config.max_link)
        if pair is None or pair[0].sidewalk != route.end.sidewalk:
            return None
        start = route.start
    end = network.wrapped(pair[1].sidewalk, pair[1].s + rng.uniform(-config.max_approach, config.max_approach))
    added: WalkRoute = replace(route, start=start, excursions=route.excursions + (Excursion(drawn, *pair),), end=end)
    return on_route(agent, added, network)


STOP_ACTIONS: Tuple[str, ...] = ("toggle", "dwell")
"""``mut_walk_stop``'s actions, drawn with equal chance once per mutation event."""

_WAITS: Tuple[str, ...] = ("curb", "island", "hesitation")
"""Wait categories available for pedestrian mutation."""

def _wait_chance(kind: str) -> float:
    """
    Return the occurrence probability for a pedestrian wait category.

    :param str kind: Wait category: ``curb``, ``island``, or ``hesitation``.
    :returns: The category's occurrence probability.
    :rtype: float
    :raises KeyError: If ``kind`` is not a supported wait category.
    """
    return {"curb": CURB_WAIT_PROBABILITY, "island": ISLAND_WAIT_PROBABILITY,
            "hesitation": HESITATION_PROBABILITY}[kind]


def mut_walk_stop(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  action: str | None = None, choice: Tuple[float, str, float] | None = None) -> PedestrianAgent | None:
    """
    Add or remove a wait of a kind used at generation, or adjust a stop's dwell.

    ``"toggle"`` selects an excursion and a wait kind: curb, island, or
    hesitation. An existing wait and its resume point ``RESUME_GAP`` later
    are removed together; otherwise, a wait is added where generation puts it.
    Adding is accepted with probability ``min(1, q / (1 - q))`` and removing
    with ``min(1, (1 - q) / q)``, where ``q`` is the kind's generation probability.

    ``"dwell"`` adjusts one stop's dwell by a normal step using ``config.sigma_t``.
    Draw the action and ``choice`` once per mutation event and reuse them across
    retries to avoid favouring whichever toggle direction succeeds more often.

    The shared ``profile_operators.mut_stop`` previously removed waits more often
    than it added them (borregas: 0.62 -> 0.30 waits per walker in 30 generations)
    and resumed at speeds up to the walker type's maximum.

    :param PedestrianAgent agent: Agent whose waits or stop dwell are mutated.
    :param WalkNetwork network: Walk network used to locate waits and construct the trajectory.
    :param Random rng: Random number generator used for unspecified event draws and wait mutation.
    :param PedestrianOperatorConfig config: Configuration controlling dwell steps and stop duration bounds.
    :param str | None action: Mutation action, ``"toggle"`` or ``"dwell"``, or ``None`` to sample one.
    :param Tuple[float, str, float] | None choice: Cached excursion-selection uniform,
        wait kind, and acceptance uniform, or ``None`` to draw them.
    :returns: The mutant, or ``None`` if no change applies or the step is rejected.
    :rtype: PedestrianAgent | None
    """
    action = rng.choice(STOP_ACTIONS) if action is None else action
    if action not in STOP_ACTIONS:
        raise ValueError(f"action must be one of {STOP_ACTIONS}, got {action!r}")
    profile: Tuple[SpeedPoint, ...] = agent.speed_profile
    if action == "dwell":
        stops: List[int] = [index for index, point in enumerate(profile) if point.speed == 0]
        if not stops:
            return None
        index: int = rng.choice(stops)
        point: SpeedPoint = replace(profile[index], dwell=gauss_step(rng, profile[index].dwell, config.sigma_t, config.max_dwell))
        return with_profile(agent, profile[:index] + (point,) + profile[index + 1:])
    route: WalkRoute = agent.route
    if not route.excursions:
        return None
    if choice is None:
        choice = (rng.random(), rng.choice(_WAITS), rng.random())
    pick: float
    kind: str
    accept: float
    pick, kind, accept = choice
    index = min(int(pick * len(route.excursions)), len(route.excursions) - 1)
    name: str = f"x{index}"
    frame: WalkFrame = network.frame(route)
    element: WalkElement = frame.elements[frame.index[name]]
    line: LineString = LineString(element.coords)
    curb: float = min(network.curb_distance(line), element.length)
    room: float = element.length - RESUME_GAP - 0.1
    islands: List[Tuple[float, float]] = network.island_spans(line)

    def in_kind(s: float) -> bool:
        """
        Check whether a path position belongs to the selected wait category.

        :param float s: Distance along the path, in metres.
        :returns: ``True`` if the position falls within the selected curb, island, or hesitation region.
        :rtype: bool
        """
        if kind == "curb":
            return s <= curb + GUTTER_RANGE[1] + 1e-6
        if kind == "island":
            return any(low - 1e-6 <= s <= high + 1e-6 for low, high in islands)
        return curb + GUTTER_RANGE[1] < s and not any(low - 1e-6 <= s <= high + 1e-6 for low, high in islands)

    present: List[int] = [i for i, point in enumerate(profile) if point.speed == 0 and point.lane_id == name and in_kind(point.s)]
    q: float = _wait_chance(kind)
    if present:
        if accept >= min(1.0, (1 - q) / q):
            return None
        i: int = present[0]
        drop: Set[int] = {i} | ({i + 1} if i + 1 < len(profile) and profile[i + 1].lane_id == name
                      and profile[i + 1].speed > 0 and abs(profile[i + 1].s - profile[i].s - RESUME_GAP) < 1e-6 else set())
        kept: Tuple[SpeedPoint, ...] = tuple(point for j, point in enumerate(profile) if j not in drop)
        if not kept or kept[-1].speed == 0:
            return None
        return with_profile(agent, kept)
    if accept >= min(1.0, q / (1 - q)) or len(profile) + 2 > MAX_SPEED_POINTS:
        return None
    if kind == "curb":
        at: float = curb + rng.uniform(*GUTTER_RANGE) if rng.random() < GUTTER_SHARE else max(curb - CURB_STAND_BACK, 0.0)
        dwell: float = rng.uniform(0, MAX_CURB_WAIT)
    elif kind == "island":
        if not islands:
            return None
        low: float
        high: float
        low, high = rng.choice(islands)
        at, dwell = (low + high) / 2, rng.uniform(*ISLAND_WAIT)
    else:
        if room - 1 <= curb + 1:
            return None
        at, dwell = rng.uniform(curb + 1, room - 1), rng.uniform(*HESITATION)
    if not 0 < at <= room:
        return None
    position: float = frame.distance(name, at)
    if any(abs(frame.distance(point.lane_id, point.s) - position) < STOP_SPACING + RESUME_GAP for point in profile):
        return None
    moving: List[float] = [point.speed for point in profile if point.speed > 0]
    after: List[SpeedPoint] = [point for point in profile if frame.distance(point.lane_id, point.s) > position and point.speed > 0]
    resume: float = after[0].speed if after else moving[-1]
    added: List[SpeedPoint] = [SpeedPoint(name, at, 0.0, dwell), SpeedPoint(name, at + RESUME_GAP, resume)]
    placed: List[Tuple[float, int, SpeedPoint]] = [(frame.distance(point.lane_id, point.s), rank, point) for rank, point in enumerate(list(profile) + added)]
    result: Tuple[SpeedPoint, ...] = ordered_points(placed, profile[0])
    if len(result) != len(profile) + 2:
        return None
    return with_profile(agent, result)


TURN_BACK_ACTIONS: Tuple[str, ...] = ("step", "redraw")
"""``mut_turn_back``'s actions, drawn with equal chance once per mutation event."""


def mut_turn_back(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                  config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                  action: str | None = None, draw: float | None = None) -> PedestrianAgent | None:
    """
    Step the last excursion's turning point or redraw whether it turns back.

    ``"step"`` moves an existing turning point by a normal step using
    ``config.sigma_turn_back``; it does nothing if no turning point exists.
    ``"redraw"`` draws a turning point as at generation with probability
    ``config.turn_back_probability``, otherwise removing any existing one.

    Draw the action with equal probability once per mutation event, and reuse
    the redraw uniform ``draw`` across retries. Redrawing this uniform on each
    retry previously increased the share of walkers turning back
    (borregas: 4.8% -> 8.5% in 30 generations).

    :param PedestrianAgent agent: Agent whose last excursion's turn-back behaviour is mutated.
    :param WalkNetwork network: Walk network used to apply the modified walk.
    :param Random rng: Random number generator used for unspecified event draws and turning-point sampling.
    :param PedestrianOperatorConfig config: Configuration defining turning-point step size and turn-back probability.
    :param str | None action: Action from ``TURN_BACK_ACTIONS``, or ``None`` to sample one.
    :param float | None draw: Uniform draw in [0, 1) deciding whether a redraw
        produces a turning point, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if the walk has no excursion or no change applies.
    :rtype: PedestrianAgent | None
    """
    if not agent.route.excursions:
        return None
    action = rng.choice(TURN_BACK_ACTIONS) if action is None else action
    if action not in TURN_BACK_ACTIONS:
        raise ValueError(f"action must be one of {TURN_BACK_ACTIONS}, got {action!r}")
    last: Excursion = agent.route.excursions[-1]
    length: float = network.excursion_line(last.kind, last.entry, last.exit).length
    high: float = length - MIN_TURN_BACK
    if high < MIN_TURN_BACK:
        return None
    if action == "step":
        if last.turn_back is None:
            return None
        turn_back: float | None = min(max(last.turn_back + rng.gauss(0, config.sigma_turn_back), MIN_TURN_BACK), high)
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
    Resample the walker type from ``config.walker_type_weights``, mapping
    positive target speeds to the new type's prior through ``retyped`` and
    capping them at its maximum. Stop points stay unchanged.

    Draw once per mutation event and pass the result as ``drawn``. Drawing the
    current type is a no-op; redrawing until it changes would flatten the mix.

    :param PedestrianAgent agent: Agent whose walker type and target speeds are mutated.
    :param Random rng: Random number generator used when ``drawn`` is not provided.
    :param PedestrianOperatorConfig config: Configuration defining walker-type sampling weights.
    :param WalkerType | None drawn: Type already drawn for this mutation event, or ``None`` to sample one.
    :returns: The mutant, or ``None`` if the drawn type is unchanged.
    :rtype: PedestrianAgent | None
    """
    walker_type: WalkerType = drawn if drawn is not None else WalkerType[draw_type(rng, config.walker_type_weights)]
    return None if walker_type is agent.walker_type else retyped(agent, walker_type)


def mut_walk_style(agent: PedestrianAgent, rng: Random) -> PedestrianAgent:
    """
    Move the walk style by a reflected normal step within [0, 1]
    through ``realism.style_step``, preserving its uniform prior.

    :param PedestrianAgent agent: Agent whose walk style is mutated.
    :param Random rng: Random number generator used to sample the style step.
    :returns: The agent with the updated walk style.
    :rtype: PedestrianAgent
    """
    return replace(agent, walk_style=style_step(rng, agent.walk_style))


def _draw_kind(network: WalkNetwork, rng: Random, config: PedestrianOperatorConfig) -> ExcursionKind | None:
    """
    Draw an excursion kind from positive configured weights available on the network.

    :param WalkNetwork network: Walk network defining the available excursion kinds.
    :param Random rng: Random number generator used for weighted sampling.
    :param PedestrianOperatorConfig config: Configuration defining excursion-kind weights.
    :returns: The drawn kind, or ``None`` if no available kind has positive weight.
    :rtype: ExcursionKind | None
    """
    available: Set[str] = {kind.name for kind in network.kinds()}
    weights: Dict[str, float] = {name: weight for name, weight in config.kind_weights.items() if name in available and weight > 0}
    return ExcursionKind[draw_type(rng, weights)] if weights else None


def _same_stops(parent: PedestrianAgent, child: PedestrianAgent | None) -> PedestrianAgent | None:
    """
    Keep a child only if it preserves the parent's stop count and does not
    replace a moving start with a stop.

    Wait counts change only through ``mut_walk_stop``, preserving generation
    rates. Speed-point additions, removals, and moves previously lost 0.09
    waits per application on borregas.

    :param PedestrianAgent parent: Agent providing the required stop count and initial movement condition.
    :param PedestrianAgent | None child: Proposed child to check, or ``None`` if no child was produced.
    :returns: The child if both conditions hold, otherwise ``None``.
    :rtype: PedestrianAgent | None
    """
    if child is None:
        return None
    count: Callable[[PedestrianAgent], int] = lambda agent: sum(point.speed == 0 for point in agent.speed_profile)
    if child.speed_profile[0].speed == 0 and parent.speed_profile[0].speed > 0:
        # Removing the walking speed before a curb wait left profiles that start
        # with the wait (shalun: 19% of adults after 30 generations).
        return None
    return child if count(child) == count(parent) else None


def mut_pedestrian(agent: PedestrianAgent, network: WalkNetwork, rng: Random,
                   config: PedestrianOperatorConfig = PedestrianOperatorConfig()) -> PedestrianAgent:
    """
    Mutate each gene independently with its own ``config.mutation_rates``
    probability, using an operator drawn by ``config.mutation_weights``.

    Every step keeps the mutant acceptable. If no mutation succeeds, return
    ``agent`` unchanged.

    :param PedestrianAgent agent: Agent whose genes are mutated.
    :param WalkNetwork network: Walk network used to construct and validate the mutant.
    :param Random rng: Random number generator used for gene selection, operator selection, and mutation.
    :param PedestrianOperatorConfig config: Configuration controlling mutation rates, operator weights, and acceptance.
    :returns: An acceptable mutant, or ``agent`` unchanged if no mutation succeeds.
    :rtype: PedestrianAgent
    """
    # One draw of each per mutation event, kept over retries.
    kind: ExcursionKind | None = _draw_kind(network, rng, config)
    added: ExcursionKind | None = _draw_kind(network, rng, config)
    count_action: str = rng.choice(COUNT_ACTIONS)
    count_accept: float = rng.random()
    turn_back_action: str = rng.choice(TURN_BACK_ACTIONS)
    turn_back_draw: float = rng.random()
    point_action: str = rng.choice(SPEED_POINT_ACTIONS)
    stop_action: str = rng.choice(STOP_ACTIONS)
    stop_choice: Tuple[float, str, float] = (rng.random(), rng.choice(_WAITS), rng.random())
    walker_type: WalkerType = WalkerType[draw_type(rng, config.walker_type_weights)]
    operators: Dict[str, Callable[[PedestrianAgent], PedestrianAgent | None]] = {
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
