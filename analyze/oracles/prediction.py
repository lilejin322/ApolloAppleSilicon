"""
Prediction: trajectories Planning cannot use, and vehicles and bicycles near
the ego whose way Prediction got wrong.

Planning makes an obstacle of every trajectory Prediction gives a road user,
whatever its probability, and a static one of a road user it gives none
(``Obstacle::CreateObstacles``, ``planning_base/common/obstacle.cc``). A road
user is mispredicted only when every trajectory is wrong.

Pedestrians are not judged. Their predicted paths stray from the walked ones
smoothly, with no gap between right and wrong to split at (2 m off was 18 to
32 degrees off course in 21025 walking predictions), and ours turn at
walkway corners without a sign beforehand, so what looked wrong was a turn
made after the prediction.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import hypot, isnan
from typing import Dict, List, Sequence, Set, Tuple

from google.protobuf.message import Message
from shapely.geometry import LineString, Point, Polygon

from ...utils import ApolloVehicleConfig
from ..footprint import box, ego_footprint
from ..replay import OBSTACLES, PREDICTION, World
from .base import Oracle, Violation

__all__ = ["GO", "HORIZON", "JUDGED", "LAPSE", "MispredictionOracle", "NEAR", "PredictionNaNOracle", "SIDEWAYS",
           "STAY", "TrackPoint", "mispredicted"]

HORIZON = 3.0
"""Seconds after a prediction over which it is set against what the road user did."""
SIDEWAYS = 3.0
"""
Metres off every predicted path at which a road user went another way: a lane over. Paths are compared along
their length, so a road user faster or slower than predicted is not off its path. Over the 392 records of the
2026-11-01/02 searches, every real wrong way (a turn from the turn lane predicted straight, a fork driven
straight through predicted as a right turn) went 3.5 m or more off, while a car predicted along the centre of
the lane it was moving into, and going there, strayed at most 2.1 m. A sum over time discounted towards the
near term ranked such a wobble above a slow bicycle's real left turn, so the most off decides; how far ahead
a prediction is held to is ``HORIZON`` and ``UNDER_WAY``.
"""
JUDGED = frozenset({"VEHICLE", "BICYCLE"})
"""The road users judged: not pedestrians (see the module's notes)."""
UNDER_WAY = 0.3
"""Metres off every predicted path within its first ``EARLY`` metres: it had begun to go another way when predicted."""
EARLY = 4.0
"""
Metres along a road user's track within which ``UNDER_WAY`` is measured. Along the track, not over time: on a
turn of radius R the offset after s metres is s^2 / 2R whatever the speed, so a car turning at 3 m/s shows the
same turn in 4 m that one at 9 m/s does, where in 0.5 s it showed a sixth of it and a real 8 m turn flickered
in and out of being judged. By 4 m a turn begun at the prediction is 0.5 to 1 m off on the junctions' 8 to
15 m radii, one begun more than about 1.3 m later under 0.3 m.
"""
LAPSE = 0.5
"""
Source-time seconds from the first wrong prediction of a road user to a later one, with none right between:
Planning plans anew every cycle, so a prediction wrong for a cycle or two, as a moving car given no trajectory
in one frame and one again in the next, is ridden over. Receive-time delays must not extend this duration.
"""
MAX_PREDICTION_GAP = 0.25
"""Maximum source-time gap in a run of wrong predictions; allows one missing cycle at the usual 10 Hz."""
STAY = 0.5
"""Metres within which a road user stays where it is over the horizon."""
GO = 1.5
"""Metres from which a road user moves over the horizon."""
MOVING = 1.0
"""
m/s of published speed from which a road user was moving when predicted: above the speed below which
Prediction takes a vehicle as still and gives it no trajectory
(``FLAGS_still_obstacle_speed_threshold``, 0.99), so slower is still by its design. Read at the predicted
frame, as Prediction reads it: a car slowing through 1 m/s at a stop line is still by its rule, and its
restart could not be foreseen.
"""
BICYCLE_MOVING = 0.2
"""m/s from which a bicycle is moving, matching Apollo's ``FLAGS_still_pedestrian_speed_threshold``."""
STANDING = 0.1
"""m/s of published speed below which a road user was standing when predicted."""
NEAR = 2.0
"""Metres from the ego, or from where the ego was yet to drive within the horizon, of a road user that matters."""
_REACH = 130.0
"""Metres from the ego beyond which a prediction is not kept: neither drives that far towards the other in it."""

TrackPoint = Tuple[float, float, float]
"""``(t, x, y)``: seconds after the prediction, and a centre, map frame."""


@dataclass(frozen=True)
class _Frame:
    """
    One obstacle frame as the oracle keeps it.

    :param float stamp: Its header stamp, wall clock: what ``perception_obstacle.timestamp`` refers to.
    :param float t: Seconds since the record began.
    :param Tuple[float, float, Polygon] | None ego: The ego's reference point ``(x, y)`` and body; None before
        the first localization.
    :param Dict[int, Tuple[float, float, float, Polygon, str]] obstacles: Each obstacle's centre ``(x, y)``,
        published speed in m/s, body and name, e.g. ``"VEHICLE 1003"``, by id.
    """
    stamp: float
    """Its header stamp, wall clock: what ``perception_obstacle.timestamp`` refers to."""
    t: float
    """Seconds since the record began."""
    ego: Tuple[float, float, Polygon] | None
    """The ego's reference point ``(x, y)`` and body; None before the first localization."""
    obstacles: Dict[int, Tuple[float, float, float, Polygon, str]]
    """Each obstacle's centre ``(x, y)``, published speed in m/s, body and name, e.g. ``"VEHICLE 1003"``, by id."""


@dataclass(frozen=True)
class _Predicted:
    """
    One road user's prediction, waiting for its horizon to pass.

    :param float stamp: The stamp of the obstacle frame it predicts from.
    :param float t: Seconds since the record began: when the prediction came.
    :param int obstacle_id: The road user's id.
    :param Tuple[Tuple[TrackPoint, ...], ...] modes: Its predicted trajectories, each in time order; none when
        Prediction gave it none, which Planning takes as static.
    """
    stamp: float
    """The stamp of the obstacle frame it predicts from."""
    t: float
    """Seconds since the record began: when the prediction came."""
    obstacle_id: int
    """The road user's id."""
    modes: Tuple[Tuple[TrackPoint, ...], ...] | None
    """Its predicted trajectories, each in time order; none when Prediction gave it none, which Planning takes as
    static. None marks an unusable prediction and interrupts a run."""


@dataclass
class _Wrong:
    """
    A road user's predictions wrong running: the first, and the closest it has come to the ego since.

    :param float t: Seconds since the record began: when the first came.
    :param float stamp: Source timestamp of the first wrong prediction.
    :param float last_stamp: Source timestamp of the latest wrong prediction in the run.
    :param str what: What it got wrong, for people (``mispredicted``).
    :param str name: The road user, e.g. ``"VEHICLE 1003"``.
    :param float | None x: The ego's reference point then, map frame; None without an ego state.
    :param float | None y: The same, its Y coordinate.
    :param float gap: Metres: the closest the road user came to the ego, or to where it drove next, within the
        horizon of any prediction of the run.
    """
    t: float
    """Seconds since the record began: when the first came."""
    stamp: float
    """Source timestamp of the first wrong prediction."""
    last_stamp: float
    """Source timestamp of the latest wrong prediction in this run."""
    what: str
    """What it got wrong, for people (``mispredicted``)."""
    name: str
    """The road user, e.g. ``"VEHICLE 1003"``."""
    x: float | None
    """The ego's reference point then, map frame; None without an ego state."""
    y: float | None
    """The same, its Y coordinate."""
    gap: float
    """
    Metres: the closest the road user came to the ego, or to where it drove next, within the horizon of any
    prediction of the run.
    """


def mispredicted(modes: Sequence[Sequence[TrackPoint]], track: Sequence[TrackPoint], speed: float, *,
                 horizon: float = HORIZON, obstacle_type: str = "VEHICLE") -> str | None:
    """
    How every predicted trajectory of a road user was wrong about where it went within the horizon.

    Wrong is one of three, each only once it was under way when predicted,
    since nothing foretells a road user starting, stopping or turning before
    it does: every trajectory kept it where it was (``STAY``) while it moved
    (``GO``), ``MOVING`` when predicted; every trajectory moved it while it
    stayed, ``STANDING`` when predicted; or it went ``SIDEWAYS`` off every path
    that moved it, ``UNDER_WAY`` off them already within its first ``EARLY``
    metres. A path is
    set where the road user was when predicted, so where a trajectory starts
    does not count, only its shape; and it is compared along its length, so how
    fast the road user went along it is not judged either: Prediction keeps
    speed (``FLAGS_enable_lane_sequence_acc`` is off), and so missed every speed
    change, no less than plain constant-velocity extrapolation did.

    :param Sequence[Sequence[TrackPoint]] modes: Its predicted trajectories; none for a road user given none,
        which Planning takes as static.
    :param Sequence[TrackPoint] track: Its actual centre from the prediction (``t`` 0) through the horizon.
    :param float speed: Its published speed in m/s at the predicted frame: what Prediction read.
    :param float horizon: Seconds after the prediction judged.
    :param str obstacle_type: Perception type, used for the vehicle or bicycle moving-speed threshold.
    :returns: What every trajectory got wrong, for people; None when one foresaw it or it was not yet under way.
    :rtype: str | None
    """
    _, x0, y0 = track[0]
    moved = hypot(track[-1][1] - x0, track[-1][2] - y0)
    paths = [tuple((t, x - mode[0][1] + x0, y - mode[0][2] + y0) for t, x, y in mode) for mode in modes]
    reach = [max((hypot(x - x0, y - y0) for t, x, y in path if t <= horizon + 1e-6), default=0.0) for path in paths]
    if all(r < STAY for r in reach):
        moving = BICYCLE_MOVING if obstacle_type == "BICYCLE" else MOVING
        if moved >= GO and speed >= moving:
            return f"every trajectory kept it in place, but it moved {moved:.1f} m"
        return None
    if all(r >= GO for r in reach) and moved < STAY and speed < STANDING:
        return f"every trajectory moved it at least {min(reach):.1f} m, but it stayed within {moved:.1f} m"
    if any(r < STAY for r in reach) and moved < GO:
        return None  # A trajectory keeping it in place was about right.
    offs = [_off(path, track) for path, r in zip(paths, reach) if r >= STAY]
    if all(early > UNDER_WAY and most >= SIDEWAYS for early, most in offs):
        return f"it went {min(most for _, most in offs):.1f} m off every predicted path"
    return None


def _off(path: Sequence[TrackPoint], track: Sequence[TrackPoint]) -> Tuple[float, float]:
    """
    How far a road user went sideways off one predicted path, along as much of its track as the path covers.

    :param Sequence[TrackPoint] path: The predicted trajectory, set where the road user was when predicted.
    :param Sequence[TrackPoint] track: Its actual centre from the prediction on.
    :returns: ``(early, most)``: the most metres off within the track's first ``EARLY`` metres, and the most
        metres off.
    :rtype: Tuple[float, float]
    """
    line = LineString([(x, y) for _, x, y in path])
    early = most = along = 0.0
    for (_, xa, ya), (_, x, y) in zip(track, track[1:]):
        along += hypot(x - xa, y - ya)
        if along > line.length + 1e-6:
            break
        d = line.distance(Point(x, y))
        most = max(most, d)
        if along <= EARLY + 1e-6:
            early = max(early, d)
    return early, most


def _near(obstacle_id: int, window: Sequence[_Frame]) -> float:
    """
    How close a road user came to the ego, or to where the ego was yet to drive, over a window of frames.

    :param int obstacle_id: The road user's id.
    :param Sequence[_Frame] window: The frames from the prediction through the horizon.
    :returns: Metres between its body and the ego's body then or later in the window; infinite without both.
    :rtype: float
    """
    egos = [frame.ego[2] if frame.ego is not None else None for frame in window]
    closest = float("inf")
    for k, frame in enumerate(window):
        other = frame.obstacles.get(obstacle_id)
        if other is None:
            continue
        for ego in egos[k:]:
            if ego is not None:
                closest = min(closest, ego.distance(other[3]))
                if closest == 0.0:
                    return 0.0
    return closest


class MispredictionOracle(Oracle):
    """
    Report each road user near the ego whose way every predicted trajectory
    got wrong for ``LAPSE`` seconds running, once, at the first (``mispredicted``).

    Near is within ``NEAR`` of the ego, or of where the ego was yet to drive,
    within the horizon of any prediction of the run: a road user Planning
    planned around as it was predicted to go, which went elsewhere close to the
    ego. It counts against the ADS, whose Prediction it judges.

    A prediction is judged once the obstacle frames through its horizon have
    come, then dropped; those of the record's last ``horizon`` seconds are not.
    Duration uses obstacle source timestamps. An unusable prediction or a gap
    exceeding ``max_prediction_gap`` interrupts the run. Duplicate and older
    source frames are ignored. Report times remain record-relative receive times.

    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param float horizon: Seconds after a prediction to judge.
    :param float max_prediction_gap: Maximum gap in source seconds, defaulting to 0.25 for 10 Hz predictions.
    """
    kind = "misprediction"
    """The ``Violation.kind`` it reports; the subject is the road user, e.g. ``"VEHICLE 1003"``."""
    channels = frozenset({OBSTACLES, PREDICTION})
    """The obstacle frames, for where everyone went, and the predictions."""
    vehicle: ApolloVehicleConfig
    """The ego's dimensions."""
    horizon: float
    """Seconds after a prediction to judge."""
    _frames: List[_Frame]
    """The obstacle frames still needed, in time order."""
    _pending: List[_Predicted]
    """The predictions waiting for their horizon, in time order."""
    _candidates: Dict[int, _Wrong]
    """Per road user whose predictions are wrong running, the first and the closest it has come to the ego."""
    _reported: Set[int]
    """The road users reported."""

    def __init__(self, vehicle: ApolloVehicleConfig, *, horizon: float = HORIZON,
                 max_prediction_gap: float = MAX_PREDICTION_GAP) -> None:
        """
        Keep the ego's dimensions and the horizon; start with nothing pending.

        :param ApolloVehicleConfig vehicle: The ego's dimensions.
        :param float horizon: Seconds after a prediction to judge.
        :param float max_prediction_gap: Maximum source-time interval between consecutive wrong predictions.
        """
        super().__init__()
        self.vehicle = vehicle
        self.horizon = horizon
        self.max_prediction_gap = max_prediction_gap
        self._frames = []
        self._pending = []
        self._candidates = {}
        self._reported = set()
        self._last_judged: Dict[int, float] = {}

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Keep an obstacle frame and judge the predictions whose horizon it closes, or keep new predictions.

        :param World world: The world at the message.
        :param str channel: ``OBSTACLES`` or ``PREDICTION``.
        :param Message message: The ``PerceptionObstacles``, or the ``PredictionObstacles``.
        """
        ego = world.ego
        if channel == OBSTACLES:
            mine = None if ego is None else (ego.x, ego.y, ego_footprint(ego.x, ego.y, ego.heading, self.vehicle))
            self._frames.append(_Frame(message.header.timestamp_sec, world.t, mine,
                                       {o.id: (o.x, o.y, o.speed, box(o.x, o.y, o.heading, o.length, o.width),
                                               f"{o.type} {o.id}") for o in world.obstacles}))
            self._judge()
            return
        if ego is None:
            return
        for predicted in message.prediction_obstacle:
            obstacle = predicted.perception_obstacle
            kind = obstacle.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[obstacle.type].name
            if kind not in JUDGED or obstacle.id in self._reported or \
                    hypot(obstacle.position.x - ego.x, obstacle.position.y - ego.y) > _REACH:
                continue
            modes = [tuple((p.relative_time, p.path_point.x, p.path_point.y) for p in trajectory.trajectory_point)
                     for trajectory in predicted.trajectory]
            valid = tuple(mode for mode in modes if mode and not any(isnan(x) or isnan(y) for _, x, y in mode))
            if modes and not valid:
                valid = None  # Unusable evidence interrupts the run when its source time is judged.
            self._pending.append(_Predicted(obstacle.timestamp, world.t, obstacle.id, valid))

    def finish(self, world: World) -> None:
        """
        Drop the predictions whose horizon the record did not reach, and the runs of wrong ones not yet long enough.

        :param World world: The world at the last message; unused.
        """
        self._pending = []
        self._candidates = {}
        self._last_judged = {}

    def _judge(self) -> None:
        """Judge each pending prediction the frames now reach through its horizon, then drop what is done."""
        stamps = [frame.stamp for frame in self._frames]
        now = stamps[-1]
        waiting = []
        for predicted in self._pending:
            if predicted.stamp + self.horizon > now + 1e-6:
                waiting.append(predicted)
            elif predicted.obstacle_id not in self._reported:
                self._one(predicted, stamps)
        self._pending = waiting
        oldest = min([p.stamp for p in waiting] + [now - 1.0])
        self._frames = self._frames[bisect_left(stamps, oldest - 1e-6):]

    def _one(self, predicted: _Predicted, stamps: Sequence[float]) -> None:
        """
        Judge one prediction against the frames through its horizon; report its road user once its predictions
        have been wrong for ``LAPSE`` seconds running.

        :param _Predicted predicted: The prediction.
        :param Sequence[float] stamps: The kept frames' stamps.
        """
        oid = predicted.obstacle_id
        if predicted.stamp <= self._last_judged.get(oid, float("-inf")):
            return  # Repeated or older source frames provide no new evidence.
        self._last_judged[oid] = predicted.stamp
        run = self._candidates.get(oid)
        if run is not None and predicted.stamp - run.last_stamp > self.max_prediction_gap + 1e-6:
            self._candidates.pop(oid)
        i = bisect_left(stamps, predicted.stamp - 1e-3)
        if predicted.modes is None:
            self._candidates.pop(oid, None)
            return
        if i == len(stamps) or abs(stamps[i] - predicted.stamp) > 1e-3:
            self._candidates.pop(oid, None)
            return  # Its frame came before the record began.
        window = self._frames[i:bisect_right(stamps, predicted.stamp + self.horizon + 1e-6)]
        track = [(frame.stamp - predicted.stamp, *frame.obstacles[oid][:2])
                 for frame in window if oid in frame.obstacles]
        if not track or track[0][0] > 1e-3 or track[-1][0] < self.horizon - 0.2:
            self._candidates.pop(oid, None)
            return  # It was not seen through the horizon.
        origin = window[0]
        what = mispredicted(predicted.modes, track, origin.obstacles[oid][2], horizon=self.horizon,
                            obstacle_type=origin.obstacles[oid][4].split()[0])
        if what is None:
            self._candidates.pop(oid, None)  # Right again: a lapse Planning rode over.
            return
        gap = _near(oid, window)
        run = self._candidates.get(oid)
        if run is None:
            x, y = (None, None) if origin.ego is None else origin.ego[:2]
            self._candidates[oid] = _Wrong(predicted.t, predicted.stamp, predicted.stamp, what,
                                           origin.obstacles[oid][4], x, y, gap)
            return
        run.gap = min(run.gap, gap)
        run.last_stamp = predicted.stamp
        duration = predicted.stamp - run.stamp
        if duration >= LAPSE - 1e-6:
            del self._candidates[oid]
            if run.gap <= NEAR:
                self._reported.add(oid)
                self.violations.append(Violation(
                    self.kind, run.t, run.name,
                    f"Prediction at t={run.t:.2f} s was wrong about {run.name}: {run.what} within {self.horizon:g} s, "
                    f"and for {duration:.1f} s of source time after; it came within {run.gap:.1f} m of the ego or where "
                    f"it drove next", run.x, run.y))


class PredictionNaNOracle(Oracle):
    """
    Report each obstacle Prediction predicted a trajectory with NaN points
    for, once, at the first such prediction.

    Planning takes no decision for such an obstacle and drives as if it were
    not there. Our obstacles once came without ``polygon_point``, from which
    Prediction draws the semantic map of its pedestrian model, and every
    pedestrian near the ego was predicted as NaN; this oracle keeps such a gap
    from going unnoticed. Every one so far came from our setup rather than
    Apollo (that, and OpenBLAS on Apple Silicon), so the search does not count
    it.
    """

    kind = "prediction_nan"
    channels = frozenset({PREDICTION})

    def __init__(self) -> None:
        """Start with no obstacle IDs reported for invalid predictions."""
        super().__init__()
        self._reported: Set[int] = set()

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """Report each obstacle at most once when a predicted trajectory contains NaN X or Y coordinates."""
        for predicted in message.prediction_obstacle:
            obstacle = predicted.perception_obstacle
            if obstacle.id in self._reported or not any(
                    isnan(point.path_point.x) or isnan(point.path_point.y)
                    for trajectory in predicted.trajectory for point in trajectory.trajectory_point):
                continue
            self._reported.add(obstacle.id)
            name = f"{obstacle.DESCRIPTOR.fields_by_name['type'].enum_type.values_by_number[obstacle.type].name} " \
                   f"{obstacle.id}"
            self.report(world, name, f"Prediction predicted {name} as a NaN trajectory; Planning ignores it")
