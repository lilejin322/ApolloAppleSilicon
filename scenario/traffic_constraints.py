"""
Map constraints shared by replay validation and genetic acceptance.

Missing boundary geometry is not evidence of free space. These checks reject
known conflicts, but cannot certify areas the map does not describe.
"""

import numpy as np
import shapely
from math import ceil, hypot, pi
from shapely.geometry import LineString, Point
from typing import Dict, List, Sequence, Tuple

from ..map import MapService
from ..map.types import LaneMarking
from ..map.geometry import CenterlineProfile, normalize_angle
from .routing import InvalidRoute, RouteSpace
from .replay import ReplayLayout, SpeedPoint

FOOTPRINT_LATERAL_TOLERANCE = 0.2
"""Metres a body may overlap a parallel prohibiting marking on each side: coarse map
polygons put lines a few decimetres inside the lane at vertices. Smaller than the
overlap of a 2.55 m bus on a 2 m lane (0.275 m), so too-narrow lanes stay rejected."""
CROSSING_ANGLE = pi / 6
"""A marking more than this far from the body heading is crossed end-on (a line stub
of another approach at a junction), not driven over sideways, and is ignored."""

TURN_SWEEP_CURVATURE = 1 / 40
"""Mean curvature over the body length (1/m) above which a body is turning. Turning
bodies may sweep over painted lines, as long vehicles' overhangs do in tight turns;
curbs still count."""

_FOOTPRINT_CACHE_SIZE = 128
"""Footprint validation results kept per map (``MapService.derived``); no trajectory arrays retained."""


StopBlocks = Dict[str, Tuple[Tuple[float, float, str], ...] | None]
"""Per lane of a replay layout, ``(low, high, reason)`` lane-metre spans where an ordinary stop may not lie;
``None`` forbids the whole lane."""


def stop_blocks(space: RouteSpace, layout: ReplayLayout) -> StopBlocks:
    """
    Where an ordinary stop may not lie on a replay layout, by lane: the one rule stop placement
    (``replay.placed_stops``) and validation (``check_stops``) share.

    A whole connector or junction lane; on other lanes, every stretch inside a junction polygon
    (partial overlap need not make the whole lane a junction lane), and every stretch the replayed
    lane changes run along, ``STOP_CLEARANCE`` wider at each end. A lane change's stretch is its
    piece of the layout projected onto its source and target lanes, so it is as long as the change
    the agent actually drives.

    :param RouteSpace space: Route space the layout was built on.
    :param ReplayLayout layout: The route's replay layout.
    :returns: Blocked spans by lane ID.
    :rtype: StopBlocks
    """
    from .replay import STOP_CLEARANCE

    blocks: Dict[str, List[Tuple[float, float, str]] | None] = {}
    area = space.junction_area
    for lanes, line, _ in layout.pieces:
        for lane_id in lanes:
            if lane_id not in blocks:
                if space.virtual_lane(lane_id) is not None or space.is_junction_lane(lane_id):
                    blocks[lane_id] = None
                    continue
                spans: List[Tuple[float, float, str]] = []
                centerline = space.centerline(lane_id)
                if area is not None and area.intersects(centerline):
                    for part in _parts(centerline.intersection(area)):
                        ends = [centerline.project(Point(xy)) for xy in part.coords]
                        spans.append((min(ends), max(ends), "ordinary stop lies inside a junction polygon"))
                blocks[lane_id] = spans
            if len(lanes) > 1 and blocks[lane_id] is not None:
                centerline = space.centerline(lane_id)
                ends = [centerline.project(Point(line.coords[0])), centerline.project(Point(line.coords[-1]))]
                blocks[lane_id].append((min(ends) - STOP_CLEARANCE, max(ends) + STOP_CLEARANCE,
                                        "ordinary stop lies in or too close to a lane change"))
    return {lane_id: None if spans is None else tuple(spans) for lane_id, spans in blocks.items()}


def stop_problem(blocks: StopBlocks, lane_id: str, s: float) -> str | None:
    """
    Why an ordinary stop may not lie at ``s`` on ``lane_id`` (``stop_blocks``), or None.

    :param StopBlocks blocks: The layout's blocked spans.
    :param str lane_id: The stop's lane.
    :param float s: The stop's position on it, in metres.
    :returns: The reason, or None where a stop may lie.
    :rtype: str | None
    """
    spans = blocks.get(lane_id, ())
    if spans is None:
        return "ordinary stop lies on a junction lane or bicycle connector"
    return next((reason for low, high, reason in spans if low <= s <= high), None)


def check_stops(space: RouteSpace, layout: ReplayLayout, profile: Sequence[SpeedPoint]) -> None:
    """
    Reject ordinary stops where ``stop_blocks`` forbids them.

    :param RouteSpace space: Route space supplying lane geometry, topology, and route-specific constraints.
    :param ReplayLayout layout: Replay layout containing the path and route-to-path mappings.
    :param Sequence['SpeedPoint'] profile: Ordered target-speed points anchored to route lanes.
    :raises InvalidRoute: If an ordinary stop lies on a virtual or junction lane,
        inside a junction polygon, or in or too close to a lane-change stretch.
    """
    if not any(point.speed == 0 for point in profile):
        return
    blocks = stop_blocks(space, layout)
    for point in profile:
        if point.speed == 0:
            problem = stop_problem(blocks, point.lane_id, point.s)
            if problem:
                raise InvalidRoute(problem)


def _parts(geometry) -> List:
    """
    The line parts of a shapely geometry.

    :param geometry: A (multi)line, collection or point geometry.
    :returns: Its parts with length.
    :rtype: List
    """
    parts = list(getattr(geometry, "geoms", [geometry]))
    return [part for part in parts if part.geom_type in ("LineString", "LinearRing") and part.length > 0]


def _boxes(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, headings: np.ndarray, length: float, width: float) -> np.ndarray:
    """
    Oriented rectangular footprints, centred on the replay positions.

    :param Sequence[float] | np.ndarray x: Map X coordinates of the sampled body centres, in metres.
    :param Sequence[float] | np.ndarray y: Map Y coordinates of the sampled body centres, in metres.
    :param np.ndarray headings: Body headings in radians at the sampled positions.
    :param float length: Body length, in metres.
    :param float width: Body width, in metres.
    :returns: An array of oriented rectangular body polygons.
    :rtype: np.ndarray
    """
    corners = np.asarray([[-length / 2, -width / 2], [length / 2, -width / 2],
                          [length / 2, width / 2], [-length / 2, width / 2]])
    c, s = np.cos(headings)[:, None], np.sin(headings)[:, None]
    coords = np.stack((np.asarray(x)[:, None] + c * corners[:, 0] - s * corners[:, 1],
                       np.asarray(y)[:, None] + s * corners[:, 0] + c * corners[:, 1]), axis=-1)
    return shapely.polygons(coords)


def check_footprint(space: RouteSpace, path: LineString, length: float, width: float, *,
                    pivots: Sequence[float] = ()) -> None:
    """
    Check sampled body footprints against real prohibiting markings and endpoint junctions.

    Samples are at most 0.5 m apart, including every path vertex and pivot;
    turning in place is sampled every five degrees. This is a map-based
    rejection check, not a continuous collision or road-surface certificate.

    :param RouteSpace space: Route space supplying lane geometry, topology, and route-specific constraints.
    :param LineString path: Replay-path geometry in map metres.
    :param float length: Body length, in metres.
    :param float width: Body width, in metres.
    :param Sequence[float] pivots: Path distances where the agent stops and turns in place, in metres.
    :raises InvalidRoute: If a sampled footprint crosses a prohibiting marking or
        an endpoint footprint overlaps a junction under endpoint restrictions.
    """
    cache = space.service.derived("footprint_checks", lambda view: {})
    key = (path.wkb, length, width, tuple(pivots), space.restricts_endpoints)
    if key not in cache:
        error = None
        try:
            _check_footprint(space, path, length, width, pivots)
        except InvalidRoute as problem:
            error = str(problem)
        if len(cache) >= _FOOTPRINT_CACHE_SIZE:
            del cache[next(iter(cache))]
        cache[key] = error
    if cache[key] is not None:
        raise InvalidRoute(cache[key])


def _nearby_prohibiting_markings(service: MapService, path: LineString, reach: float) -> List[LaneMarking]:
    """
    Prohibiting markings within ``reach`` of ``path``, in index order.

    Queries the marking index directly: ``MapService.get_lane_markings`` also
    sorts by exact distance to the whole path, which dominated GA run time.

    :param MapService service: Loaded map service supplying geometry and the lane-marking spatial index.
    :param LineString path: Replay-path geometry in map metres.
    :param float reach: Maximum distance from the path at which markings are considered, in metres.
    :returns: Prohibiting lane markings within ``reach`` of the path, in index order.
    :rtype: List[LaneMarking]
    """
    index = service._index("lane marking")
    found = sorted(int(i) for i in index.tree.query(path, predicate="dwithin", distance=reach))
    return [index.items[i] for i in found if index.items[i].prohibits_crossing]


def _check_footprint(space: RouteSpace, path: LineString, length: float, width: float, pivots: Sequence[float]) -> None:
    """
    Reject sampled body footprints that conflict with map markings or endpoint junctions.

    :param RouteSpace space: Route space supplying lane geometry, topology, and route-specific constraints.
    :param LineString path: Replay-path geometry in map metres.
    :param float length: Body length, in metres.
    :param float width: Body width, in metres.
    :param Sequence[float] pivots: Path distances where the agent stops and turns in place, in metres.
    :raises InvalidRoute: If a sampled footprint crosses a prohibiting marking or
        an endpoint footprint overlaps a junction under endpoint restrictions.
    """
    reach = hypot(length, width) / 2
    markings = _nearby_prohibiting_markings(space.service, path, reach)
    area = space.junction_area
    if not markings and (area is None or not space.restricts_endpoints):
        return
    from .replay import _pieces

    xs, ys, headings, turning = [], [], [], []
    pieces = _pieces(path, pivots)
    previous = None
    for _, line in pieces:
        profile = CenterlineProfile(line)
        positions = np.asarray(sorted(set(np.linspace(0, line.length, max(2, ceil(line.length / 0.5) + 1)).tolist())
                                      | set(profile.cumulative_s)))
        vertices = np.asarray(line.coords)[:, :2]
        cumulative = np.asarray(profile.cumulative_s)
        xs.extend(np.interp(positions, cumulative, vertices[:, 0]))
        ys.extend(np.interp(positions, cumulative, vertices[:, 1]))
        headings.extend(_headings_at(profile, positions))
        swing = _headings_at(profile, positions + length / 2) - _headings_at(profile, positions - length / 2)
        turning.extend(np.abs((swing + pi) % (2 * pi) - pi) > TURN_SWEEP_CURVATURE * length)
        incoming = profile.heading_at(0)
        if previous is not None:
            angle = normalize_angle(incoming - previous)
            point = line.interpolate(0)
            for heading in np.linspace(previous, previous + angle, max(2, ceil(abs(angle) / (pi / 36)) + 1)):
                xs.append(point.x)
                ys.append(point.y)
                headings.append(heading)
                turning.append(False)
        previous = profile.heading_at(line.length)
    headings = np.asarray(headings)
    boxes = _boxes(xs, ys, headings, length, max(width - 2 * FOOTPRINT_LATERAL_TOLERANCE, 0.0))
    if markings:
        mark_index, box_index = shapely.STRtree(boxes).query([mark.geometry for mark in markings], predicate="intersects")
        for m, b in sorted(zip(mark_index.tolist(), box_index.tolist())):
            mark = markings[m]
            if (not turning[b] or "CURB" in mark.types) and _parallel(mark.geometry, xs[b], ys[b], headings[b]):
                raise InvalidRoute(f"vehicle footprint crosses {mark.types} on lane {mark.lane_id!r}")
    if area is not None and space.restricts_endpoints:
        first, last = pieces[0][1], pieces[-1][1]
        endpoints = _boxes([first.coords[0][0], last.coords[-1][0]],
                           [first.coords[0][1], last.coords[-1][1]],
                           np.asarray([CenterlineProfile(first).heading_at(0),
                                       CenterlineProfile(last).heading_at(last.length)]), length, width)
        if np.any(shapely.intersects(endpoints, area)):
            raise InvalidRoute("endpoint vehicle footprint overlaps a junction")


def _headings_at(profile: CenterlineProfile, positions: np.ndarray) -> np.ndarray:
    """
    Vectorized ``CenterlineProfile.heading_at``.

    :param CenterlineProfile profile: Centerline heading profile of one replay-path piece.
    :param np.ndarray positions: Positions along that path piece, in metres.
    :returns: Interpolated headings wrapped into [-pi, pi), in radians.
    :rtype: np.ndarray
    """
    mids = np.asarray(profile.midpoint_s)
    heads = np.asarray(profile.headings)
    index = np.searchsorted(mids, positions, side="right")
    out = np.empty(len(positions))
    low, high = index == 0, index == len(heads)
    out[low], out[high] = heads[0], heads[-1]
    mid = ~(low | high)
    i = index[mid]
    before = heads[i - 1]
    turn = (heads[i] - before + pi) % (2 * pi) - pi
    out[mid] = before + turn * (positions[mid] - mids[i - 1]) / (mids[i] - mids[i - 1])
    return (out + pi) % (2 * pi) - pi


def _parallel(line: LineString, x: float, y: float, heading: float) -> bool:
    """
    Whether ``line`` runs within ``CROSSING_ANGLE`` of ``heading`` near (x, y), either direction.

    :param LineString line: Line geometry in map metres.
    :param float x: Map X coordinate, in metres.
    :param float y: Map Y coordinate, in metres.
    :param float heading: Heading in radians.
    :returns: ``True`` if the nearby marking direction is within the crossing-angle threshold of the body axis.
    :rtype: bool
    """
    s = line.project(shapely.Point(x, y))
    a, b = line.interpolate(max(s - 0.5, 0.0)), line.interpolate(min(s + 0.5, line.length))
    if a.equals(b):
        return True
    angle = np.arctan2(b.y - a.y, b.x - a.x) - heading
    return abs((angle + pi / 2) % pi - pi / 2) <= CROSSING_ANGLE
