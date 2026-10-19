"""Map constraints shared by replay validation and genetic acceptance.

Missing boundary geometry is not evidence of free space. These checks reject
known conflicts, but cannot certify areas the map does not describe.
"""

from math import ceil, hypot, pi
from typing import TYPE_CHECKING, Sequence
from weakref import WeakKeyDictionary

import numpy as np
import shapely
from shapely.geometry import LineString

from ..map.geometry import CenterlineProfile, normalize_angle
from .routing import InvalidRoute, RouteSpace, _junction_info

if TYPE_CHECKING:
    from .replay import ReplayLayout, SpeedPoint

_FOOTPRINT_CACHE = WeakKeyDictionary()
"""At most 128 validation results per immutable loaded map; no trajectory arrays retained."""


def check_stops(space: RouteSpace, layout: "ReplayLayout", profile: Sequence["SpeedPoint"],
                positions: Sequence[float]) -> None:
    """Reject ordinary stops on junctions, connectors or lane-change stretches."""
    from .replay import STOP_CLEARANCE

    changes = [(layout.moved(offset), layout.moved(offset + line.length))
               for lanes, line, offset in layout.pieces if len(lanes) > 1]
    for point, position in zip(profile, positions):
        if point.speed != 0:
            continue
        if space.virtual_lane(point.lane_id) is not None or space.is_junction_lane(point.lane_id):
            raise InvalidRoute("ordinary stop lies on a junction lane or bicycle connector")
        if any(low - STOP_CLEARANCE <= position <= high + STOP_CLEARANCE for low, high in changes):
            raise InvalidRoute("ordinary stop lies in or too close to a lane change")
        # Partial junction overlap need not classify the whole lane as a junction.
        area = _junction_info(space.service, space._no_turn)["union"]
        if area is not None and area.intersects(layout.path.interpolate(position)):
            raise InvalidRoute("ordinary stop lies inside a junction polygon")


def _boxes(x, y, headings, length: float, width: float):
    """Oriented rectangular footprints, centred on the replay positions."""
    corners = np.asarray([[-length / 2, -width / 2], [length / 2, -width / 2],
                          [length / 2, width / 2], [-length / 2, width / 2]])
    c, s = np.cos(headings)[:, None], np.sin(headings)[:, None]
    coords = np.stack((np.asarray(x)[:, None] + c * corners[:, 0] - s * corners[:, 1],
                       np.asarray(y)[:, None] + s * corners[:, 0] + c * corners[:, 1]), axis=-1)
    return shapely.polygons(coords)


def check_footprint(space: RouteSpace, path: LineString, length: float, width: float, *,
                    pivots: Sequence[float] = ()) -> None:
    """Check sampled body footprints against real prohibiting markings and endpoint junctions.

    Samples are at most 0.5 m apart, including every path vertex and pivot;
    turning in place is sampled every five degrees. This is a map-based
    rejection check, not a continuous collision or road-surface certificate.
    """
    cache = _FOOTPRINT_CACHE.setdefault(space.service, {})
    key = (path.wkb, length, width, tuple(pivots), space.restricts_endpoints)
    if key not in cache:
        error = None
        try:
            _check_footprint(space, path, length, width, pivots)
        except InvalidRoute as problem:
            error = str(problem)
        if len(cache) >= 128:
            del cache[next(iter(cache))]
        cache[key] = error
    if cache[key] is not None:
        raise InvalidRoute(cache[key])


def _check_footprint(space, path, length, width, pivots):
    reach = hypot(length, width) / 2
    markings = [mark for mark in space.service.get_lane_markings(path, reach)
                if mark.prohibits_crossing]
    area = _junction_info(space.service, space._no_turn)["union"]
    if not markings and (area is None or not space.restricts_endpoints):
        return
    from .replay import _pieces

    xs, ys, headings = [], [], []
    pieces = _pieces(path, pivots)
    previous = None
    for _, line in pieces:
        profile = CenterlineProfile(line)
        positions = sorted(set(np.linspace(0, line.length, max(2, ceil(line.length / 0.5) + 1)))
                           | set(profile.cumulative_s))
        for position in positions:
            point = line.interpolate(position)
            xs.append(point.x)
            ys.append(point.y)
            headings.append(profile.heading_at(position))
        incoming = profile.heading_at(0)
        if previous is not None:
            angle = normalize_angle(incoming - previous)
            point = line.interpolate(0)
            for heading in np.linspace(previous, previous + angle, max(2, ceil(abs(angle) / (pi / 36)) + 1)):
                xs.append(point.x)
                ys.append(point.y)
                headings.append(heading)
        previous = profile.heading_at(line.length)
    boxes = _boxes(xs, ys, np.asarray(headings), length, width)
    for mark in markings:
        if np.any(shapely.intersects(boxes, mark.geometry)):
            raise InvalidRoute(f"vehicle footprint crosses {mark.types} on lane {mark.lane_id!r}")
    if area is not None and space.restricts_endpoints:
        first, last = pieces[0][1], pieces[-1][1]
        endpoints = _boxes([first.coords[0][0], last.coords[-1][0]],
                           [first.coords[0][1], last.coords[-1][1]],
                           np.asarray([CenterlineProfile(first).heading_at(0),
                                       CenterlineProfile(last).heading_at(last.length)]), length, width)
        if np.any(shapely.intersects(endpoints, area)):
            raise InvalidRoute("endpoint vehicle footprint overlaps a junction")
