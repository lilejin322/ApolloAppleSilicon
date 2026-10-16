"""
Re-implementation the map service using shapely STRtree for retrival

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp
"""

from functools import cached_property
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

from google.protobuf.message import Message

import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, Point

from .geometry import CenterlineProfile, curve_to_line, normalize_angle
from .relations import MapRelations
from .types import LaneSignalOverlap, LaneStopLine, PositionEstimate
from ..message.descriptor import DescriptorBundle


class _Snapshot(NamedTuple):
    """One map file's shared data: its tables, and the indexed view once built."""

    tables: "MapService"
    indexed: Optional["MapService"] = None


_snapshots: Dict[Tuple[type, Path, Path], _Snapshot] = {}
"""File snapshots by service class, resolved map path, and descriptor path."""
_snapshots_lock = Lock()
"""Serializes loads and evictions. Cache hits read ``_snapshots`` without it:
an entry is only ever replaced by a complete, more indexed one."""


def _resolve(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _read_map(path: Path, descriptor_path: Path) -> Message:
    """Decode a binary map file with the given schema, uncached."""
    return DescriptorBundle(descriptor_path).message("apollo.hdmap.Map").FromString(path.read_bytes())


def _validate_point(point: Point) -> None:
    if not isinstance(point, Point):
        raise TypeError(f"point must be a shapely Point, not {type(point).__name__}")
    if point.is_empty or not (isfinite(point.x) and isfinite(point.y)):
        raise ValueError("point must have finite x and y coordinates")


def is_allowed_to_cross(boundary: Message):
    for boundary_type in boundary.boundary_type:
        # DOTTED_YELLOW = 1
        # DOTTED_WHITE = 2
        if boundary_type.types[0] not in [1, 2]:
            return False
    return True


class MapService:
    """Own map objects, spatial indexes, and topology for one loaded map.

    File loads share a cached snapshot keyed by map and descriptor paths.
    Each service can switch maps independently. Treat shared protobuf objects,
    tables, graphs, and derived relations as read-only. Pass ``load_only=True``
    to access tables without spatial indexes or routing graphs.
    """
    kSearchRadius = 3.0
    kMaxHeadingDiff = 1.0

    FLAGS_min_length_for_lane_change = 1.0

    lane_index: Optional[STRtree]

    def __init__(
        self, descriptor_path: Union[str, Path], *,
        map_path: Union[str, Path, None] = None, load_only: bool = False,
    ) -> None:
        self.descriptor_path = DescriptorBundle(descriptor_path).descriptor_path
        self._reset()
        if map_path is not None:
            self.load_map_from_file(map_path, load_only=load_only)

    def _reset(self) -> None:
        self.__dict__.pop("relations", None)
        self.map_path: Optional[Path] = None
        self.lane_index = None
        self.lane_table: Dict[str, Message] = dict()
        self.junction_table: Dict[str, Message] = dict()
        self.crosswalk_table: Dict[str, Message] = dict()
        self.signal_table: Dict[str, Message] = dict()
        self.stop_sign_table: Dict[str, Message] = dict()
        self.overlap_table: Dict[str, Message] = dict()
        # Retained for junction membership (Road.junction_id); no road queries.
        self.road_table: Dict[str, Message] = dict()

        # Lane IDs in lane index order: STRtree results are positions in this list.
        self._indexed_lanes: List[str] = list()

        # None until build_routing_graph(), as lane_index is for spatial queries.
        self.routing_graph: Optional[nx.DiGraph] = None
        self._centerlines: Dict[str, LineString] = dict()
        self._centerline_profiles: Dict[str, CenterlineProfile] = dict()

    def load_map_from_file(self, filename: Union[str, Path], load_only: bool = False):
        """Reuse a file snapshot; a failed load leaves this service unchanged."""
        self._adopt_map(self._snapshot(_resolve(filename), load_only))

    def _snapshot(self, path: Path, load_only: bool) -> "MapService":
        """Return the shared snapshot of a map file, reading it on first use.

        Path aliases share one snapshot, and the file is read at most once until
        ``clear_cache``. A load is published only when every step succeeded, so
        a failed first load caches nothing and the next caller reads again.
        """
        key = (type(self), path, self.descriptor_path)
        cached = _snapshots.get(key)
        if cached is None or (not load_only and cached.indexed is None):
            with _snapshots_lock:
                cached = _snapshots.get(key)
                if cached is None:
                    tables = type(self)(self.descriptor_path)
                    tables._populate(_read_map(path, self.descriptor_path), load_only=True)
                    tables.map_path = path
                    # Bind relations to the snapshot, never to a caller that can switch maps.
                    _ = tables.relations
                    cached = _Snapshot(tables)
                if not load_only and cached.indexed is None:
                    indexed = type(self)(self.descriptor_path)
                    indexed._adopt_map(cached.tables)
                    indexed.build_lane_index()
                    indexed.build_routing_graph()
                    cached = cached._replace(indexed=indexed)
                _snapshots[key] = cached
        return cached.tables if load_only else cached.indexed

    def _adopt_map(self, snapshot: "MapService") -> None:
        # Drop relations bound to the previous map if the snapshot has none yet.
        self.__dict__.pop("relations", None)
        self.__dict__.update(snapshot.__dict__)

    @classmethod
    def clear_cache(
        cls, map_path: Union[str, Path, None] = None, *,
        descriptor_path: Union[str, Path, None] = None,
    ) -> None:
        """Evict matching file snapshots without changing existing services.

        With no filters, evict all snapshots for this class. An eviction waits
        for loads in progress, so nothing read before it stays cached. Reload a
        service afterwards to read changed files; descriptor bundles retain
        their own independent immutable-file cache.
        """
        path = None if map_path is None else _resolve(map_path)
        schema = None if descriptor_path is None else _resolve(descriptor_path)
        with _snapshots_lock:
            for key in [key for key in _snapshots if key[0] is cls
                        and (path is None or key[1] == path) and (schema is None or key[2] == schema)]:
                del _snapshots[key]

    def load_map_from_proto(self, map: Message, load_only: bool = False):
        """Replace this service's map; a failed load leaves it unchanged."""
        expected = DescriptorBundle(self.descriptor_path).message("apollo.hdmap.Map")
        if not isinstance(map, expected):
            raise TypeError("map must use the service's descriptor_path schema")
        staged = type(self)(self.descriptor_path)
        staged._populate(map, load_only)
        self._adopt_map(staged)

    def _populate(self, map: Message, load_only: bool) -> None:
        obj_table_map = [
            (map.lane, self.lane_table),
            (map.junction, self.junction_table),
            (map.crosswalk, self.crosswalk_table),
            (map.signal, self.signal_table),
            (map.stop_sign, self.stop_sign_table),
            (map.overlap, self.overlap_table),
            (map.road, self.road_table),
        ]
        for objects, table_name in obj_table_map:
            for obj in objects:
                table_name[obj.id.id] = obj

        if not load_only:
            self.build_lane_index()
            self.build_routing_graph()

    def get_lanes(self, point: Optional[Point] = None, distance=0.0) -> Iterable[str]:
        if not isfinite(distance) or distance < 0:
            raise ValueError("distance must be finite and nonnegative")
        if distance == 0.0:
            return self.lane_table.keys()
        _validate_point(point)
        return self._lanes_within(point, distance)

    def get_junctions(self, point: Optional[Point] = None, distance=0.0) -> Iterable[str]:
        if distance == 0.0:
            return self.junction_table.keys()
        raise NotImplementedError()

    def get_crosswalks(self, point: Optional[Point] = None, distance=0.0) -> Iterable[str]:
        if distance == 0.0:
            return self.crosswalk_table.keys()
        raise NotImplementedError()

    def get_signals(self, point: Optional[Point] = None, distance=0.0) -> Iterable[str]:
        if distance == 0.0:
            return self.signal_table.keys()
        raise NotImplementedError()

    def get_stop_signs(self, point: Optional[Point] = None, distance=0.0) -> Iterable[str]:
        if distance == 0.0:
            return self.stop_sign_table.keys()
        raise NotImplementedError()

    def get_lane_by_id(self, lane_id: str) -> Message:
        return self.lane_table[lane_id]

    def get_lane_central_curve_by_id(self, lane_id: str) -> LineString:
        # Built once per loaded map. Shapely geometries are immutable, so
        # handles sharing a cached snapshot may also share these lines.
        line = self._centerlines.get(lane_id)
        if line is None:
            line = curve_to_line(self.get_lane_by_id(lane_id).central_curve)
            self._centerlines[lane_id] = line
        return line

    def get_lane_boundary_types_by_id(self, lane_id: str) -> Tuple[int, int]:
        lane_obj = self.get_lane_by_id(lane_id)
        return (
            lane_obj.left_boundary.boundary_type[0].types[0],
            lane_obj.right_boundary.boundary_type[0].types[0],
        )

    def get_lane_boundaries_by_id(self, lane_id: str) -> Tuple[LineString, LineString]:
        lane_obj = self.get_lane_by_id(lane_id)

        return (
            curve_to_line(lane_obj.left_boundary.curve),
            curve_to_line(lane_obj.right_boundary.curve),
        )

    def get_junction_by_id(self, junction_id: str):
        return self.junction_table[junction_id]

    def get_crosswalk_by_id(self, crosswalk_id: str):
        return self.crosswalk_table[crosswalk_id]

    def get_signal_by_id(self, signal_id: str):
        return self.signal_table[signal_id]

    def get_stop_sign_by_id(self, stop_sign_id: str):
        return self.stop_sign_table[stop_sign_id]

    def get_overlap_by_id(self, overlap_id: str):
        return self.overlap_table[overlap_id]

    def get_path_from(
        self, lane_id: str, forward_only=True, limit=15
    ) -> List[List[str]]:
        graph = self._routing_graph()
        if lane_id not in self.lane_table:
            raise KeyError(lane_id)
        result: List[List[str]] = []
        if limit <= 0:
            return result
        for _, target in graph.edges(lane_id):
            edge_data = graph.get_edge_data(lane_id, target)
            if forward_only and edge_data['direction_type'] in ['L', 'R']:
                continue
            result.append([lane_id, target])
            for path in self.get_path_from(target, forward_only, limit - 1):
                result.append([lane_id] + path)
        return result

    @property
    def junction_lanes(self) -> List[str]:
        """Lane IDs in ``relations.lanes_at_junction``, in table order."""
        outside = self.get_lanes_not_in_junction()
        return [lane for lane in self.lane_table if lane not in outside]

    @property
    def non_junction_lanes(self) -> List[str]:
        outside = self.get_lanes_not_in_junction()
        return [lane for lane in self.lane_table if lane in outside]

    def build_routing_graph(self):
        # modules/routing/topo_creator/graph_creator.cc
        self.routing_graph = nx.DiGraph()
        self.routing_graph.add_nodes_from(self.lane_table)

        def known(ids) -> List[str]:
            # Skip unknown IDs instead of adding nodes, as in GraphCreator::AddEdge.
            return [x.id for x in ids if x.id in self.lane_table]

        for lane_id, lane in self.lane_table.items():
            for pid in known(lane.predecessor_id):
                self.routing_graph.add_edge(pid, lane_id, direction_type='F')
            for pid in known(lane.successor_id):
                self.routing_graph.add_edge(lane_id, pid, direction_type='F')

            if lane.length < MapService.FLAGS_min_length_for_lane_change:
                continue

            # Lane changes cross the source lane's own boundary.
            if lane.HasField('left_boundary') and is_allowed_to_cross(lane.left_boundary):
                for pid in known(lane.left_neighbor_forward_lane_id):
                    self.routing_graph.add_edge(lane_id, pid, direction_type='L')
            if lane.HasField('right_boundary') and is_allowed_to_cross(lane.right_boundary):
                for pid in known(lane.right_neighbor_forward_lane_id):
                    self.routing_graph.add_edge(lane_id, pid, direction_type='R')

    def build_lane_index(self):
        # Whole centerlines: queries need lane IDs, and GEOS measures the exact
        # distance to every segment of each candidate. Coincident lanes keep
        # separate entries.
        self._indexed_lanes = list(self.lane_table)
        self.lane_index = STRtree([
            self.get_lane_central_curve_by_id(lane_id) for lane_id in self._indexed_lanes
        ])

    def get_lane_coord_and_heading(
        self, lane_id: str, s: float
    ) -> Tuple[Point, float]:
        """Return the centerline point and heading at geometric arc length s.

        s uses the same centreline distance as get_position(), not Lane.length.
        Headings interpolate between segment midpoints; see CenterlineProfile.
        """
        profile = self._centerline_profile(lane_id)
        length = profile.line.length
        # Negative s would make Shapely interpolate backwards from the end.
        if not 0 <= s <= length:
            raise ValueError(f"s={s} is outside lane {lane_id!r} [0, {length}]")

        ip = profile.line.interpolate(s)
        return Point(ip.x, ip.y), profile.heading_at(s)

    def get_pose_with_regard_to_lane(self, point: Point) -> float:
        """Heading of the nearest lane where the point projects onto it."""
        _validate_point(point)
        profile = self._centerline_profile(self.get_nearest_lane(point))
        return profile.heading_at(profile.line.project(point))

    def _centerline_profile(self, lane_id: str) -> CenterlineProfile:
        # Derived from immutable lane geometry, so handles sharing a cached
        # snapshot may also share these lazily built profiles.
        profile = self._centerline_profiles.get(lane_id)
        if profile is None:
            profile = CenterlineProfile(self.get_lane_central_curve_by_id(lane_id))
            self._centerline_profiles[lane_id] = profile
        return profile

    def get_nearest_lane(self, point: Point) -> str:
        _validate_point(point)
        indices = self._lane_index().query_nearest(point)
        if len(indices) == 0:
            raise ValueError("Map has no lanes")
        # Equally near lanes, such as coincident ones, resolve to the earliest.
        return self._indexed_lanes[min(indices)]

    def get_nearest_lanes_with_heading(
        self, point: Point, heading: float
    ) -> List[str]:
        """Lanes within kSearchRadius whose heading at the point's projection
        differs by less than kMaxHeadingDiff, nearest and best aligned first."""
        _validate_point(point)
        if not isfinite(heading):
            raise ValueError("heading must be finite")
        candidates = []
        for lane_id in self._lanes_within(point, self.kSearchRadius):
            profile = self._centerline_profile(lane_id)
            # Resolve the closest projection before filtering. A farther segment
            # of this lane must not qualify it. Ties use the first point along
            # the full centerline, as in get_position() and the pose query.
            s = profile.line.project(point)
            unsigned_diff = abs(normalize_angle(heading - profile.heading_at(s)))
            if unsigned_diff < self.kMaxHeadingDiff:
                candidates.append((profile.line.distance(point), unsigned_diff, lane_id))
        candidates.sort()
        return [lane_id for _, _, lane_id in candidates]

    def _lanes_within(self, point: Point, distance: float) -> set[str]:
        indices = self._lane_index().query(point, predicate="dwithin", distance=distance)
        return {self._indexed_lanes[index] for index in indices}

    def _lane_index(self) -> STRtree:
        if self.lane_index is None:
            raise RuntimeError("Lane index is unavailable; load with load_only=False")
        return self.lane_index

    def _routing_graph(self) -> nx.DiGraph:
        if self.routing_graph is None:
            raise RuntimeError("Routing graph is unavailable; load with load_only=False")
        return self.routing_graph

    def get_position(self, point: Point) -> PositionEstimate:
        """Project a point onto the nearest lane's complete centerline."""
        _validate_point(point)
        self._lane_index()
        if not self.lane_table:
            return PositionEstimate()
        lane_id = self.get_nearest_lane(point)
        return PositionEstimate(lane_id, self.get_lane_central_curve_by_id(lane_id).project(point))

    @cached_property
    def relations(self) -> MapRelations:
        return MapRelations(self)

    def get_signals_wrt(self, signal_id: str) -> List[Tuple[str, str]]:
        return self.relations.get_signals_wrt(signal_id)

    def get_lanes_controlled_by_signal(self, signal_id: str) -> List[str]:
        """Return lane IDs associated with the signal by explicit overlaps."""
        return list(self.relations.lanes_controlled_by_signal[signal_id])

    def get_signals_for_lane(self, lane_id: str) -> List[str]:
        """Return signal IDs associated with the lane by explicit overlaps."""
        return list(self.relations.signals_for_lane[lane_id])

    def get_signal_lane_overlaps(self, signal_id: str) -> tuple[LaneSignalOverlap, ...]:
        """Return all lane intervals associated with a signal, in lane metres."""
        return self.relations.signal_lane_overlaps[signal_id]

    def get_lane_signal_overlaps(self, lane_id: str) -> tuple[LaneSignalOverlap, ...]:
        """Return all signal intervals on a lane, retaining separate overlaps."""
        return self.relations.lane_signal_overlaps[lane_id]

    def get_signal_stop_lines(
        self, signal_id: str, lane_id: str, *, max_distance: float = 1.0
    ) -> tuple[LaneStopLine, ...]:
        """Match a signal's stop lines to this lane's overlap intervals.

        max_distance is a geometric tolerance in metres, not a search radius
        around the vehicle. Multiple candidates are retained; no nearest-only
        fallback is used. The returned geometry is the full signal stop line.
        """
        return self.relations.get_signal_stop_lines(signal_id, lane_id, max_distance)

    def get_lane_stop_lines(
        self, lane_id: str, *, max_distance: float = 1.0
    ) -> tuple[LaneStopLine, ...]:
        """Return matching stop lines from every signal associated with a lane.

        Stop-sign stop lines are not included. Shared geometry keeps separate
        records for each signal so that signal provenance is preserved.
        """
        return self.relations.get_lane_stop_lines(lane_id, max_distance)

    def is_conflict_lanes(self, lane_ids1: List[str], lane_ids2: List[str]) -> bool:
        return self.relations.is_conflict_lanes(lane_ids1, lane_ids2)

    def get_lanes_not_in_junction(self) -> set[str]:
        junction_lanes = {
            lane for lanes in self.relations.lanes_at_junction.values() for lane in lanes
        }
        return set(self.lane_table) - junction_lanes

    def get_geometric_paths_from(self, lane_id: str, limit: int = 4) -> List[List[str]]:
        """Infer endpoint-connected paths ending outside junctions.

        This heuristic is separate from the explicit-topology routing graph.
        ``limit`` is the maximum number of edges, as in ``get_path_from``.
        """
        return self.relations.get_geometric_paths_from(lane_id, limit)

    def get_predecessors_for_lane(self, lane_id: str) -> List[str]:
        return [x.id for x in self.lane_table[lane_id].predecessor_id]

    def get_successors_for_lane(self, lane_id: str) -> List[str]:
        return [x.id for x in self.lane_table[lane_id].successor_id]

    def get_length_of_lane(self, lane_id: str) -> float:
        return self.lane_table[lane_id].length
