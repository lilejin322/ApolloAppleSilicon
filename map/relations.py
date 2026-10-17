"""Derived map relationships; all source objects belong to MapService."""

from collections import defaultdict
from functools import cached_property
from math import isfinite
from typing import TYPE_CHECKING

import networkx as nx
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import substring

from .geometry import curve_to_line
from .types import LaneSignalOverlap, LaneStopLine

if TYPE_CHECKING:
    from .map_service import MapService


def _overlap_ids(obj) -> set[str]:
    return {overlap.id for overlap in obj.overlap_id}


def _validate_max_distance(max_distance: float) -> None:
    if not isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")


class MapRelations:
    """Lazy overlap, signal, and geometric connectivity analysis.

    Signal EQ/NE labels retain the old parser's geometric heuristic. They
    describe shared controlled lanes or intersecting centerlines, not verified
    traffic-light phase constraints.
    """

    def __init__(self, service: "MapService") -> None:
        self._service = service

    def _at_junction(self, objects) -> dict[str, list[str]]:
        by_overlap = defaultdict(list)
        for object_id, obj in objects.items():
            for overlap_id in _overlap_ids(obj):
                by_overlap[overlap_id].append(object_id)
        result = {}
        for junction_id, junction in self._service.junction_table.items():
            matches = {
                object_id for overlap_id in _overlap_ids(junction)
                for object_id in by_overlap[overlap_id]
            }
            result[junction_id] = sorted(matches)
        return result

    @cached_property
    def signals_at_junction(self) -> dict[str, list[str]]:
        """Signals sharing an overlap with a junction or controlling its lanes.

        Some maps (e.g. the Sunnyvale loops) have no signal-junction overlaps,
        so a signal also joins each junction containing a lane it controls.
        """
        result = {junction_id: set(signals) for junction_id, signals
                  in self._at_junction(self._service.signal_table).items()}
        # Overlapping lanes, not road membership: a few controlled lanes in
        # sunnyvale_loop overlap a junction without belonging to its road.
        for junction_id, lanes in self.lanes_overlapping_junction.items():
            for lane_id in lanes:
                result[junction_id].update(self.signals_for_lane[lane_id])
        return {junction_id: sorted(signals) for junction_id, signals in result.items()}

    @cached_property
    def lanes_overlapping_junction(self) -> dict[str, list[str]]:
        """Lanes sharing an overlap with each junction, even only at one end."""
        return self._at_junction(self._service.lane_table)

    @cached_property
    def lanes_at_junction(self) -> dict[str, list[str]]:
        """Membership priority: valid Lane.junction_id, road, then overlaps.

        Junction polygons often cover a few metres of ordinary lanes entering
        or leaving them, so an overlap alone overstates membership. Therefore
        a lane's explicit junction wins; otherwise its road decides. Lanes in
        no road, or in a road naming an unknown junction, fall back to overlaps.
        Empty or unknown lane junction IDs do not override the fallbacks.
        """
        junctions = self._service.junction_table
        decided: dict[str, set[str]] = {}
        for road in self._service.road_table.values():
            junction_id = road.junction_id.id if road.HasField("junction_id") else ""
            if junction_id and junction_id not in junctions:
                continue
            for section in road.section:
                for lane in section.lane_id:
                    lane_junctions = decided.setdefault(lane.id, set())
                    if junction_id:
                        lane_junctions.add(junction_id)
        for lane_id, lane in self._service.lane_table.items():
            junction_id = lane.junction_id.id
            if junction_id and junction_id in junctions:
                decided[lane_id] = {junction_id}
        result = {junction_id: set() for junction_id in junctions}
        for junction_id, lanes in self.lanes_overlapping_junction.items():
            result[junction_id].update(lane for lane in lanes if lane not in decided)
        for lane_id, lane_junctions in decided.items():
            if lane_id in self._service.lane_table:
                for junction_id in lane_junctions:
                    result[junction_id].add(lane_id)
        return {junction_id: sorted(lanes) for junction_id, lanes in result.items()}

    @cached_property
    def lane_signal_overlaps(self) -> dict[str, tuple[LaneSignalOverlap, ...]]:
        """Resolve each lane's overlaps directly, without a junction prerequisite.

        As in Apollo's LaneInfo, follow lane.overlap_id into the overlap table.
        Require typed lane/signal participants and existing signal objects;
        dangling references or unrelated overlap objects cannot create a link.
        """
        result = {}
        for lane_id, lane in self._service.lane_table.items():
            records = set()
            for overlap_id in sorted(_overlap_ids(lane)):
                overlap = self._service.overlap_table.get(overlap_id)
                if overlap is None:
                    continue
                lane_objects = [obj for obj in overlap.object
                                if obj.id.id == lane_id and obj.HasField("lane_overlap_info")]
                signal_ids = {obj.id.id for obj in overlap.object
                              if obj.HasField("signal_overlap_info")
                              and obj.id.id in self._service.signal_table}
                for obj in lane_objects:
                    info = obj.lane_overlap_info
                    for signal_id in signal_ids:
                        records.add(LaneSignalOverlap(
                            overlap_id, lane_id, signal_id,
                            info.start_s if info.HasField("start_s") else None,
                            info.end_s if info.HasField("end_s") else None,
                        ))
            result[lane_id] = tuple(sorted(records, key=lambda record: (
                record.overlap_id, record.signal_id,
                (record.start_s is not None, record.start_s),
                (record.end_s is not None, record.end_s),
            )))
        return result

    @cached_property
    def signal_lane_overlaps(self) -> dict[str, tuple[LaneSignalOverlap, ...]]:
        result = {signal_id: [] for signal_id in self._service.signal_table}
        for lane_id in sorted(self.lane_signal_overlaps):
            for record in self.lane_signal_overlaps[lane_id]:
                result[record.signal_id].append(record)
        return {key: tuple(value) for key, value in result.items()}

    @cached_property
    def lanes_controlled_by_signal(self) -> dict[str, list[str]]:
        return {signal_id: sorted({record.lane_id for record in records})
                for signal_id, records in self.signal_lane_overlaps.items()}

    @cached_property
    def signals_for_lane(self) -> dict[str, list[str]]:
        return {lane_id: sorted({record.signal_id for record in records})
                for lane_id, records in self.lane_signal_overlaps.items()}

    def get_signal_stop_lines(
        self, signal_id: str, lane_id: str, max_distance: float
    ) -> tuple[LaneStopLine, ...]:
        """Match a signal's stop-line curves to the lane's overlap intervals.

        Apollo has no lane ID on Signal.stop_line. The explicit lane-signal
        association is authoritative; matching individual curves is geometric,
        so every curve within max_distance is returned, never a guessed one.
        """
        _validate_max_distance(max_distance)
        signal = self._service.get_signal_by_id(signal_id)
        self._service.get_lane_by_id(lane_id)  # Unknown IDs are errors, even if unassociated.
        overlaps = [record for record in self.lane_signal_overlaps[lane_id]
                    if record.signal_id == signal_id]
        if not overlaps or not signal.stop_line:
            return ()

        centerline = self._service.get_lane_central_curve_by_id(lane_id)
        intervals = []
        for record in overlaps:
            start, end = record.start_s, record.end_s
            if (start is None or end is None or not isfinite(start) or not isfinite(end)
                    or start < 0 or end < start or start > centerline.length):
                continue
            # Clip an interval's end to the available centerline geometry.
            intervals.append((record, substring(centerline, start, min(end, centerline.length))))
        if not intervals:
            return ()

        result = []
        for index, curve in enumerate(signal.stop_line):
            try:
                geometry = curve_to_line(curve)
            except ValueError as error:
                raise ValueError(f"Invalid stop_line[{index}] for signal {signal_id!r}") from error
            if not geometry.is_valid:
                raise ValueError(f"Invalid stop_line[{index}] for signal {signal_id!r}")
            matches = [(record, geometry.distance(section)) for record, section in intervals]
            matches = [(record, distance) for record, distance in matches if distance <= max_distance]
            if matches:
                result.append(LaneStopLine(
                    lane_id=lane_id,
                    signal_id=signal_id,
                    stop_line_index=index,
                    geometry=geometry,
                    overlaps=tuple(record for record, _ in matches),
                    distance=min(distance for _, distance in matches),
                ))
        return tuple(result)

    def get_lane_stop_lines(self, lane_id: str, max_distance: float) -> tuple[LaneStopLine, ...]:
        # Validate even when the lane has no signals and nothing is matched.
        _validate_max_distance(max_distance)
        return tuple(
            match
            for signal_id in self.signals_for_lane[lane_id]
            for match in self.get_signal_stop_lines(signal_id, lane_id, max_distance)
        )

    def is_conflict_lanes(self, lane_ids1: list[str], lane_ids2: list[str]) -> bool:
        return any(
            left != right and self._service.get_lane_central_curve_by_id(left).intersects(
                self._service.get_lane_central_curve_by_id(right)
            )
            for left in lane_ids1 for right in lane_ids2
        )

    @cached_property
    def signal_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self._service.signal_table)
        for signals in self.signals_at_junction.values():
            for index, left in enumerate(signals):
                for right in signals[index + 1:]:
                    left_lanes = self.lanes_controlled_by_signal[left]
                    right_lanes = self.lanes_controlled_by_signal[right]
                    if left_lanes and left_lanes == right_lanes:
                        graph.add_edge(left, right, relation="EQ")
                    elif self.is_conflict_lanes(left_lanes, right_lanes):
                        graph.add_edge(left, right, relation="NE")
        return graph

    def get_signals_wrt(self, signal_id: str) -> list[tuple[str, str]]:
        if signal_id not in self._service.signal_table:
            raise KeyError(signal_id)
        return [(other, data["relation"])
                for other, data in self.signal_graph[signal_id].items()]

    @cached_property
    def geometric_lane_graph(self) -> nx.DiGraph:
        """Infer directed edges where lane endpoints are less than 1 mm apart.

        Successors are added in lane table order.
        """
        lane_ids = list(self._service.lane_table)
        lines = [self._service.get_lane_central_curve_by_id(lane_id) for lane_id in lane_ids]
        starts = [Point(line.coords[0]) for line in lines]
        index = STRtree(starts)
        graph = nx.DiGraph()
        graph.add_nodes_from(lane_ids)
        for lane_id, line in zip(lane_ids, lines):
            end = Point(line.coords[-1])
            # dwithin includes the 1 mm boundary; the check below excludes it.
            for position in sorted(index.query(end, predicate="dwithin", distance=0.001)):
                if lane_ids[position] != lane_id and end.distance(starts[position]) < 0.001:
                    graph.add_edge(lane_id, lane_ids[position])
        return graph

    def get_geometric_paths_from(self, lane_id: str, limit: int) -> list[list[str]]:
        if lane_id not in self._service.lane_table:
            raise KeyError(lane_id)
        result = []
        targets = self._service.get_lanes_not_in_junction()

        def visit(path, remaining):
            if remaining <= 0:
                return
            for successor in self.geometric_lane_graph.successors(path[-1]):
                extended = path + [successor]
                if successor in targets:
                    result.append(extended)
                visit(extended, remaining - 1)

        visit([lane_id], limit)
        return result
