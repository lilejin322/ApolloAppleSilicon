"""
Derived map relationships; all source objects belong to MapService.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp
"""

from collections import defaultdict
from functools import cached_property
from math import atan2, cos, degrees, hypot, isfinite, sin
from typing import TYPE_CHECKING, Iterator, Mapping
import networkx as nx
from google.protobuf.message import Message
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .geometry import curve_to_line, normalize_angle
from .types import LaneSignalOverlap, LaneStopLine

if TYPE_CHECKING:
    from .map_service import MapService


SAME_APPROACH = 45.0
"""Signals whose controlled lanes enter within this many degrees share an approach."""
OPPOSING_APPROACH = 135.0
"""Signals whose controlled lanes enter more than this many degrees apart are opposing."""
LEFT_TURN = 45.0
"""A movement turning left by more than this many degrees, or any U-turn, yields to oncoming traffic."""


def _overlap_ids(obj) -> set[str]:
    return {overlap.id for overlap in obj.overlap_id}


def forward_links(lane_id: str, lane: Message, lanes: Mapping[str, Message]) -> Iterator[tuple[str, str]]:
    """
    Yield the forward (from, to) lane pairs that one lane declares.

    A lane links each known predecessor to itself and itself to each known
    successor; unknown IDs are skipped, as in Apollo's ``GraphCreator::AddEdge``.

    :param str lane_id: The lane's ID.
    :param Message lane: The ``apollo.hdmap.Lane`` message.
    :param Mapping[str, Message] lanes: All lanes by ID.
    :rtype: Iterator[tuple[str, str]]
    """
    for other in lane.predecessor_id:
        if other.id in lanes:
            yield other.id, lane_id
    for other in lane.successor_id:
        if other.id in lanes:
            yield lane_id, other.id


def _validate_max_distance(max_distance: float) -> None:
    if not isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")


class MapRelations:
    """Lazy overlap and signal analysis.

    Signal EQ/NE labels model a junction with circular lights from the lanes
    each signal controls; Apollo maps carry no signal phases, so they are
    inferred, not read.
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
    def forward_successors(self) -> dict[str, list[str]]:
        """Each lane's forward successors, as in ``routing_graph``, without needing it."""
        lanes = self._service.lane_table
        successors: dict[str, dict[str, None]] = defaultdict(dict)
        for lane_id, lane in lanes.items():
            for source, target in forward_links(lane_id, lane, lanes):
                successors[source][target] = None
        return {lane_id: list(targets) for lane_id, targets in successors.items()}

    def movement_paths(self, signal_id: str, junction_id: str) -> list[tuple[str, ...]]:
        """
        Return the lane paths a signal's traffic takes through a junction.

        Signals may control only a short entry piece of each lane, so each
        controlled lane is followed forward through every successor that still
        overlaps the junction, until the path leaves it.

        :param str signal_id: Signal ID.
        :param str junction_id: A junction the signal belongs to.
        :returns: Paths of lane IDs, each starting with a controlled lane.
        :rtype: list[tuple[str, ...]]
        """
        inside = set(self.lanes_overlapping_junction[junction_id])
        paths = []

        def extend(path: tuple[str, ...]) -> None:
            successors = [lane for lane in self.forward_successors.get(path[-1], ())
                          if lane in inside and lane not in path]
            if not successors:
                paths.append(path)
            for successor in successors:
                extend(path + (successor,))

        for lane_id in self.lanes_controlled_by_signal[signal_id]:
            extend((lane_id,))
        return paths

    @cached_property
    def signal_graph(self) -> nx.Graph:
        """
        Signals at a shared junction joined by ``relation`` ``"EQ"`` (same colour)
        or ``"NE"`` (never green together), as at a junction with circular lights:

        - controlled lanes that are the same, or enter from the same approach: EQ;
        - crossing approaches: NE;
        - opposing approaches: EQ, as left turns yield to oncoming traffic, unless
          movements cross where neither turns left (a skewed junction): NE.

        NE evidence takes precedence across junctions. EQ candidates merge
        groups only if no NE pair would end up inside a group; same-lane and
        same-approach candidates take priority over opposing approaches.
        Rejected EQ candidates become NE so configuration generation can
        separate their phases. Edges retain all original junction evidence.
        """
        graph = nx.Graph()
        graph.add_nodes_from(sorted(self._service.signal_table))
        evidence = defaultdict(list)
        for junction_id, signals in sorted(self.signals_at_junction.items()):
            paths = {signal: self.movement_paths(signal, junction_id) for signal in signals}
            for index, left in enumerate(signals):
                for right in signals[index + 1:]:
                    relation = self._relation(left, right, paths)
                    if relation is not None:
                        evidence[left, right].append((junction_id, left, right, relation))
        for (left, right), records in sorted(evidence.items()):
            graph.add_edge(left, right,
                           relation="NE" if any(record[3] == "NE" for record in records) else "EQ",
                           junctions=tuple(record[0] for record in records),
                           evidence=tuple(records))
        self._resolve_signal_groups(graph)
        return graph

    def _resolve_signal_groups(self, graph: nx.Graph) -> None:
        """Resolve inferred EQ preferences into deterministic groups respecting every NE edge."""
        def priority(pair: tuple[str, str]) -> tuple[int, float, str, str]:
            left, right = pair
            if self.lanes_controlled_by_signal[left] == self.lanes_controlled_by_signal[right]:
                return 0, 0.0, left, right
            between = abs(degrees(normalize_angle(self._approach(left) - self._approach(right))))
            # Quantize ranking noise so global rotation preserves tie-breaking.
            return (1 if between < SAME_APPROACH else 2,
                    round(min(between, 180 - between), 9), left, right)

        groups = {signal: {signal} for signal in graph}
        candidates = sorted(((a, b) for a, b, r in graph.edges(data="relation") if r == "EQ"),
                            key=priority)
        for left, right in candidates:
            first, second = groups[left], groups[right]
            if first is second:
                continue
            if any(graph.has_edge(a, b) and graph[a][b]["relation"] == "NE"
                   for a in first for b in second):
                continue
            merged = first | second
            for signal in merged:
                groups[signal] = merged
        for left, right, data in graph.edges(data=True):
            if data["relation"] == "EQ" and groups[left] is not groups[right]:
                data["relation"] = "NE"

    def _relation(self, left: str, right: str, paths: Mapping[str, list[tuple[str, ...]]]) -> str | None:
        left_lanes = self.lanes_controlled_by_signal[left]
        right_lanes = self.lanes_controlled_by_signal[right]
        if left_lanes and left_lanes == right_lanes:
            return "EQ"
        if not left_lanes or not right_lanes:
            return None
        left_direction, right_direction = self._approach(left), self._approach(right)
        if left_direction is None or right_direction is None:
            return None
        between = abs(degrees(normalize_angle(left_direction - right_direction)))
        if between < SAME_APPROACH:
            return "EQ"
        if between <= OPPOSING_APPROACH:
            return "NE"
        unyielding = any(not self._turns_left(left, p) and not self._turns_left(right, q)
                         and self._movements_conflict(left, p, right, q)
                         for p in paths[left] for q in paths[right])
        return "NE" if unyielding else "EQ"

    def _direction(self, lane_id: str, *, at_end: bool = False, s: float = 0.0) -> float:
        """Segment direction at geometric s, using the outgoing segment at a vertex."""
        coords = list(self._service.get_lane_central_curve_by_id(lane_id).coords)
        if not at_end:
            travelled = 0.0
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                travelled += hypot(x2 - x1, y2 - y1)
                if s < travelled:
                    return atan2(y2 - y1, x2 - x1)
        (x1, y1), (x2, y2) = coords[-2], coords[-1]
        return atan2(y2 - y1, x2 - x1)

    @cached_property
    def _signal_entries(self) -> dict[tuple[str, str], float]:
        """Earliest finite signal overlap position on each controlled lane."""
        entries = {}
        for signal_id, records in self.signal_lane_overlaps.items():
            for record in records:
                start = record.start_s
                if start is None or not isfinite(start):
                    continue
                length = self._service.get_lane_central_curve_by_id(record.lane_id).length
                if 0 <= start <= length:
                    key = signal_id, record.lane_id
                    entries[key] = min(entries.get(key, start), start)
        return entries

    def _approach(self, signal_id: str) -> float | None:
        """Mean direction at signal overlaps, or None if the direction is unknown."""
        lanes = self.lanes_controlled_by_signal[signal_id]
        if not lanes or any((signal_id, lane) not in self._signal_entries for lane in lanes):
            return None
        directions = [self._direction(lane, s=self._signal_entries[signal_id, lane]) for lane in lanes]
        x, y = sum(map(cos, directions)), sum(map(sin, directions))
        return None if hypot(x, y) < 1e-9 * len(lanes) else atan2(y, x)

    def _turns_left(self, signal_id: str, path: tuple[str, ...]) -> bool:
        """Whether a movement turns left or makes a U-turn after its signal overlap."""
        turn = degrees(normalize_angle(self._direction(path[-1], at_end=True)
                                       - self._direction(path[0], s=self._signal_entries[signal_id, path[0]])))
        return turn > LEFT_TURN or turn < -OPPOSING_APPROACH

    def _movement_curves(self, signal_id: str, path: tuple[str, ...]) -> Iterator[tuple[str, LineString | Point]]:
        """Lane geometries after the signal, excluding the uncontrolled upstream prefix."""
        for index, lane_id in enumerate(path):
            line = self._service.get_lane_central_curve_by_id(lane_id)
            if index == 0:
                line = substring(line, self._signal_entries[signal_id, lane_id], line.length)
            yield lane_id, line

    def _movements_conflict(self, left: str, p: tuple[str, ...], right: str, q: tuple[str, ...]) -> bool:
        left_curves, right_curves = list(self._movement_curves(left, p)), list(self._movement_curves(right, q))
        return any(a != b and x.intersects(y) for a, x in left_curves for b, y in right_curves)

    def get_signals_wrt(self, signal_id: str) -> list[tuple[str, str]]:
        if signal_id not in self._service.signal_table:
            raise KeyError(signal_id)
        return [(other, data["relation"])
                for other, data in self.signal_graph[signal_id].items()]
