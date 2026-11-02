"""
Analyze recorded scenarios: read Cyber records, replay them, and judge what the ADS did wrong.

``RecordAnalyzer(map_service, vehicle, descriptor_path).analyze(record)``
returns what one record shows: its violations, the ego's RSS margin and the
states Planning went through; ``RecordReader`` and ``Replay`` are the parts.
"""

from .analyzer import RecordAnalysis, RecordAnalyzer, default_oracles
from .oracles import Oracle, Violation
from .planning_states import planning_transitions
from .record import RecordError, RecordMessage, RecordReader
from .replay import EgoState, Obstacle, Replay, World

__all__ = ["EgoState", "Obstacle", "Oracle", "RecordAnalysis", "RecordAnalyzer", "RecordError", "RecordMessage",
           "RecordReader", "Replay", "Violation", "World", "default_oracles", "planning_transitions"]
