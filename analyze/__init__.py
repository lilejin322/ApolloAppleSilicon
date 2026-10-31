"""
Analyze recorded scenarios: read Cyber records, replay them, and judge what the ADS did wrong.

``analyze(record, map_service=..., vehicle=..., descriptor_path=...)`` returns
the violations of one record; ``RecordReader`` and ``Replay`` are the parts.
"""

from .analyzer import analyze, default_oracles
from .oracles import Oracle, Violation
from .record import RecordError, RecordMessage, RecordReader
from .replay import EgoState, Obstacle, Replay, World

__all__ = ["EgoState", "Obstacle", "Oracle", "RecordError", "RecordMessage", "RecordReader", "Replay", "Violation",
           "World", "analyze", "default_oracles"]
