"""
Analyze recorded scenarios: read Cyber records, replay them, and judge what the ADS did wrong.

``RecordAnalyzer(map_service, vehicle, descriptor_path).analyze(record)``
returns what one record shows, a ``RecordAnalysis``: its violations, the ego's
RSS margin and the ``Moment`` it was least, the RSS ``Verdict`` of each
collision, and the states Planning went through.

``RecordReader`` and ``Replay`` are the parts: one reads a record's messages,
the other keeps the ``World`` they describe and shows it to the oracles.
``oracles`` holds the oracles, ``rss`` the Responsibility-Sensitive Safety that
settles who answers for a collision, and ``footprint`` the body geometry they
share. Channel names are in ``replay``.
"""

from .analyzer import RecordAnalysis, RecordAnalyzer, default_oracles
from .oracles import Oracle, Violation
from .planning_states import planning_transitions
from .record import RecordError, RecordMessage, RecordReader
from .replay import EgoState, Obstacle, Replay, World
from .rss import Moment, Verdict

__all__ = ["EgoState", "Moment", "Obstacle", "Oracle", "RecordAnalysis", "RecordAnalyzer", "RecordError",
           "RecordMessage", "RecordReader", "Replay", "Verdict", "Violation", "World", "default_oracles",
           "planning_transitions"]
