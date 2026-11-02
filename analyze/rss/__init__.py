"""
Responsibility-Sensitive Safety, ported from Intel's ad-rss-lib 5.0.0, to
judge recorded Apollo driving: who owed which response in each conflict, and
who failed it.

``physics`` and ``formulas`` hold the library's arithmetic, ``structured``
and ``unstructured`` its checkers, ``trajectories`` the unstructured
trajectory sets, ``constellations`` the map integration on ``MapService``,
and ``judge`` the record-level verdicts and margin that ``RssOracle`` reports.
"""

from .judge import MARGIN_CAP, RssJudge, Verdict
from .types import CANONICAL, Dynamics, RssParameters

__all__ = ["CANONICAL", "Dynamics", "MARGIN_CAP", "RssJudge", "RssParameters", "Verdict"]
