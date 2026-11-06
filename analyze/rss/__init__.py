"""
Responsibility-Sensitive Safety, ported from Intel's ad-rss-lib 5.0.0, to
judge recorded Apollo driving: who owed which response in each conflict, and
who failed it.
See: https://github.com/intel/ad-rss-lib

``types`` holds the library's data, ``physics`` and ``formulas`` its
arithmetic, ``structured`` and ``unstructured`` its checkers, ``trajectories``
the unstructured trajectory sets, and ``constellations`` the map integration
on ``MapService``. ``preventable`` is not from the library: it is the
competent and careful driver of UN Regulation No. 157, Annex 3, which settles
whether a collision the other road user caused was still preventable.
``judge`` puts them together into the record-level verdicts and margin that
``RssOracle`` reports.

Only what a caller of the judge needs is exported here: the judge, its
parameters, and what it returns (``Verdict``, one of ``KINDS``, and the
``Moment`` of the least margin). The rest is imported from its module.
"""

from .judge import KINDS, MARGIN_CAP, Moment, RssJudge, Verdict
from .types import CANONICAL, Dynamics, RssParameters

__all__ = ["CANONICAL", "Dynamics", "KINDS", "MARGIN_CAP", "Moment", "RssJudge", "RssParameters", "Verdict"]
