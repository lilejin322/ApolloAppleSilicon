"""
The scenario search's three objectives, from one analyzed record.

f1 is the ego's RSS margin (``analyze.oracles.RssOracle.margin``), minimised:
while the ego failed its RSS response, the room in metres left before a
collision braking hard could not avoid, and in dangers the other caused, the
ego's closest approach where a careful driver would have kept clearer; at most
0 at a collision the ego answers for, ``MARGIN_CAP`` when nothing lowered it.
f2 is the number of distinct kinds of violation the ego committed, maximised.
``collision_by_other`` is the other road user's fault and ``prediction_nan`` judges
Prediction, not the ego's driving; neither counts.
f3 is the run's rarity in the search so far (``rarity``), maximised: what it
found and each change of Planning's state, every one worth less the more runs
had it. More changes score more, so the search drives Planning into more of
its scenarios and stages, rare ones the most, and a finding made ten times
before is worth little. It depends on the runs before, so the search computes
it anew before every selection; an evaluation scores 0.
"""

from typing import Iterable, Mapping, NamedTuple, Sequence

from ..analyze import Violation
from ..analyze.oracles.collision import MARGIN_CAP

__all__ = ["EGO_KINDS", "Objectives", "WEIGHTS", "WORST", "rarity", "score"]

EGO_KINDS = frozenset({"collision", "collision_preventable", "red_light", "speeding", "lane_marking", "comfort",
                       "estop", "junction_lane_change", "planning_gap", "planning_error"})
"""
The kinds the ego answers for: every kind ``default_oracles`` reports but ``collision_by_other``
(the other road user's fault, which no careful driver could have prevented) and ``prediction_nan``
(Prediction's output, not the ego's driving). ``collision_preventable`` counts: the other caused it, but
the ego could have prevented it.
"""
WEIGHTS = (-1.0, 1.0, 1.0)
"""DEAP weights of ``Objectives``: the margin minimised, the kinds and the rarity maximised."""


class Objectives(NamedTuple):
    """
    One scenario's fitness.

    :param float margin: The ego's RSS margin in metres (f1, minimised).
    :param int kinds: Distinct kinds of violation the ego committed (f2, maximised).
    :param float rarity: How rare what it found and Planning's changes were in the search (f3, maximised).
    """

    margin: float
    kinds: int
    rarity: float = 0.0


WORST = Objectives(MARGIN_CAP, 0)
"""The fitness of a run that tested nothing: the scenario could not be built, or the ego never drove."""


def score(violations: Iterable[Violation], margin: float) -> Objectives:
    """
    The objectives of one record.

    :param violations: What the oracles found.
    :param float margin: ``RssOracle.margin``.
    :rtype: Objectives
    """
    return Objectives(float(margin), len({violation.kind for violation in violations} & EGO_KINDS))


def rarity(signature: Iterable[str], transitions: Sequence[str], counts: Mapping[str, int]) -> float:
    """
    f3 of one run: 1/n for each key of what it found and for each change of
    Planning's state it went through, n the runs of the search that had that
    key or change, this one included. Each change counts as often as it
    happened; a key once.

    :param signature: The keys of what it found (``search.signature``).
    :param transitions: Planning's changes of state (``RecordAnalysis.transitions``).
    :param counts: For each key and change, the runs that had it.
    :rtype: float
    """
    return (sum(1.0 / max(counts.get(key, 0), 1) for key in set(signature))
            + sum(1.0 / max(counts.get(change, 0), 1) for change in transitions))
