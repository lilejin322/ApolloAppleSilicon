"""
Collision: the ego's body touches another road user's, and who is to blame,
by Responsibility-Sensitive Safety (Shalev-Shwartz, Shammah and Shashua,
2017) as ad-rss-lib implements it (``analyze.rss``).
See: https://github.com/intel/ad-rss-lib

At every obstacle frame the ego and each road user near it are checked from
both sides: when their constellation turns dangerous, each owes its proper
response (brake, stop moving sideways, drive away) after its response time.
At a touch the one that failed its response is to blame; when that is not the
ego, it still answers for a collision a competent and careful driver would
have avoided (UN R157 5.1.1, Annex 3). The replayed agents do not react, so
without this every agent running into a braking ego would count against the
ego, and the ego cutting in just ahead of a fast agent would not.
"""

from typing import FrozenSet, List
from google.protobuf.message import Message

from ...map import MapService
from ...utils import ApolloVehicleConfig
from ..replay import OBSTACLES, World
from ..rss.judge import MARGIN_CAP, Moment, RssJudge, Verdict
from ..rss.types import CANONICAL, RssParameters
from .base import Oracle

__all__ = ["MARGIN_CAP", "RssOracle"]


class RssOracle(Oracle):
    """
    Report each obstacle the ego touches, once, as ``collision`` when the ego
    failed its RSS response, unless braking as owed would still have left it
    touched. Otherwise the obstacle failed its own, left RSS's bounds (appeared
    in danger, out-accelerated its ``accel_max``), or neither was observed but
    the ego met every response it owed: ``collision_preventable`` when a
    competent and careful driver braking in the ego's place would have avoided
    it, ``collision_by_other`` when not.

    Also measures ``margin`` (the search's f1): over the moments the ego was
    failing RSS, the least room in metres both had left before a collision
    that braking at ``brake_max`` could not avoid; in dangers the other caused
    without a touch, the ego's closest approach where a careful driver would
    have kept clearer; at most 0 after a touch the ego answers for;
    ``MARGIN_CAP`` when none; ``worst`` says when and against whom.

    The touch itself is the ego's footprint meeting the obstacle's, checked at
    every obstacle frame; RSS only settles who answers for it.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param RssParameters params: RSS dynamics; the library's canonical set by default.
    """
    kind = "collision"
    """The kind it is registered under; it reports ``KINDS``: ``collision`` and the two the ego is not to blame for."""
    channels: FrozenSet[str] = frozenset({OBSTACLES})
    """The obstacle frames, at which the judge takes the world."""
    judge: RssJudge
    """The RSS judge of this record's frames."""

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig,
                 params: RssParameters = CANONICAL) -> None:
        """
        Initialize the RSS judge with the map, ego geometry and response parameters.

        :param MapService map_service: The map Apollo drove.
        :param ApolloVehicleConfig vehicle: The ego's dimensions.
        :param RssParameters params: RSS dynamics.
        """
        super().__init__()
        self.judge = RssJudge(map_service, vehicle, params)

    @property
    def margin(self) -> float:
        """
        The ego's least margin in metres once the record has ended.

        :returns: Metres; at most 0 after a touch the ego answers for; ``MARGIN_CAP`` when nothing lowered it.
        :rtype: float
        """
        return self.judge.margin

    @property
    def verdicts(self) -> List[Verdict]:
        """
        Each touch's ``Verdict``, once the record has ended.

        :returns: Who answers for each touch, in time order; one per obstacle at most.
        :rtype: List[Verdict]
        """
        return self.judge.verdicts

    @property
    def worst_at(self) -> Moment | None:
        """
        The ``Moment`` of the least margin, None while nothing lowered it.

        :returns: The road user's type, the ego's constellation with it and the ego's body centre then.
        :rtype: Moment | None
        """
        return self.judge.worst_at

    @property
    def worst(self) -> str | None:
        """
        What happened at the least margin, for people; None while nothing lowered it.

        :returns: When, against whom and how close; None while nothing lowered the margin.
        :rtype: str | None
        """
        return self.judge.worst

    def on_message(self, world: World, channel: str, message: Message) -> None:
        """
        Buffer the current obstacle frame for RSS collision analysis.

        :param World world: The world at the obstacle frame.
        :param str channel: ``OBSTACLES``.
        :param Message message: The perception obstacles; the judge reads them from ``world``.
        """
        self.judge.add(world)

    def finish(self, world: World) -> None:
        """
        Finish RSS analysis and append the resulting collision violations.

        :param World world: The world at the last message; unused.
        """
        self.judge.finish()
        self.violations.extend(self.judge.violations())
