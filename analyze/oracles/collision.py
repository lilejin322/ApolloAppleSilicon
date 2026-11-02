"""
Collision: the ego's body touches another road user's, and who is to blame,
by Responsibility-Sensitive Safety (Shalev-Shwartz, Shammah and Shashua,
2017) as ad-rss-lib implements it (``analyze.rss``).

At every obstacle frame the ego and each road user near it are checked from
both sides: when their constellation turns dangerous, each owes its proper
response (brake, stop moving sideways, drive away) after its response time.
At a touch the one that failed its response is to blame; when that is not the
ego, it still answers for a collision a competent and careful driver would
have avoided (UN R157 5.1.1, Annex 3). The replayed agents do not react, so
without this every agent running into a braking ego would count against the
ego, and the ego cutting in just ahead of a fast agent would not.
"""

from typing import FrozenSet

from google.protobuf.message import Message

from ...map import MapService
from ...utils import ApolloVehicleConfig
from ..replay import OBSTACLES, World
from ..rss.judge import MARGIN_CAP, RssJudge
from ..rss.types import CANONICAL, RssParameters
from .base import Oracle

__all__ = ["MARGIN_CAP", "RssOracle"]


class RssOracle(Oracle):
    """
    Report each obstacle the ego touches, once, as ``collision`` when the ego
    failed its RSS response. Otherwise the obstacle failed its own, left RSS's
    bounds (appeared in danger, out-accelerated its ``accel_max``), or neither
    was observed but the ego met every response it owed: ``collision_preventable``
    when a competent and careful driver braking in the ego's place would have
    avoided it, ``collision_by_other`` when not.

    Also measures ``margin`` (the search's f1): over the moments the ego was
    failing RSS, the least room in metres both had left before a collision
    that braking at ``brake_max`` could not avoid; in dangers the other caused
    without a touch, the ego's closest approach where a careful driver would
    have kept clearer; ``MARGIN_CAP`` when none; ``worst`` says when and
    against whom.

    :param MapService map_service: The map Apollo drove.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param RssParameters params: RSS dynamics; the library's canonical set by default.
    """

    kind = "collision"
    channels: FrozenSet[str] = frozenset({OBSTACLES})

    def __init__(self, map_service: MapService, vehicle: ApolloVehicleConfig,
                 params: RssParameters = CANONICAL) -> None:
        super().__init__()
        self.judge = RssJudge(map_service, vehicle, params)

    @property
    def margin(self) -> float:
        """The ego's least margin in metres once the record has ended."""
        return self.judge.margin

    @property
    def verdicts(self):
        """Each touch's ``Verdict``, once the record has ended."""
        return self.judge.verdicts

    @property
    def worst_at(self):
        """The ``Moment`` of the least margin, None while nothing lowered it."""
        return self.judge.worst_at

    @property
    def worst(self) -> str | None:
        """What happened at the least margin, for people; None while the ego never failed RSS."""
        return self.judge.worst

    def on_message(self, world: World, channel: str, message: Message) -> None:
        self.judge.add(world)

    def finish(self, world: World) -> None:
        self.judge.finish()
        self.violations.extend(self.judge.violations())
