"""Pedestrian parts of the section operators.

``PedestrianKind`` draws, mutates, crosses and rebalances ``PedestrianAgent``
values on the walk network.
"""

from random import Random
from typing import List, Sequence, Tuple, Type

from ...scenario import PedestrianAgent, WalkNetwork, WalkerType, random_walker_type
from ...scenario.routing import InvalidRoute
from ...scenario.sections import PedestrianSection, SectionRules
from ..pedestrian import PedestrianOperatorConfig, cx_pedestrian, mut_pedestrian, rebalance_crossing_kinds
from .base import SectionKind


class PedestrianKind(SectionKind[PedestrianAgent]):
    """
    Pedestrians on the walk network. Generation keeps the defaults of
    ``PedestrianAgent.random``: 25% of crossings off the crosswalk and 5% mid-road
    hesitations stay as approved. Walker types need no rebalancing: the walker-type
    mutation resamples from the mix and every type fits every walk (P5: stable over 30 generations).
    The crosswalk / off-crosswalk split is not: walkers sharing a crosswalk are rejected
    more often by the no-overlap check, so ``rebalance`` restores it over the population.

    :param WalkNetwork network: Walk network the pedestrians replay on.
    :param PedestrianOperatorConfig config: Pedestrian operators. Defaults to ``PedestrianOperatorConfig()``.
    :param SectionRules | None rules: In-type constraints. None uses ``PedestrianSection.RULES``.
    """

    SECTION: Type[PedestrianSection] = PedestrianSection
    """Pedestrian section class."""

    def __init__(self, network: WalkNetwork, config: PedestrianOperatorConfig = PedestrianOperatorConfig(),
                 rules: SectionRules | None = None) -> None:
        """
        Bind the pedestrian section to its walk network and operators.

        :param WalkNetwork network: Walk network. Stored as ``context``.
        :param PedestrianOperatorConfig config: Pedestrian operators. Defaults to ``PedestrianOperatorConfig()``.
        :param SectionRules | None rules: In-type constraints. None uses ``PedestrianSection.RULES``.
        """
        super().__init__(PedestrianSection, network, rules)
        self.config: PedestrianOperatorConfig = config

    def random_agent(self, rng: Random) -> PedestrianAgent | None:
        """
        Draw one pedestrian, or None when the draw fails.

        :param Random rng: Random source for the walker type and the agent.
        :returns: The agent, or None.
        :rtype: PedestrianAgent | None
        """
        walker_type: WalkerType = random_walker_type(rng, self.config.walker_type_weights)
        try:
            return PedestrianAgent.random(self.context, rng, walker_type=walker_type,
                                          max_start_t=self.config.max_start_t)
        except (InvalidRoute, ValueError, RuntimeError):
            return None

    def mutate_agent(self, agent: PedestrianAgent, rng: Random) -> PedestrianAgent:
        """
        Bottom-layer mutation of one pedestrian.

        :param PedestrianAgent agent: Agent to mutate.
        :param Random rng: Random source passed to ``mut_pedestrian``.
        :returns: The mutant.
        :rtype: PedestrianAgent
        """
        return mut_pedestrian(agent, self.context, rng, self.config)

    def crossover_agents(self, first: PedestrianAgent, second: PedestrianAgent,
                         rng: Random) -> Tuple[PedestrianAgent, PedestrianAgent]:
        """
        Bottom-layer crossover of two pedestrians.

        :param PedestrianAgent first: Leading parent.
        :param PedestrianAgent second: Other parent.
        :param Random rng: Random source passed to ``cx_pedestrian``.
        :returns: ``(child led by first, child led by second)``.
        :rtype: Tuple[PedestrianAgent, PedestrianAgent]
        """
        return cx_pedestrian(first, second, self.context, rng, self.config)

    def rebalance(self, agents: Sequence[PedestrianAgent], rng: Random) -> List[PedestrianAgent]:
        """
        Restore the crosswalk / off-crosswalk split of ``agents``.

        :param Sequence[PedestrianAgent] agents: Agents to rebalance. The input sequence is not modified.
        :param Random rng: Random source passed to ``rebalance_crossing_kinds``.
        :returns: The rebalanced agents.
        :rtype: List[PedestrianAgent]
        """
        return rebalance_crossing_kinds(agents, self.context, rng, self.config)
