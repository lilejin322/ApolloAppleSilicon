"""Bicycle parts of the section operators.

``BicycleKind`` draws, mutates, crosses and rebalances ``BicycleAgent`` values
on the bicycles' route space.
"""

from random import Random
from typing import List, Sequence, Tuple, Type

from ...scenario import BicycleAgent, RiderType, RouteSpace, random_rider_type
from ...scenario.routing import InvalidRoute
from ...scenario.sections import BicycleSection, SectionRules
from ..bicycle import BicycleOperatorConfig, cx_bicycle, mut_bicycle, rebalance_rider_types
from .base import SectionKind


class BicycleKind(SectionKind[BicycleAgent]):
    """
    Bicycles on the bicycles' space; rider types drawn from ``config.rider_type_weights``.

    :param RouteSpace space: Bicycle route space.
    :param BicycleOperatorConfig config: Bicycle operators. Defaults to ``BicycleOperatorConfig()``.
    :param SectionRules | None rules: In-type constraints. None uses ``BicycleSection.RULES``.
    """

    SECTION: Type[BicycleSection] = BicycleSection
    """Bicycle section class."""

    def __init__(self, space: RouteSpace, config: BicycleOperatorConfig = BicycleOperatorConfig(),
                 rules: SectionRules | None = None) -> None:
        """
        Bind the bicycle section to its space and operators.

        :param RouteSpace space: Bicycle route space. Stored as ``context``.
        :param BicycleOperatorConfig config: Bicycle operators. Defaults to ``BicycleOperatorConfig()``.
        :param SectionRules | None rules: In-type constraints. None uses ``BicycleSection.RULES``.
        """
        super().__init__(BicycleSection, space, rules)
        self.config: BicycleOperatorConfig = config

    def random_agent(self, rng: Random) -> BicycleAgent | None:
        """
        Draw one bicycle, or None when the draw fails.

        :param Random rng: Random source for the rider type and the agent.
        :returns: The agent, or None.
        :rtype: BicycleAgent | None
        """
        rider_type: RiderType = random_rider_type(rng, self.config.rider_type_weights)
        try:
            return BicycleAgent.random(self.context, rng, rider_type=rider_type, max_start_t=self.config.max_start_t)
        except (InvalidRoute, ValueError, RuntimeError):
            return None

    def mutate_agent(self, agent: BicycleAgent, rng: Random) -> BicycleAgent:
        """
        Bottom-layer mutation of one bicycle.

        :param BicycleAgent agent: Agent to mutate.
        :param Random rng: Random source passed to ``mut_bicycle``.
        :returns: The mutant.
        :rtype: BicycleAgent
        """
        return mut_bicycle(agent, self.context, rng, self.config)

    def crossover_agents(self, first: BicycleAgent, second: BicycleAgent, rng: Random,
                         ) -> Tuple[BicycleAgent, BicycleAgent]:
        """
        Bottom-layer crossover of two bicycles.

        :param BicycleAgent first: Leading parent.
        :param BicycleAgent second: Other parent.
        :param Random rng: Random source passed to ``cx_bicycle``.
        :returns: ``(child led by first, child led by second)``.
        :rtype: Tuple[BicycleAgent, BicycleAgent]
        """
        return cx_bicycle(first, second, self.context, rng, self.config)

    def rebalance(self, agents: Sequence[BicycleAgent], rng: Random) -> List[BicycleAgent]:
        """
        Restore the rider-type mix of ``agents``.

        :param Sequence[BicycleAgent] agents: Agents to rebalance. The input sequence is not modified.
        :param Random rng: Random source passed to ``rebalance_rider_types``.
        :returns: The rebalanced agents.
        :rtype: List[BicycleAgent]
        """
        return rebalance_rider_types(agents, self.context, rng, self.config)
