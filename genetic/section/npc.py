"""NPC-vehicle parts of the section operators.

``NPCKind`` draws, mutates, crosses and rebalances ``NPCVehicleAgent`` values
on the per-type route spaces.
"""

from random import Random
from typing import List, Sequence, Tuple, Type

from ...scenario import NPCVehicleAgent, VehicleType, random_vehicle_type
from ...scenario.routing import InvalidRoute
from ...scenario.sections import NPCSection, SectionRules, Spaces
from ..npc_agent import NPCOperatorConfig, cx_npc_agent, mut_npc_agent, rebalance_vehicle_types
from .base import SectionKind


class NPCKind(SectionKind[NPCVehicleAgent]):
    """
    NPC vehicles on per-type spaces; types drawn from ``config.vehicle_type_weights``.

    :param Spaces spaces: Route space per vehicle type.
    :param NPCOperatorConfig config: NPC operators. Defaults to ``NPCOperatorConfig()``.
    :param SectionRules | None rules: In-type constraints. None uses ``NPCSection.RULES``.
    """

    SECTION: Type[NPCSection] = NPCSection
    """NPC section class."""

    def __init__(self, spaces: Spaces, config: NPCOperatorConfig = NPCOperatorConfig(),
                 rules: SectionRules | None = None) -> None:
        """
        Bind the NPC section to its spaces and operators.

        :param Spaces spaces: Route space per vehicle type. Stored as ``context``.
        :param NPCOperatorConfig config: NPC operators. Defaults to ``NPCOperatorConfig()``.
        :param SectionRules | None rules: In-type constraints. None uses ``NPCSection.RULES``.
        """
        super().__init__(NPCSection, spaces, rules)
        self.config: NPCOperatorConfig = config

    def random_agent(self, rng: Random) -> NPCVehicleAgent | None:
        """
        Draw one NPC, or None when its type has no space or the draw fails.

        :param Random rng: Random source for the vehicle type and the agent.
        :returns: The agent, or None.
        :rtype: NPCVehicleAgent | None
        """
        vehicle_type: VehicleType = random_vehicle_type(rng, self.config.vehicle_type_weights)
        if vehicle_type not in self.context:
            return None
        try:
            return NPCVehicleAgent.random(self.context[vehicle_type], rng, vehicle_type=vehicle_type,
                                          max_start_t=self.config.max_start_t)
        except (InvalidRoute, ValueError, RuntimeError):
            return None

    def mutate_agent(self, agent: NPCVehicleAgent, rng: Random) -> NPCVehicleAgent:
        """
        Bottom-layer mutation of one NPC.

        :param NPCVehicleAgent agent: Agent to mutate.
        :param Random rng: Random source passed to ``mut_npc_agent``.
        :returns: The mutant.
        :rtype: NPCVehicleAgent
        """
        return mut_npc_agent(agent, self.context, rng, self.config)

    def crossover_agents(self, first: NPCVehicleAgent, second: NPCVehicleAgent,
                         rng: Random) -> Tuple[NPCVehicleAgent, NPCVehicleAgent]:
        """
        Bottom-layer crossover of two NPCs.

        :param NPCVehicleAgent first: Leading parent.
        :param NPCVehicleAgent second: Other parent.
        :param Random rng: Random source passed to ``cx_npc_agent``.
        :returns: ``(child led by first, child led by second)``.
        :rtype: Tuple[NPCVehicleAgent, NPCVehicleAgent]
        """
        return cx_npc_agent(first, second, self.context, rng, self.config)

    def rebalance(self, agents: Sequence[NPCVehicleAgent], rng: Random) -> List[NPCVehicleAgent]:
        """
        Restore the vehicle-type mix of ``agents``.

        :param Sequence[NPCVehicleAgent] agents: Agents to rebalance. The input sequence is not modified.
        :param Random rng: Random source passed to ``rebalance_vehicle_types``.
        :returns: The rebalanced agents.
        :rtype: List[NPCVehicleAgent]
        """
        return rebalance_vehicle_types(agents, self.context, rng, self.config)
