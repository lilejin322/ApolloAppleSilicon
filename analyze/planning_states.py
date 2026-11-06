"""
The states Planning went through in a record.

Planning is a two-level state machine: it picks a scenario (lane follow, a
protected or unprotected junction, a stop sign, pull over, ...) and steps
through that scenario's stages (approach, creep, intersection cruise, ...).
Each ``ADCTrajectory`` names both in ``debug.planning_data.scenario``.
"""

from typing import Iterable, List, Tuple
from google.protobuf.message import Message

__all__ = ["planning_transitions"]


def planning_transitions(planning: Iterable[Message]) -> Tuple[str, ...]:
    """
    Each change of Planning's state, in order, as ``SCENARIO/STAGE->SCENARIO/STAGE``.

    Messages without a scenario, published while Planning fails, are left
    out, so that a failure is not a change and the recovery from it not another.

    :param Iterable[Message] planning: The record's ``ADCTrajectory`` messages, in time order.
    :returns: One entry per change, oldest first; empty when Planning stayed in one state.
    :rtype: Tuple[str, ...]
    """
    changes: List[str] = []
    state = None
    for message in planning:
        scenario = message.debug.planning_data.scenario
        if not scenario.scenario_plugin_type:
            continue
        now = f"{scenario.scenario_plugin_type}/{scenario.stage_plugin_type}"
        if state is not None and now != state:
            changes.append(f"{state}->{now}")
        state = now
    return tuple(changes)
