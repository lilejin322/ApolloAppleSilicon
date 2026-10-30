"""
Stable perception-obstacle IDs carried on the participant chromosomes.

Each agent type owns one block of ``ID_BLOCK`` IDs, so IDs never collide across
types: NPC vehicles ``[1000, 1999]``, bicycles ``[2000, 2999]``, pedestrians
``[3000, 3999]``. The ID is a gene-free label (``lineage_id`` is the GA identity, below): every section renumbers its agents
densely in section order (``assign_ids``: ``base, base + 1, ...``, no holes),
after any GA operator too, and ``ObstacleManager`` publishes it unchanged by ``ObstacleManager``, so a
scenario JSON dumped before a run matches the ``PerceptionObstacles`` log.
"""

from dataclasses import replace
from uuid import uuid4
from typing import List, Sequence, TypeVar

NPC_ID_BASE = 1000
"""First obstacle ID for NPC vehicles."""
BICYCLE_ID_BASE = 2000
"""First obstacle ID for bicycles."""
PEDESTRIAN_ID_BASE = 3000
"""First obstacle ID for pedestrians."""
ID_BLOCK = 1000
"""IDs per agent type."""

A = TypeVar("A")


def new_lineage_id() -> str:
    """
    A fresh ``lineage_id``: 32 hex digits of ``uuid4``.

    ``obstacle_id`` is a dense per-section position; ``lineage_id`` is the
    agent's identity for GA tracing. It is drawn once when an agent is created
    (random generation, a retyped replacement keeps the old one), survives
    ``dataclasses.replace`` (mutation, in-place crossover) and every
    ``obstacle_id`` renumbering, and is unique across runs without a counter.
    """
    return uuid4().hex


def dedupe_lineage(agents: Sequence[A]) -> List[A]:
    """
    ``agents`` where a repeated ``lineage_id`` (clones pooled by set crossover)
    is replaced by a fresh one; the first holder keeps it.
    """
    seen, out = set(), []
    for agent in agents:
        if agent.lineage_id in seen:
            agent = replace(agent, lineage_id=new_lineage_id())
        seen.add(agent.lineage_id)
        out.append(agent)
    return out


def check_id(obstacle_id: int | None, base: int, kind: str) -> None:
    """
    Check that ``obstacle_id`` is unset or an int in ``[base, base + ID_BLOCK)``.

    :raises ValueError: If it is not an int or lies outside the type's block.
    """
    if obstacle_id is None:
        return
    if isinstance(obstacle_id, bool) or not isinstance(obstacle_id, int):
        raise ValueError(f"{kind} obstacle_id must be an int, got {obstacle_id!r}")
    if not base <= obstacle_id < base + ID_BLOCK:
        raise ValueError(f"{kind} obstacle_id {obstacle_id} outside [{base}, {base + ID_BLOCK - 1}]")


def assign_ids(agents: Sequence[A], base: int, kind: str) -> List[A]:
    """
    ``agents`` renumbered densely in order: ``base, base + 1, ...``, no holes.

    Within one scenario the IDs are always contiguous, so a dump never shows
    gaps left by agents a GA operator removed. An agent whose ID already
    matches keeps the same object.

    :raises ValueError: When there are more than ``ID_BLOCK`` agents.
    """
    if len(agents) > ID_BLOCK:
        raise ValueError(f"more than {ID_BLOCK} {kind} agents")
    return [agent if agent.obstacle_id == base + index else replace(agent, obstacle_id=base + index)
            for index, agent in enumerate(agents)]
