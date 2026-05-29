from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from ..extraction.keys import normalize_value
from ..models.domain import MemoryCandidate


class Action(str, Enum):
    INSERT = "insert"
    SKIP = "skip"


@dataclass
class ReconcileAction:
    kind: Action
    candidate: MemoryCandidate
    supersedes: str | None = None


def plan_reconciliation(
    candidates: list[MemoryCandidate],
    existing_active: dict[str, tuple[str, str]],  # key -> (memory_id, value)
) -> list[ReconcileAction]:
    """Decide insert/supersede/skip per candidate against current active memories.
    existing_active maps key -> (id, value) of the current active memory for that key."""
    actions: list[ReconcileAction] = []
    for c in candidates:
        prior = existing_active.get(c.key)
        if prior is None:
            actions.append(ReconcileAction(Action.INSERT, c, supersedes=None))
        elif normalize_value(prior[1]).lower() == normalize_value(c.value).lower():
            actions.append(ReconcileAction(Action.SKIP, c))
        else:
            actions.append(ReconcileAction(Action.INSERT, c, supersedes=prior[0]))
    return actions
