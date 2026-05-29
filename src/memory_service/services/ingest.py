from __future__ import annotations
import uuid
from typing import Callable
from ..models.domain import Turn
from ..repositories.turns import TurnRepository
from ..repositories.memories import MemoryRepository
from ..recall.embedder import Embedder
from ..llm.client import LLMClient
from ..extraction.pipeline import extract_candidates
from .reconcile import plan_reconciliation, Action


class IngestService:
    def __init__(self, turns: TurnRepository, memories: MemoryRepository,
                 embedder: Embedder, llm: LLMClient,
                 id_factory: Callable[[], str] | None = None):
        self._turns = turns
        self._memories = memories
        self._embedder = embedder
        self._llm = llm
        self._id = id_factory or (lambda: uuid.uuid4().hex)

    def ingest(self, turn: Turn) -> str:
        turn_id = self._id()
        stored = Turn(id=turn_id, session_id=turn.session_id, user_id=turn.user_id,
                      messages=turn.messages, timestamp=turn.timestamp, metadata=turn.metadata)
        self._turns.insert(stored)

        candidates = extract_candidates(turn.messages, self._llm)
        if not candidates:
            return turn_id

        existing = self._memories.active_by_key(turn.user_id)
        actions = plan_reconciliation(candidates, existing)
        to_embed = [a for a in actions if a.kind == Action.INSERT]
        vectors = self._embedder.embed([a.candidate.value for a in to_embed]) if to_embed else []
        for action, vec in zip(to_embed, vectors):
            self._memories.insert(self._id(), turn.user_id, turn.session_id,
                                  action.candidate, turn_id, vec, action.supersedes)
        return turn_id
