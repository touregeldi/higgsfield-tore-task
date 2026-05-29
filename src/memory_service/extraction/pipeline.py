from __future__ import annotations
from ..llm.client import LLMClient
from ..models.domain import MemoryCandidate
from .rules import extract_rules
from .llm_extractor import extract_llm


def extract_candidates(messages: list[dict], llm: LLMClient) -> list[MemoryCandidate]:
    """Rule output is the always-present baseline; LLM output augments it.
    On (key) collisions the higher-confidence candidate wins."""
    merged: dict[str, MemoryCandidate] = {}
    for c in extract_rules(messages) + extract_llm(messages, llm):
        cur = merged.get(c.key)
        if cur is None or c.confidence > cur.confidence:
            merged[c.key] = c
    return list(merged.values())
