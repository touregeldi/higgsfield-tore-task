import pytest
from datetime import datetime, timezone
from memory_service.services.ingest import IngestService
from memory_service.services.recall_service import RecallService
from memory_service.repositories.turns import TurnRepository
from memory_service.repositories.memories import MemoryRepository
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _ingest(pool):
    seq = {"i": 0}
    def idf():
        seq["i"] += 1
        return f"id-{seq['i']}"
    return IngestService(TurnRepository(pool), MemoryRepository(pool),
                         FakeEmbedder(384), NullLLMClient(), id_factory=idf)


def _recall(pool):
    return RecallService(MemoryRepository(pool), TurnRepository(pool),
                         FakeEmbedder(384), FakeReranker(), NullLLMClient())


def _turn(content):
    return Turn(id="x", session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": content}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={})


def test_recall_returns_relevant_fact_with_citation(pool):
    ing = _ingest(pool)
    ing.ingest(_turn("I live in Berlin"))
    ing.ingest(_turn("I work at Stripe"))
    res = _recall(pool).recall("where do I live", "s1", "u1", max_tokens=200)
    assert "Berlin" in res.context
    assert any(c.snippet for c in res.citations)


def test_cold_session_returns_empty_never_errors(pool):
    res = _recall(pool).recall("anything", "cold-session", "nobody", max_tokens=200)
    assert res.context == "" and res.citations == []


def test_noise_query_does_not_invent(pool):
    _ingest(pool).ingest(_turn("I live in Berlin"))
    res = _recall(pool).recall("what is the capital of France", "s1", "u1", max_tokens=200)
    assert "France" not in res.context
