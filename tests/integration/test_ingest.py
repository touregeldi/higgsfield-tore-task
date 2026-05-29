import pytest
from datetime import datetime, timezone
from memory_service.services.ingest import IngestService
from memory_service.repositories.turns import TurnRepository
from memory_service.repositories.memories import MemoryRepository
from memory_service.recall.embedder import FakeEmbedder
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _seq():
    n = {"i": 0}
    def f():
        n["i"] += 1
        return f"id-{n['i']}"
    return f


def _svc(pool):
    return IngestService(TurnRepository(pool), MemoryRepository(pool),
                         FakeEmbedder(384), NullLLMClient(), id_factory=_seq())


def _turn(content):
    return Turn(id="ignored", session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": content}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={})


def test_ingest_persists_turn_and_extracts(pool):
    svc = _svc(pool)
    turn_id = svc.ingest(_turn("I live in Berlin"))
    mems = MemoryRepository(pool).list_for_user("u1")
    assert any(m.key == "location" and m.value == "Berlin" and m.source_turn == turn_id for m in mems)


def test_ingest_evolves_fact(pool):
    svc = _svc(pool)
    svc.ingest(_turn("I live in Berlin"))
    svc.ingest(_turn("I just moved to Munich"))
    active = MemoryRepository(pool).active_by_key("u1")
    assert active["location"][1] == "Munich"
    all_rows = MemoryRepository(pool).list_for_user("u1", include_superseded=True)
    assert any(m.value == "Berlin" and m.active is False for m in all_rows)
