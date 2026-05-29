import pytest
from memory_service.repositories.memories import MemoryRepository
from memory_service.repositories.turns import TurnRepository
from memory_service.models.domain import Turn, MemoryCandidate, MemoryType
from datetime import datetime, timezone
pytestmark = pytest.mark.integration


def _seed_turn(pool, ids, tid):
    TurnRepository(pool).insert(Turn(id=tid, session_id="s1", user_id="u1",
        messages=[{"role": "user", "content": "x"}],
        timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={}))


def _cand(key, value):
    return MemoryCandidate(type=MemoryType.fact, key=key, value=value, confidence=0.9)


def test_insert_and_active_by_key(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    repo.insert(ids(), "u1", "s1", _cand("location", "Berlin"), tid, [0.1] * 384, None)
    active = repo.active_by_key("u1")
    assert active["location"][1] == "Berlin"


def test_supersede_marks_old_inactive(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    old_id = ids()
    repo.insert(old_id, "u1", "s1", _cand("location", "Berlin"), tid, [0.1] * 384, None)
    repo.insert(ids(), "u1", "s1", _cand("location", "Munich"), tid, [0.2] * 384, supersedes=old_id)
    active = repo.active_by_key("u1")
    assert active["location"][1] == "Munich"
    all_rows = repo.list_for_user("u1", include_superseded=True)
    assert any(m.value == "Berlin" and m.active is False and m.supersedes is None for m in all_rows)
    assert any(m.value == "Munich" and m.supersedes == old_id for m in all_rows)


def test_search_vector_and_fts_return_ids(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    mid = ids()
    repo.insert(mid, "u1", "s1", _cand("location", "Berlin"), tid, [0.5] * 384, None)
    vec_ids = repo.search_vector("u1", "s1", [0.5] * 384, limit=5)
    fts_ids = repo.search_fts("u1", "s1", "Berlin", limit=5)
    assert mid in vec_ids
    assert mid in fts_ids
