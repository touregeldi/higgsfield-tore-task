import pytest
from datetime import datetime, timezone
from memory_service.repositories.turns import TurnRepository
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _turn(ids):
    return Turn(id=ids(), session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": "I live in Berlin"}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={"x": 1})


def test_insert_and_get(pool, ids):
    repo = TurnRepository(pool)
    t = _turn(ids)
    repo.insert(t)
    got = repo.get(t.id)
    assert got.messages[0]["content"] == "I live in Berlin"
    assert got.metadata == {"x": 1}


def test_recent_for_session(pool, ids):
    repo = TurnRepository(pool)
    for _ in range(3):
        repo.insert(_turn(ids))
    recent = repo.recent_for_session("s1", limit=2)
    assert len(recent) == 2


def test_delete_session(pool, ids):
    repo = TurnRepository(pool)
    t = _turn(ids)
    repo.insert(t)
    repo.delete_session("s1")
    assert repo.get(t.id) is None
