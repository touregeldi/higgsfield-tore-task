import pytest
from pydantic import ValidationError
from memory_service.models.schemas import TurnRequest, RecallRequest
from memory_service.models.domain import MemoryType, MemoryCandidate


def test_turn_request_parses_minimal():
    req = TurnRequest(
        session_id="s1",
        user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
        timestamp="2026-05-29T00:00:00Z",
    )
    assert req.messages[0].role == "user"
    assert req.metadata == {}


def test_turn_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        TurnRequest(session_id="s1", messages=[], timestamp="2026-05-29T00:00:00Z")


def test_recall_request_defaults_max_tokens():
    req = RecallRequest(query="where do I live", session_id="s1")
    assert req.max_tokens == 1024


def test_memory_candidate_holds_fields():
    c = MemoryCandidate(type=MemoryType.fact, key="location", value="Berlin", confidence=0.9)
    assert c.type is MemoryType.fact
