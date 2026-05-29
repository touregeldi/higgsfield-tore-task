import os
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from memory_service.app import build_app
from memory_service.config import Settings
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
pytestmark = pytest.mark.integration

DB = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB), embedder=FakeEmbedder(384),
                    reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def _turn(content, session="s1", user="u1"):
    return {"session_id": session, "user_id": user,
            "messages": [{"role": "user", "content": content}],
            "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(),
            "metadata": {}}


def test_health(client):
    assert client.get("/health").status_code == 200


def test_turns_then_recall_roundtrip(client):
    r = client.post("/turns", json=_turn("I live in Berlin"))
    assert r.status_code == 201 and "id" in r.json()
    rec = client.post("/recall", json={"query": "where do I live", "session_id": "s1",
                                       "user_id": "u1", "max_tokens": 200})
    assert rec.status_code == 200
    assert "Berlin" in rec.json()["context"]


def test_memories_listing_shape(client):
    client.post("/turns", json=_turn("I work at Stripe"))
    body = client.get("/users/u1/memories").json()
    assert body["memories"]
    m = body["memories"][0]
    assert {"id", "type", "key", "value", "confidence", "active"} <= set(m)


def test_malformed_turn_returns_422(client):
    assert client.post("/turns", json={"session_id": "s1"}).status_code == 422


def test_delete_session_204(client):
    client.post("/turns", json=_turn("I live in Berlin"))
    assert client.delete("/sessions/s1").status_code == 204


def test_recall_cold_session_200_empty(client):
    r = client.post("/recall", json={"query": "x", "session_id": "none", "user_id": "none"})
    assert r.status_code == 200 and r.json()["context"] == ""
