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
    app = build_app(Settings(database_url=DB, max_body_bytes=2048),
                    embedder=FakeEmbedder(384), reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def test_unicode_roundtrip(client):
    # Use "Tokyo東京" — the location regex [A-Z][\w .'-]+ with Python 3's unicode-aware \w
    # will capture "Tokyo東京" as the location value, preserving the CJK characters.
    r = client.post("/turns", json={"session_id": "u", "user_id": "u",
        "messages": [{"role": "user", "content": "I live in Tokyo東京 now"}],
        "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(), "metadata": {}})
    assert r.status_code == 201
    body = client.get("/users/u/memories").json()
    assert any("東京" in m["value"] for m in body["memories"])


def test_oversized_payload_413(client):
    big = "x" * 5000
    r = client.post("/turns", json={"session_id": "s", "messages": [{"role": "user", "content": big}],
                                    "timestamp": "2026-05-29T00:00:00Z"})
    assert r.status_code == 413


def test_bad_role_422(client):
    r = client.post("/turns", json={"session_id": "s",
        "messages": [{"role": "wizard", "content": "hi"}], "timestamp": "2026-05-29T00:00:00Z"})
    assert r.status_code == 422
