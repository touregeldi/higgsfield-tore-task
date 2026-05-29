import os
import pytest
from concurrent.futures import ThreadPoolExecutor
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


def test_sessions_do_not_bleed(client):
    def post(session, city):
        return client.post("/turns", json={"session_id": session, "user_id": session,
            "messages": [{"role": "user", "content": f"I live in {city}"}],
            "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(), "metadata": {}})
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda a: post(*a), [("a", "Paris"), ("b", "Tokyo"), ("c", "Cairo")]))
    a = client.post("/recall", json={"query": "where do I live", "session_id": "a",
                                     "user_id": "a", "max_tokens": 200}).json()
    assert "Paris" in a["context"]
    assert "Tokyo" not in a["context"] and "Cairo" not in a["context"]
