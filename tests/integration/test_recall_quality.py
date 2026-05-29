import json, os, pathlib
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
FIX = json.loads((pathlib.Path(__file__).parents[2] / "fixtures/recall_quality.json").read_text())


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB), embedder=FakeEmbedder(384),
                    reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def _load(client):
    for convo in FIX["conversations"]:
        for i, text in enumerate(convo["turns"]):
            client.post("/turns", json={
                "session_id": convo["session_id"], "user_id": convo["user_id"],
                "messages": [{"role": "user", "content": text}],
                "timestamp": datetime(2026, 5, 29, i, tzinfo=timezone.utc).isoformat(),
                "metadata": {}})


def test_recall_quality_fixture(client):
    _load(client)
    passed = 0
    details = []
    for probe in FIX["probes"]:
        res = client.post("/recall", json={
            "query": probe["query"], "session_id": probe["session_id"],
            "user_id": probe["user_id"], "max_tokens": 256}).json()
        ctx = res["context"].lower()
        ok = True
        if "expect_substring" in probe:
            ok = ok and probe["expect_substring"].lower() in ctx
        if "expect_absent" in probe:
            ok = ok and probe["expect_absent"].lower() not in ctx
        details.append((probe["query"], ok))
        passed += int(ok)
    score = passed / len(FIX["probes"])
    print(f"RECALL_QUALITY_SCORE={score:.2f}")
    for q, ok in details:
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
    assert score >= 0.6
