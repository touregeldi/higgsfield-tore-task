import os, subprocess, time, json, urllib.request
import pytest
pytestmark = pytest.mark.integration

BASE = "http://localhost:8080"


def _up():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _up(), reason="live compose stack not running")
def test_data_survives_restart():
    payload = json.dumps({"session_id": "persist", "user_id": "persist",
        "messages": [{"role": "user", "content": "I live in Lisbon"}],
        "timestamp": "2026-05-29T00:00:00Z", "metadata": {}}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{BASE}/turns", payload,
        {"content-type": "application/json"}))
    subprocess.run(["docker", "compose", "restart", "app", "db"], check=True)
    for _ in range(30):
        if _up():
            break
        time.sleep(2)
    q = json.dumps({"query": "where do I live", "session_id": "persist",
                    "user_id": "persist", "max_tokens": 200}).encode()
    resp = urllib.request.urlopen(urllib.request.Request(f"{BASE}/recall", q,
        {"content-type": "application/json"}))
    assert "Lisbon" in json.loads(resp.read())["context"]
