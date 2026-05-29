from memory_service.llm.client import NullLLMClient


def test_null_client_unavailable_and_returns_none():
    c = NullLLMClient()
    assert c.available is False
    assert c.complete_json("anything") is None
