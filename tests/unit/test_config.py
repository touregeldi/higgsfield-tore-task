from memory_service.config import Settings


def test_settings_defaults_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.max_body_bytes == 1048576
    assert s.llm_enabled is False


def test_llm_enabled_when_key_present(monkeypatch):
    s = Settings(_env_file=None, openai_api_key="sk-test")
    assert s.llm_enabled is True
