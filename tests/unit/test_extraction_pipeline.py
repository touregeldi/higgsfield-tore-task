from memory_service.extraction.pipeline import extract_candidates
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import MemoryType, MemoryCandidate


class FakeLLM:
    available = True

    def complete_json(self, prompt, timeout=8.0):
        return {"memories": [
            {"type": "opinion", "key": "opinion:ts", "value": "TS generics annoying", "confidence": 0.6}
        ]}


def _user(c):
    return [{"role": "user", "content": c}]


def test_rule_only_when_llm_unavailable():
    cands = extract_candidates(_user("I live in Berlin"), NullLLMClient())
    assert any(c.key == "location" for c in cands)
    assert all(isinstance(c, MemoryCandidate) for c in cands)


def test_llm_candidates_merged_when_available():
    cands = extract_candidates(_user("I live in Berlin"), FakeLLM())
    assert any(c.key == "location" for c in cands)            # from rules
    assert any(c.type is MemoryType.opinion for c in cands)   # from LLM


def test_malformed_llm_output_is_ignored():
    class BadLLM:
        available = True
        def complete_json(self, prompt, timeout=8.0):
            return {"memories": [{"type": "nonsense", "value": 123}]}
    cands = extract_candidates(_user("I work at Stripe"), BadLLM())
    assert any(c.key == "employment" for c in cands)  # rules still present, bad item dropped
