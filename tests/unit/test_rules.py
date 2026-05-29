from memory_service.extraction.rules import extract_rules
from memory_service.models.domain import MemoryType


def _user(content):
    return [{"role": "user", "content": content}]


def test_extracts_location():
    cands = extract_rules(_user("I live in Berlin"))
    assert any(c.key == "location" and c.value == "Berlin" and c.type is MemoryType.fact for c in cands)


def test_extracts_employment():
    cands = extract_rules(_user("I work at Stripe"))
    assert any(c.key == "employment" and c.value == "Stripe" for c in cands)


def test_extracts_implicit_pet():
    cands = extract_rules(_user("just got back from walking Biscuit"))
    assert any(c.key.startswith("pet.name") and "Biscuit" in c.value for c in cands)


def test_extracts_preference():
    cands = extract_rules(_user("I love TypeScript"))
    assert any(c.type is MemoryType.preference and "TypeScript" in c.value for c in cands)


def test_correction_lowers_nothing_but_still_extracts():
    cands = extract_rules(_user("Actually I no longer work at Stripe, I work at Google now"))
    assert any(c.key == "employment" and c.value == "Google" for c in cands)


def test_ignores_assistant_and_tool_messages():
    msgs = [{"role": "assistant", "content": "I live in Berlin"},
            {"role": "tool", "name": "x", "content": "I work at Stripe"}]
    assert extract_rules(msgs) == []


def test_preference_stops_at_trailing_conjunction():
    cands = extract_rules(_user("I love Python and I use it daily"))
    prefs = [c for c in cands if c.type is MemoryType.preference]
    assert prefs, "expected at least one preference"
    assert all("and" not in c.key for c in prefs)
    assert any(c.value == "likes Python" for c in prefs)
