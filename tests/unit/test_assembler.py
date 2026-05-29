from memory_service.tokens import count_tokens
from memory_service.recall.assembler import assemble_context, ContextItem


def test_count_tokens_monotonic():
    assert count_tokens("a b c d") < count_tokens("a b c d e f g h i j")


def test_assembler_prioritizes_facts_under_budget():
    facts = [ContextItem(text="User lives in Berlin.", turn_id="t1", score=1.0)]
    relevant = [ContextItem(text="User likes pizza.", turn_id="t2", score=0.5)]
    recent = [ContextItem(text="Long recent chatter " * 50, turn_id="t3", score=0.1)]
    ctx, cites = assemble_context(facts, relevant, recent, max_tokens=20)
    assert "Berlin" in ctx                 # stable fact kept first
    assert "chatter" not in ctx            # low-priority recent trimmed
    assert cites[0].turn_id == "t1"


def test_assembler_empty_inputs():
    ctx, cites = assemble_context([], [], [], max_tokens=100)
    assert ctx == "" and cites == []


def test_assembler_skips_lower_tier_after_truncation():
    facts = []
    relevant = [ContextItem(text="word " * 40, turn_id="r1", score=0.9)]  # too big to fit
    recent = [ContextItem(text="tiny", turn_id="rec1", score=0.1)]
    ctx, cites = assemble_context(facts, relevant, recent, max_tokens=10)
    assert "tiny" not in ctx               # recent NOT added: a higher tier was truncated
    assert all(c.turn_id != "rec1" for c in cites)
