from memory_service.recall.query_rewrite import rewrite_query
from memory_service.llm.client import NullLLMClient


def test_heuristic_includes_original_and_entities():
    out = rewrite_query("what city does the user with the dog named Biscuit live in?", NullLLMClient())
    assert "what city does the user with the dog named Biscuit live in?" in out.variants
    assert "Biscuit" in out.entities


def test_variants_are_unique():
    out = rewrite_query("where do I live", NullLLMClient())
    assert len(out.variants) == len(set(out.variants))
