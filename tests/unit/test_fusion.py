from memory_service.recall.fusion import reciprocal_rank_fusion


def test_rrf_rewards_items_high_in_both_lists():
    vec = ["a", "b", "c"]
    lex = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([vec, lex])
    ranked = [k for k, _ in fused]
    assert ranked[0] in ("a", "b")
    assert set(ranked) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
