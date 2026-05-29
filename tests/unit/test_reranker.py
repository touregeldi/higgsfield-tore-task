from memory_service.recall.reranker import FakeReranker


def test_fake_reranker_scores_lexical_overlap():
    r = FakeReranker()
    scores = r.rerank("where do I live", ["user lives in Berlin", "user likes pizza"])
    assert len(scores) == 2
    assert scores[0] > scores[1]  # first doc shares 'live/lives' token
