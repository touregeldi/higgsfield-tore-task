from memory_service.recall.embedder import FakeEmbedder


def test_fake_embedder_dim_and_determinism():
    e = FakeEmbedder(dim=8)
    a = e.embed(["hello", "world"])
    b = e.embed(["hello"])
    assert len(a) == 2 and len(a[0]) == 8
    assert a[0] == b[0]  # deterministic for same text
