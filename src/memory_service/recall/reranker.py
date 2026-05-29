from __future__ import annotations
import re
from typing import Protocol

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return {t[:4] for t in _TOK.findall(s.lower())}  # crude stemming via prefix


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class FakeReranker:
    """Token-overlap reranker for unit tests (no model download)."""

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        q = _tokens(query)
        out = []
        for d in docs:
            dt = _tokens(d)
            out.append(len(q & dt) / (len(q) or 1))
        return out


class CEReranker:
    """Local cross-encoder reranker. Model is baked into the image."""

    def __init__(self, model_id: str):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_id)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        scores = self._model.predict([(query, d) for d in docs])
        return [float(s) for s in scores]
