from __future__ import annotations
from ..repositories.memories import MemoryRepository
from ..repositories.turns import TurnRepository
from ..recall.embedder import Embedder
from ..recall.reranker import Reranker
from ..recall.fusion import reciprocal_rank_fusion
from ..recall.query_rewrite import rewrite_query
from ..recall.assembler import assemble_context, ContextItem
from ..llm.client import LLMClient
from ..models.schemas import RecallResponse, SearchHit

_CANDIDATE_LIMIT = 20
_RERANK_TOP = 10


class RecallService:
    def __init__(self, memories: MemoryRepository, turns: TurnRepository,
                 embedder: Embedder, reranker: Reranker, llm: LLMClient):
        self._memories = memories
        self._turns = turns
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm

    def _retrieve(self, query: str, session_id: str, user_id: str | None) -> list[str]:
        rw = rewrite_query(query, self._llm)
        rankings: list[list[str]] = []
        vecs = self._embedder.embed(rw.variants)
        mean = [sum(col) / len(vecs) for col in zip(*vecs)]
        rankings.append(self._memories.search_vector(user_id, session_id, mean, _CANDIDATE_LIMIT))
        for variant in rw.variants:
            rankings.append(self._memories.search_fts(user_id, session_id, variant, _CANDIDATE_LIMIT))
        fused = reciprocal_rank_fusion(rankings)
        return [mid for mid, _ in fused[:_CANDIDATE_LIMIT]]

    def recall(self, query: str, session_id: str, user_id: str | None,
               max_tokens: int) -> RecallResponse:
        candidate_ids = self._retrieve(query, session_id, user_id)
        mem_map = self._memories.get_many(candidate_ids)
        candidates = [mem_map[i] for i in candidate_ids if i in mem_map]

        relevant: list[ContextItem] = []
        if candidates:
            scores = self._reranker.rerank(query, [m.value for m in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)[:_RERANK_TOP]
            relevant = [ContextItem(text=m.value, turn_id=m.source_turn, score=float(s))
                        for m, s in ranked if s > 0]

        facts = [ContextItem(text=f"{m.key}: {m.value}", turn_id=m.source_turn,
                             score=m.confidence)
                 for m in self._memories.active_facts(user_id)]

        recent_turns = self._turns.recent_for_session(session_id, limit=2, user_id=user_id)
        recent = [ContextItem(text=self._summarize(t.messages), turn_id=t.id, score=0.1)
                  for t in recent_turns]

        context, citations = assemble_context(facts, relevant, recent, max_tokens)
        return RecallResponse(context=context, citations=citations)

    def search(self, query: str, session_id: str | None, user_id: str | None,
               limit: int) -> list[SearchHit]:
        sid = session_id or ""
        ids = []
        if query.strip():
            vec = self._embedder.embed([query])[0]
            ids = self._memories.search_vector(user_id, sid, vec, limit) + \
                  self._memories.search_fts(user_id, sid, query, limit)
        mem_map = self._memories.get_many(ids)
        seen, hits = set(), []
        for i in ids:
            if i in seen or i not in mem_map:
                continue
            seen.add(i)
            m = mem_map[i]
            hits.append(SearchHit(content=f"{m.key}: {m.value}", score=m.confidence,
                                  session_id=m.session_id,
                                  timestamp=m.created_at.isoformat(),
                                  metadata={"type": m.type.value, "active": m.active}))
            if len(hits) >= limit:
                break
        return hits

    @staticmethod
    def _summarize(messages: list[dict]) -> str:
        parts = [f"{m.get('role')}: {m.get('content','')}" for m in messages]
        return " | ".join(parts)[:300]
