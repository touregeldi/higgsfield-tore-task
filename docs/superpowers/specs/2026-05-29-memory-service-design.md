# Memory Service — Design Spec

**Date:** 2026-05-29
**Context:** Higgsfield AI Engineering Challenge (see `CHALLENGE.md`). Build a Dockerized
memory service for an AI agent: ingests conversation turns, extracts structured knowledge,
answers recall queries. Hosted and eval'd privately by Higgsfield against the HTTP contract.

## 1. Goals & Scope

**In scope:**
- Exact compliance with the HTTP contract (§3 of `CHALLENGE.md`).
- Structured extraction of typed memories (fact / preference / opinion / event) with
  confidence and provenance — not raw chunks.
- Fact evolution: contradiction detection, supersession chains, history preservation.
- Hybrid recall pipeline scored hardest by the eval: vector + lexical + RRF + rerank +
  query rewriting, with explicit token-budgeted context assembly.
- Real persistence across `docker compose down/up` via a named volume.
- Synchronous `/turns` — data queryable the instant the call returns.
- Internal tests incl. a recall-quality fixture; a CHANGELOG with iteration metrics.

**Out of scope (per challenge §12):** agent-side code, UI, multi-tenant production
hardening, horizontal-scale proofs, schema migrations.

## 2. Foundational Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language / framework | **Python + FastAPI** | Richest LLM/embedding ecosystem, async HTTP, fastest to build in the 2-day box. |
| Backing store | **Postgres + pgvector** | One store for the relational fact-graph (supersession chains), vector search, and SQL/FTS filtering. Defensible, hybrid-friendly. |
| Extraction | **Hybrid** — rule-based core + optional LLM enhancer | Deterministic extraction always works (offline-safe for the eval); LLM improves quality when a key is present. We don't control the eval's keys. |
| Embeddings | **Local `sentence-transformers` model (MiniLM, ~80MB) baked into the image** | Vector recall must work without any external key. Always-on, deterministic backbone. |
| LLM | **Optional enhancer only** | No-ops gracefully without a key; bounded + timeout-guarded so `/turns` never exceeds the 60s eval timeout. |
| Recall | **Hybrid + RRF + rerank + query-rewrite** | Vanilla cosine-top-k explicitly won't score. Covers multi-hop and noise-resistance eval categories. |
| Reranker | **Small local cross-encoder baked into image (~100MB)** | Keeps rerank quality deterministic and offline. (Alt considered: LLM-only rerank with heuristic fallback — rejected to avoid key dependence on the scored path.) |
| Fact model | **Typed memories keyed by `(user_id, type, key)` with supersession chain** | Directly satisfies the `/memories` contract shape and the fact-evolution eval. |
| Multi-hop | **Entity-linking within the fact layer** (not a full graph) | Lighter to build; resolves cases like "city of the user with dog Biscuit" by linking the entity to the owner's `location` fact. |

## 3. Architecture

Monolith, layered (`routes → services → repositories`), single Postgres with pgvector.
Each layer is independently testable. Two enrichment dependencies: a local embedding model
(always on) and an optional LLM client (no-ops without a key).

```
                    ┌─────────────────────────────────────────┐
   HTTP (8080)      │             FastAPI app                  │
 ───────────────►   │  routes → services → repositories        │
                    │                 │                        │
                    │           ┌─────┴─────┐                  │
                    │           │Extraction │ (rule + opt LLM) │
                    │           └─────┬─────┘                  │
                    │           ┌─────┴─────┐                  │
                    │           │  Recall   │ (hybrid+RRF+rerank)
                    │           └─────┬─────┘                  │
                    │   local embed model + optional LLM client│
                    └─────────────────┬────────────────────────┘
                                      │
                            ┌─────────┴──────────┐
                            │ Postgres+pgvector  │  ◄── named volume
                            │ turns | memories   │
                            │ + FTS + vector idx │
                            └────────────────────┘
```

## 4. Data Model (Postgres)

**`turns`** — provenance anchor for citations.
- `id` (text/uuid pk), `session_id`, `user_id` (nullable), `messages` (jsonb),
  `timestamp` (timestamptz), `metadata` (jsonb), `created_at`.

**`memories`** — the structured knowledge layer.
- `id` (pk), `user_id`, `session_id`, `type` (`fact|preference|opinion|event`),
  `key` (normalized topic, e.g. `employment`, `location`, `pet.name`), `value` (text),
  `confidence` (float), `source_session`, `source_turn` (fk → turns.id),
  `created_at`, `updated_at`, `supersedes` (fk → memories.id, nullable),
  `active` (bool), `embedding` (`vector`), `fts` (`tsvector`, generated).

**Indexes:** ivfflat/hnsw on `embedding`; GIN on `fts`; btree on
`(user_id, type, key, active)` and `(session_id)`.

**Fact-evolution invariant:** at most one `active=true` row per `(user_id, type, key)`.
New same-key fact → previous row set `active=false`, new row `supersedes = old.id`.
History never deleted. `/recall` and `/memories` filter `active=true` by default;
the chain is walkable for history.

## 5. `/turns` Flow (synchronous, <60s)

1. Validate (Pydantic) and persist the raw turn → immediately citable.
2. **Extraction** → candidate typed memories:
   - **Rule layer (always):** regex/heuristic patterns for the named categories —
     employment ("I work at/as"), location ("I live in"), family/pets including *implicit*
     ("walking Biscuit" → `pet.name=Biscuit`), preferences/opinions ("I love/hate/prefer"),
     corrections ("actually", "no longer"). spaCy NER for entities.
   - **LLM layer (if key present):** one structured-output call returning the same typed
     schema; merged with rule output (more-specific / higher-confidence wins). Bounded +
     timeout-guarded; any failure falls back to rule-only output.
3. **Reconcile** each candidate vs existing active memories on `(user, type, key)` →
   insert / supersede / skip-duplicate.
4. Embed + FTS-index new memories. Commit. Return `201 {id}`.
   No async, no eventual consistency — everything queryable on return.

## 6. `/recall` Flow (primary scored path)

1. **Query rewrite** — heuristic expansion/decomposition always; LLM if key. Multi-hop
   (e.g. "city of the user with dog Biscuit") pulls the entity and fetches the linked
   owner's `location` fact from the structured layer.
2. **Retrieve two ways** — pgvector cosine top-N + Postgres FTS (`ts_rank`, BM25-style)
   top-N, scoped by `session_id` / `user_id`.
3. **Fuse** via Reciprocal Rank Fusion (RRF).
4. **Rerank** fused candidates with the local cross-encoder.
5. **Assemble context under `max_tokens`** with explicit priority:
   **(1) stable active user facts → (2) query-relevant reranked memories →
   (3) recent turn context**, trimming from the bottom. Output: readable formatted text
   for a frozen LLM + `citations[{turn_id, score, snippet}]`.
6. Cold / unrelated query → `200` with empty results. Never error, never hallucinate.

## 7. Other Endpoints
- `/search` — structured hits (`content, score, session_id, timestamp, metadata`), no token budgeting.
- `GET /users/{user_id}/memories` — clean typed rows matching the contract shape
  (active by default; superseded included when requested).
- `DELETE /sessions/{session_id}` → cascade, `204`.
- `DELETE /users/{user_id}` → delete all user data, `204`.
- `GET /health` → `200` when DB reachable.

## 8. Resilience & Edge Cases
- Pydantic validation → `422` on malformed bodies.
- Payload size cap → `413` on oversized.
- Unicode-safe throughout.
- LLM / embedding failures degrade (fall back), never crash → sensible 5xx only on true faults.
- Cross-session isolation enforced in every query; `user_id` data shared only for the same user.

## 9. Testing (`tests/` + `fixtures/`)
- **Contract roundtrip** — every endpoint, shapes, status codes.
- **Restart persistence** — `compose down/up`, data survives the named volume.
- **Concurrent sessions** — no cross-session bleed.
- **Malformed / oversized / unicode input** — correct 4xx, no crash.
- **Recall-quality fixture** — 3–5 scripted conversations + probe queries with expected
  facts, covering fact-evolution, multi-hop, and noise-resistance. Run after every change;
  CHANGELOG records the metric per iteration.

## 10. Deliverables Layout
```
memory-service/
├── README.md          # architecture, store choice, extraction, recall, fact evolution, tradeoffs, failure modes, how to test
├── CHANGELOG.md       # one entry per significant iteration, with recall-fixture metrics
├── docker-compose.yml # app + postgres+pgvector, named volume, port 8080
├── Dockerfile
├── src/
├── tests/
├── fixtures/
└── .env.example       # optional LLM key, DB config
```

## 11. Tradeoffs (for the README, surfaced here)
- **Local models in-image (size) vs key-independence (reliability):** chose reliability —
  the scored path must work offline.
- **Entity-linking vs full graph for multi-hop:** chose lighter entity-linking to fit the
  time box; full graph is the documented extension path.
- **Single Postgres vs split vector/metadata stores:** chose one store to keep
  fact-evolution logic transactional and in one place.
