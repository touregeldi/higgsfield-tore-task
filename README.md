# Memory Service

A memory service for an AI agent. It ingests completed conversation turns, extracts
**typed, structured memories** with confidence and provenance, evolves facts over time via
**supersession chains**, and answers **token-budgeted recall** queries with a hybrid
retrieval pipeline (vector + lexical + Reciprocal Rank Fusion + cross-encoder rerank +
query rewriting).

It runs entirely from `docker compose up` with no manual setup, persists across restarts via
a named volume, and is **fully functional offline** — embeddings and reranking use local
models baked into the image; a hosted LLM is an optional quality enhancer, never a hard
dependency.

## Quick start

```bash
cp .env.example .env          # optional: set OPENAI_API_KEY to enable the LLM enhancer
docker compose up -d --build
until curl -sf http://localhost:8080/health; do sleep 2; done

# write a turn (synchronous: extraction completes before it returns)
curl -s -X POST http://localhost:8080/turns -H 'content-type: application/json' -d '{
  "session_id":"s1","user_id":"u1",
  "messages":[{"role":"user","content":"I live in Berlin and I work at Stripe"}],
  "timestamp":"2026-05-29T00:00:00Z","metadata":{}}'

# recall (the primary, scored endpoint)
curl -s -X POST http://localhost:8080/recall -H 'content-type: application/json' -d '{
  "query":"where do I live","session_id":"s1","user_id":"u1","max_tokens":1024}'

# inspect the structured memory store
curl -s http://localhost:8080/users/u1/memories
```

## 1. Architecture

FastAPI monolith, layered `routes → services → repositories`, over a single
Postgres + pgvector store. Each layer has one responsibility and is tested independently.
Two enrichment dependencies are injected at startup: a local **embedder** (always on) and an
optional **LLM client** (no-ops without a key).

```
                    ┌─────────────────────────────────────────────┐
   HTTP (8080)      │                 FastAPI app                  │
 ───────────────►   │   routes ──► services ──► repositories       │
                    │                │                             │
                    │          ┌─────┴──────┐                      │
                    │          │ Extraction │  rules + optional LLM │
                    │          └─────┬──────┘                      │
                    │          ┌─────┴──────┐                      │
                    │          │   Recall   │  rewrite→hybrid→RRF→  │
                    │          └─────┬──────┘  rerank→budget assemble│
                    │   local embedder + cross-encoder + opt LLM    │
                    └─────────────────┬────────────────────────────┘
                                      │
                            ┌─────────┴──────────┐
                            │  Postgres+pgvector │  ◄── named volume (persistence)
                            │  turns | memories  │      vector ANN + GIN FTS + btree
                            └────────────────────┘
```

**Request flow.** `POST /turns` persists the raw turn (immediately citable), runs extraction,
reconciles candidates against existing facts, embeds + indexes new memories, and commits —
all synchronously, so data is queryable the instant the call returns. `POST /recall` rewrites
the query, retrieves via two independent channels, fuses, reranks, and assembles a
budgeted context string with citations.

## 2. Backing store — Postgres + pgvector

One transactional store handles all three access patterns the problem needs:

- **Relational fact graph** — the `memories` table with a self-referential `supersedes`
  foreign key models fact-evolution chains; supersession + insert happen in one transaction,
  so there is never a window where two "active" values for a topic coexist.
- **Vector ANN** — a `vector(384)` column with an `ivfflat` cosine index for semantic recall.
- **Lexical search** — a generated `tsvector` column with a GIN index for BM25-style FTS.

Keeping these together (rather than splitting a vector DB from a metadata DB) means the
fact-evolution logic stays consistent and the hybrid retrieval reads from one source of
truth. It's also trivially reproducible: the official `pgvector/pgvector` image plus a
single `schema.sql` applied idempotently on startup.

## 3. Extraction pipeline — hybrid (rules + optional LLM)

Raw text in, **typed candidates** out: each memory has a `type`
(`fact | preference | opinion | event`), a normalized `key` (e.g. `location`, `employment`,
`pet.name:biscuit`), a `value`, a `confidence`, and provenance (`source_turn`,
`source_session`).

- **Rule layer (always on, deterministic).** Regex patterns over *user* messages capture
  employment, location, pets — including **implicit** facts ("walking Biscuit" →
  `pet.name:biscuit = Biscuit`) — and preferences/opinions. Captures are bounded by a
  trailing-stop lookahead so "I love Python and ..." doesn't swallow the rest of the
  sentence. Python's unicode-aware `\w` means CJK values round-trip cleanly.
- **LLM layer (optional).** When `OPENAI_API_KEY` is set, one structured-output call returns
  the same typed schema and is merged with the rule output (higher confidence wins per key).
  The call is timeout-guarded and wrapped so **any failure degrades to rule-only output** —
  it can never crash `/turns` or blow the 60s budget.

`/users/{user_id}/memories` returns these structured rows, not raw chunks.

## 4. Recall strategy

`/recall` is deliberately *not* vanilla cosine-top-k:

1. **Query rewrite** — heuristic always (extracts proper-noun entities and appends
   entity-bridging variants so a query like *"city of the user with the dog Biscuit"* reaches
   facts linked to that entity); LLM paraphrases added when a key is present.
2. **Two-channel retrieval** — pgvector cosine **and** Postgres FTS, each over all rewrite
   variants, scoped by `session_id`/`user_id`.
3. **Reciprocal Rank Fusion** — combines the channels without needing comparable score
   scales; items ranked high in multiple lists rise to the top.
4. **Cross-encoder rerank** — a local cross-encoder rescores the fused candidates against the
   original query for precision; non-positive scores are dropped (noise resistance).
5. **Budgeted assembly** — see below.

## 5. Fact evolution

Memories are keyed by `(user_id, type, key)`. On a new turn, each candidate is reconciled
against the current active memory for its key:

- **New key** → insert active.
- **Same key, same value** → skip (no churn).
- **Same key, different value** → insert the new value as active **and** set its
  `supersedes` pointer to the old row, which is flipped `active = false`.

History is never deleted. `/recall` and `/memories` show active memories by default; the full
supersession chain is preserved and inspectable (`?include_superseded=true`). This is what
turns "I live in Berlin" → "I moved to Munich" into a current answer of *Munich* with Berlin
retained as history.

## 6. Context assembly under budget

`assemble_context` fills the `max_tokens` budget in strict priority order:

1. **Stable user facts** — durable identity (location, employment, pets).
2. **Query-relevant memories** — the reranked hits for this query.
3. **Recent context** — summaries of the latest turns in the session.

Within a tier we greedily pack items that fit; once a tier is **truncated** for budget, lower
tiers are not started — so a low-value recent snippet can never displace a higher-priority
fact. The output is readable, section-labelled text aimed at a frozen LLM, plus
`citations[{turn_id, score, snippet}]`. Token counting is an approximate word/punctuation
counter (the contract permits approximation) which keeps the hot path dependency-free.

## 7. Tradeoffs

- **Local models baked into the image (~larger image) vs. key-independence (reliability).**
  Chose reliability: the scored recall path must work with no external key and no network.
- **Entity-linking in the fact layer vs. a full knowledge graph for multi-hop.** Chose the
  lighter approach to fit the time box; the query-rewrite entity bridge plus always-surfaced
  user facts resolves the eval's multi-hop cases. A full graph is the documented extension
  path.
- **Regex extraction vs. spaCy/NER.** Chose regex: leaner image, and the capture groups
  double as entities. The cost is recall on phrasings the patterns don't cover (see Failure
  modes) — exactly where the optional LLM layer earns its keep.
- **Approximate token counter vs. tiktoken.** tiktoken downloads encoding files on first use,
  which would break the offline guarantee; the approximation is sufficient for budgeting.
- **Single Postgres vs. dedicated vector store.** Chose one store for transactional
  fact-evolution and a single source of truth over best-in-class ANN at this scale.

## 8. Failure modes

- **No LLM key** → rule-only extraction and heuristic query rewrite; everything still works.
- **LLM / embedding / rerank error** → caught and degraded (rule-only, or skip rerank); the
  request never crashes.
- **Cold or unrelated query** → `/recall` returns `200` with empty context and no citations;
  no hallucination.
- **Malformed body** → `422` (Pydantic). **Oversized body** → `413` (size middleware).
  **Unicode** → preserved end-to-end.
- **DB unreachable** → `/health` returns `503`.
- **Known extraction gap** → phrasings outside the rule patterns (e.g. "I'm a teacher") aren't
  captured as structured facts; recall still surfaces them via the recent-context tier, and
  the LLM layer captures them when enabled.

### Known limitations (deliberately out of scope)

These are documented rather than fixed, because they only bite outside the stated target of
"single user, a few concurrent sessions":

- **Concurrent writes to the same `(user_id, key)`.** `/turns` reads the current active
  memory, then supersedes + inserts in one transaction. Two *simultaneous* `POST /turns` for
  the **same user and same key** could each supersede the other's predecessor, briefly leaving
  two active rows. Sequential ingestion (the eval pattern) is always correct. The fix is a
  `SELECT … FOR UPDATE` on the key or a `UNIQUE (user_id, key) WHERE active` partial index;
  omitted to avoid turning a rare race into hard request failures at this scale.
- **Sync handlers on a threadpool.** Route handlers are `def` (FastAPI runs them in a
  threadpool) and use the synchronous psycopg pool. Fine for a handful of concurrent sessions;
  a high-concurrency deployment would move to `async def` + an async pool.

## 9. Running the tests

Unit tests need no database:

```bash
pip install -r requirements.txt    # or just: pytest pydantic pydantic-settings
pytest tests/unit -v
```

Integration tests need a pgvector-enabled Postgres. Point `DATABASE_URL` at one (the compose
`db` works, or any local pgvector Postgres):

```bash
docker compose up -d db
DATABASE_URL=postgresql://memory:memory@localhost:5432/memory pytest tests/integration -v
```

The recall-quality fixture (`fixtures/recall_quality.json`, 2 scripted conversations + 5
probe queries covering fact-evolution, multi-hop, and noise-resistance) runs as
`tests/integration/test_recall_quality.py` and prints a `RECALL_QUALITY_SCORE`. The
restart-persistence test activates automatically when a live stack is up on `:8080`.

## 10. Layout

```
memory-service/
├── README.md  CHANGELOG.md  docker-compose.yml  Dockerfile  .env.example
├── scripts/prefetch_models.py     # bakes embed + rerank models at build time
├── src/memory_service/
│   ├── routes/        # one module per endpoint group
│   ├── services/      # ingest, recall, reconcile (fact evolution)
│   ├── repositories/  # turns, memories (SQL + supersession + hybrid search)
│   ├── extraction/    # keys, rules, llm_extractor, pipeline
│   ├── recall/        # embedder, reranker, fusion, query_rewrite, assembler
│   ├── llm/  models/  db/  config.py  tokens.py  app.py  main.py
├── tests/unit/  tests/integration/
└── fixtures/recall_quality.json
```

## Notes on scope

Single user / a few concurrent sessions is the target (per the brief). The design happens to
be horizontally scalable for reads (stateless app, all state in Postgres) but nothing is
built specifically for scale. Single schema version, applied idempotently; no migration story.
