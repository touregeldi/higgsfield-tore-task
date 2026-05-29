# Higgsfield AI Engineering Challenge

# 🧠 Build a Memory Service for an AI Agent

> **Role:** AI Engineer (mid-level)
>
> **Time-box:** 2 days of focused work
>
> **Deliverable:** A reproducible Git repository with a Dockerized memory service that conforms to the HTTP contract below.
>
> **Submission:** https://higgsfieldcareers.typeform.com/to/HoCFpdJC

---

## 1. Overview

We're hiring an AI engineer. The task: design and build a **memory service** for an AI agent. The service ingests conversation turns, persists them, extracts structured knowledge, and answers recall queries that decide what context the agent sees on the next turn.

You build the service. We host it ourselves and run a private eval against it. You have full freedom over backing store, recall pipeline, extraction strategy, and language.

---

## 2. Your Task

Ship a single Docker-deployable memory service that:

1. Conforms to the HTTP contract in §3.
2. Persists data across container restarts via a Docker volume.
3. Comes up with `docker compose up` — no manual setup steps.
4. Has its own internal tests (including at least a small recall-quality fixture — see §7).
5. Ships a `CHANGELOG.md` documenting your iteration history (see §6).
6. Has a `README.md` explaining the architecture, the backing store choice, and the recall strategy.

You have full design freedom on:

- **Language / framework:** Python, Go, Rust, TypeScript — anything Docker can run.
- **Backing store:** Postgres + pgvector, SQLite + FTS, Qdrant, Redis, Mongo, flat files with a clever index — defend it.
- **Extraction pipeline:** How you turn raw conversation turns into structured, queryable knowledge. LLM-based (any provider — OpenAI, Anthropic, local via Ollama, etc.), rule-based, NLP, hybrid — your call. Raw-message-in-vector-DB-out is not extraction.
- **Recall pipeline:** embeddings, BM25, hybrid, graph traversal, rerankers, query rewriting. **Vanilla cosine-top-k will not score well.**
- **Internal architecture:** monolith, multi-service — your call.

---

## 3. The HTTP Contract

Your service must expose these endpoints. Auth is via an optional `Authorization: Bearer <token>` header.

### `GET /health`
Liveness/readiness probe. Returns 200 when ready.

### `POST /turns`
Write a completed conversation turn. Service persists, runs extraction, and returns when done. Eval harness uses a **60-second timeout**.

Request:
```
{
  "session_id":"string",
  "user_id":"string | null",
  "messages": [
    { "role":"user", "content":"string" },
    { "role":"assistant", "content":"string" },
    { "role":"tool", "name":"string | null", "content":"string" }
  ],
  "timestamp":"ISO-8601 string",
  "metadata": { "...":"..." }
}
```
Response: `201 Created`, body `{ "id": "string" }`. Turn and extracted memories must be queryable via `/recall` immediately after this returns.

### `POST /recall`
**Primary signal — most of the eval scores this endpoint.**

Request:
```
{
  "query":"string",
  "session_id":"string",
  "user_id":"string | null",
  "max_tokens":1024
}
```
Response:
```
{
  "context":"string",
  "citations": [
    { "turn_id":"string", "score":0.0, "snippet":"string" }
  ]
}
```
- `context` is formatted text injected into agent's prompt. Make it readable to a frozen LLM.
- Respect `max_tokens` (approximate is fine).
- When budget is tight, prioritize: stable user facts first, then query-relevant memories, then recent context.
- Returns 200 with empty results on cold sessions — never error.

### `POST /search`
Structured search.
```
Request: { "query","session_id","user_id","limit":10 }
Response: { "results":[ {"content","score","session_id","timestamp","metadata"} ] }
```

### `GET /users/{user_id}/memories`
All stored memories for a user. Used for debugging/inspection.
```
{
  "memories": [
    {
      "id","type":"fact|preference|opinion|event",
      "key","value","confidence",
      "source_session","source_turn",
      "created_at","updated_at",
      "supersedes":"string|null","active":true
    }
  ]
}
```

### `DELETE /sessions/{session_id}`
204 No Content.

### `DELETE /users/{user_id}`
204 No Content. Deletes all data for a user.

---

## 4. The Hard Problems

### Fact evolution and contradiction handling
- Detect same-topic facts (employment, location, etc).
- Store new as active, mark old as superseded (not deleted).
- Return current from `/recall`.
- Preserve history.
- Harder variant: opinion arcs ("I love TS" → "TS generics annoying" → "TS for big projects, Python for scripts").

### Extraction, not just storage
At minimum extract:
- Personal facts (employment, location, family, pets)
- Preferences and opinions
- Corrections
- Implicit facts ("walking Biscuit" → has dog named Biscuit)

`/users/{user_id}/memories` returning raw chunks = red flag.

### Context assembly under budget
Defend the priority logic in README.

---

## 5. Hard Constraints
- **Persistence** across `docker compose down/up` via named volume.
- **Concurrent sessions** don't bleed unless intentionally shared for same `user_id`.
- **Synchronous correctness** — after `POST /turns` returns, data is readable.
- **Recall budget** reasonable latency.
- **Resilience** to malformed input, oversized payloads, unicode.
- **LLM usage encouraged.** Document in README. Keys via `.env.example`.

---

## 6. Submission Format

```
memory-service/
├── README.md
├── CHANGELOG.md
├── docker-compose.yml
├── Dockerfile
├── src/
├── tests/
├── fixtures/
└── .env.example
```

### README must contain
1. Architecture (diagram + 1-2 paragraphs)
2. Backing store choice
3. Extraction pipeline
4. Recall strategy
5. Fact evolution
6. Tradeoffs
7. Failure modes
8. How to run tests

### CHANGELOG (most important deliverable)
One entry per significant design iteration. Show what you tried, what you observed, why you changed. Example: v3 hybrid retrieval with RRF, etc. Mediocre score + thoughtful 5-entry CHANGELOG > higher score + no iteration history.

---

## 7. Testing and Self-Eval

### Required: contract tests
- Contract roundtrip
- Restart persistence
- Concurrent sessions
- Malformed input

### Required: recall quality fixture
Small fixture in `fixtures/` (3-5 scripted conversations + probe queries with expected facts). Build early, run after every change.

### Provided smoke test
Standard curl-based health/turns/recall/memories flow.

---

## 8. Setup We'll Use
```
git clone <repo> memory-service
cd memory-service
docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done
```
Default port 8080. No manual setup. Keys via `.env.example`.

---

## 9. How It Will Be Tested

### A. Automated private eval — categories:
- **Recall quality** (primary signal)
- **Fact evolution** — current fact returned, history preserved, supersession chain
- **Multi-hop recall** — "what city does the user with the dog named Biscuit live in?"
- **Noise resistance** — empty context on unrelated queries, no hallucinations
- **Extraction quality** — structured, typed, implicit facts, corrections
- **Persistence across restarts**
- **Cross-session scoping**
- **Robustness** — 4xx/5xx not crashes
- **Correctness** — no eventual consistency gaps
- **Contract compliance**

### B. Human architecture review
- Sound architecture, justified backing store
- Real extraction (not chunks)
- Thoughtful recall (not vanilla cosine top-k)
- Genuine CHANGELOG iteration
- Clean tested code
- Extensibility

30-minute follow-up interview to defend choices.

---

## 10. What "Excellent" Looks Like

- Exact contract compliance
- Structured memories with types, confidence, provenance
- Fact evolution: contradictions detected, supersession, history preserved
- Real ranking (hybrid, graph, multi-hop, query rewriting)
- Explicit priority logic under token budget, defended in README
- Synchronous `/turns`
- Token-budgeted `/recall`
- Real persistence
- Graceful degradation
- Tests: contract, restart, concurrency, malformed, recall quality
- CHANGELOG 4+ entries with metrics
- README walks reviewer in 5 min
- Clean inspectable `/users/{user_id}/memories`

---

## 11. Originality Rule
Read mem0, hindsight, honcho, mnemonic agents for inspiration. Don't lift API shape or recall pipeline. Be ready to defend resemblance line-by-line.

---

## 12. Out of Scope
- No agent-side code
- No UI
- No multi-tenant production
- No horizontal scalability proofs
- No migration story
