# Changelog

Each entry is a design iteration: what was tried, what the recall-quality fixture
(`fixtures/recall_quality.json`, 5 probe queries spanning fact-evolution, multi-hop, and
noise-resistance) showed, and why the next change was made. Score = fraction of probes whose
expected fact appears (or, for the noise probe, is correctly absent) in `/recall` context.

## v0.1 — Rule-only extraction + vector-only recall (baseline)

First cut: regex extraction of location/employment/pets into a typed `memories` table, a
local MiniLM embedder, and recall = pgvector cosine top-k only. No fact evolution, no lexical
channel, no rerank.

- **Fixture: ~0.4.** Observations:
  - "where does alice live now" returned *Berlin* (the first value), not *Munich* — there was
    no notion of a fact being superseded.
  - The multi-hop probe ("city of the owner of dog Biscuit") missed: cosine alone didn't bridge
    the entity to the location fact.
  - Noise probe was fine (nothing relevant stored).

## v0.2 — Fact evolution via supersession chains

Added `(user_id, type, key)` keying and a reconciliation step: a contradicting same-key fact
inserts a new active row pointing `supersedes` at the old one, which is flipped inactive — in
one transaction. History preserved. Required teaching the location pattern the past tense
("moved to", not just "moving to") so "I moved to Munich" actually produced a candidate.

- **Fixture: ~0.6.** "live now" → *Munich*, with Berlin retained as history. Multi-hop still
  weak.

## v0.3 — Hybrid retrieval + Reciprocal Rank Fusion

Added a Postgres FTS (`tsvector` + GIN, `ts_rank`) channel alongside vector search and fused
the two ranked lists with RRF. Lexical matching anchored exact tokens (names like "Biscuit",
"Stripe") that pure cosine ranked inconsistently.

- **Fixture: ~0.8.** The dog-name probe became reliable. Remaining miss: the multi-hop probe
  still depended on the *location* fact being retrievable from a query phrased around the
  *dog*.

## v0.4 — Query rewrite (entity bridging) + cross-encoder rerank + budgeted assembly

- **Query rewrite:** extract proper-noun entities from the query and emit entity-focused
  variants, so a dog-centric question also retrieves on the dog's name; the always-surfaced
  "stable user facts" tier then carries the linked location. This is the lightweight
  multi-hop bridge (in lieu of a full graph).
- **Cross-encoder rerank:** rescced the fused candidates for precision; non-positive scores
  dropped, sharpening noise-resistance.
- **Budgeted assembly:** explicit priority — stable facts → query-relevant → recent — filling
  `max_tokens`, with strict cross-tier priority (a truncated higher tier blocks lower tiers).

- **Fixture: 1.00 (5/5).** All probes pass, including multi-hop and the noise probe (empty
  context, no fabrication).

## v0.5 — Hardening from review & tests

Issues caught by code review and integration tests, fixed before release:

- **Greedy extraction capture** ("I work at Google now" → value "Google now"): added a
  trailing-stop lookahead with non-greedy capture for all fact patterns. Later extended the
  same guard to **preference** patterns ("I love Python and JavaScript" was becoming one junk
  key).
- **Assembler priority leak:** the budget loop used `continue`, letting a small recent-context
  item slip in after a higher-priority memory was dropped. Switched to strict cross-tier
  priority (truncation in a tier stops lower tiers).
- **Honest extraction gap, documented not hidden:** "I'm a teacher" isn't matched by the
  employment patterns, so "what does bob do" is answered from the **recent-context tier**
  rather than a structured fact. Captured in the README failure modes; the optional LLM
  extractor closes this gap when enabled. Kept the fixture honest rather than adding a
  bespoke pattern to inflate the score.

## v0.6 — Live-stack verification (caught what the fakes hid)

Running the real `docker compose` stack (real MiniLM + cross-encoder, not the in-test fakes)
surfaced two issues unit/integration tests structurally could not:

- **`Numpy is not available` in `/turns` and `/recall`.** torch 2.2.2's `tensor.numpy()` breaks
  against numpy 2.x; an unpinned numpy got resolved to 2.x in the image. Pinned `numpy==1.26.4`.
  Invisible to the test suite because tests inject `FakeEmbedder`/`FakeReranker` — only the
  containerized `STEmbedder` exercises torch.
- **Dirty fact value `"Munich last month"`.** The fixture's substring check ("Munich" ∈ value)
  masked it; the live `/memories` view exposed it. Extended the trailing-stop guard with
  temporal/filler words so the stored value is `"Munich"`.

- **Release state: 35 unit + 24 integration passing.** Coverage: contract roundtrip, restart
  persistence (verified live — write, `docker compose restart`, data survives the named
  volume), concurrent-session isolation, malformed/oversized/unicode input, recall-quality
  fixture. **Recall-quality score 1.00 (5/5)** measured both in-test (fakes) and against the
  live real-model stack.

## Possible next iterations

- Promote the entity bridge to a real entity table for genuine multi-hop joins.
- LLM-assisted reconciliation for opinion *arcs* ("love TS" → "TS generics annoy me" → "TS for
  big projects") rather than hard supersession.
- Confidence decay over time for stale facts.
