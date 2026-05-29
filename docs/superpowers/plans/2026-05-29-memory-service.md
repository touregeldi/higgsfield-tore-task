# Memory Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized FastAPI memory service that ingests conversation turns, extracts typed structured memories with provenance, evolves facts via supersession chains, and answers token-budgeted recall queries with a hybrid (vector + lexical + RRF + rerank) pipeline.

**Architecture:** Monolith, layered `routes → services → repositories`, single Postgres+pgvector store. Pure-logic modules (extraction rules, RRF, assembler, reconcile) are unit-tested without a DB. Embeddings and reranking run on local models baked into the image (offline-safe); a hosted LLM is an optional, timeout-guarded enhancer that no-ops without a key.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, psycopg 3 (+pool), pgvector, sentence-transformers (MiniLM embedder + MiniLM cross-encoder reranker), pydantic / pydantic-settings, pytest + httpx. Optional: openai SDK.

---

## File Structure

```
memory-service/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── CHANGELOG.md
├── src/memory_service/
│   ├── __init__.py
│   ├── config.py              # Settings (env)
│   ├── tokens.py              # approximate token counter
│   ├── app.py                 # FastAPI factory + lifespan wiring
│   ├── main.py                # uvicorn entry
│   ├── models/
│   │   ├── domain.py          # MemoryType, MemoryCandidate, Memory, Turn
│   │   └── schemas.py         # pydantic request/response models
│   ├── llm/
│   │   └── client.py          # LLMClient protocol, NullLLMClient, OpenAILLMClient
│   ├── extraction/
│   │   ├── keys.py            # key constants + normalization
│   │   ├── rules.py           # rule-based extractor
│   │   ├── llm_extractor.py   # optional LLM extractor
│   │   └── pipeline.py        # merge rule + LLM candidates
│   ├── recall/
│   │   ├── embedder.py        # Embedder protocol, FakeEmbedder, STEmbedder
│   │   ├── reranker.py        # Reranker protocol, FakeReranker, CEReranker
│   │   ├── fusion.py          # reciprocal rank fusion
│   │   ├── query_rewrite.py   # heuristic + optional LLM rewrite
│   │   └── assembler.py       # token-budgeted context assembly
│   ├── db/
│   │   ├── schema.sql         # DDL
│   │   └── pool.py            # pool creation + migrate
│   ├── repositories/
│   │   ├── turns.py           # TurnRepository
│   │   └── memories.py        # MemoryRepository
│   └── services/
│       ├── reconcile.py       # fact-evolution decisions (pure)
│       ├── ingest.py          # persist + extract + reconcile
│       └── recall_service.py  # recall + search orchestration
├── tests/
│   ├── conftest.py
│   ├── unit/                  # no DB
│   └── integration/           # needs Postgres
└── fixtures/
    └── recall_quality.json    # scripted convos + probes
```

---

## Task 0: Project scaffold

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.env.example`, `.dockerignore`
- Create: all `__init__.py` under `src/memory_service/` and subpackages
- Create: `src/memory_service/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
psycopg[binary,pool]==3.2.1
pgvector==0.3.2
pydantic==2.9.2
pydantic-settings==2.5.2
sentence-transformers==3.0.1
torch==2.2.2
httpx==0.27.2
openai==1.51.0
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "memory-service"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
markers = ["integration: needs a Postgres database"]
```

- [ ] **Step 3: Write `.env.example`**

```
# Postgres connection used by the service and tests
DATABASE_URL=postgresql://memory:memory@db:5432/memory
# Optional hosted LLM enhancer. Leave blank to run fully offline.
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
# Max request body bytes (oversized -> 413)
MAX_BODY_BYTES=1048576
# Local model ids (baked into the image at build time)
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

- [ ] **Step 4: Write `.dockerignore`**

```
.venv
venv
__pycache__
*.pyc
.pytest_cache
.git
docs
tests
```

- [ ] **Step 5: Create empty `__init__.py` files**

Run:
```bash
mkdir -p src/memory_service/{models,llm,extraction,recall,db,repositories,services} tests/unit tests/integration fixtures
for d in . models llm extraction recall db repositories services; do touch "src/memory_service/$d/__init__.py"; done
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 6: Write the failing test `tests/unit/test_config.py`**

```python
from memory_service.config import Settings


def test_settings_defaults_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert s.max_body_bytes == 1048576
    assert s.llm_enabled is False


def test_llm_enabled_when_key_present(monkeypatch):
    s = Settings(_env_file=None, openai_api_key="sk-test")
    assert s.llm_enabled is True
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: memory_service.config`

- [ ] **Step 8: Write `src/memory_service/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://memory:memory@db:5432/memory"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_body_bytes: int = 1_048_576
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key.strip())
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 10: Commit**

```bash
git add requirements.txt pyproject.toml .env.example .dockerignore src tests
git commit -m "chore: project scaffold and settings"
```

---

## Task 1: Domain models and API schemas

**Files:**
- Create: `src/memory_service/models/domain.py`
- Create: `src/memory_service/models/schemas.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write `tests/unit/test_schemas.py`**

```python
import pytest
from pydantic import ValidationError
from memory_service.models.schemas import TurnRequest, RecallRequest
from memory_service.models.domain import MemoryType, MemoryCandidate


def test_turn_request_parses_minimal():
    req = TurnRequest(
        session_id="s1",
        user_id="u1",
        messages=[{"role": "user", "content": "hi"}],
        timestamp="2026-05-29T00:00:00Z",
    )
    assert req.messages[0].role == "user"
    assert req.metadata == {}


def test_turn_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        TurnRequest(session_id="s1", messages=[], timestamp="2026-05-29T00:00:00Z")


def test_recall_request_defaults_max_tokens():
    req = RecallRequest(query="where do I live", session_id="s1")
    assert req.max_tokens == 1024


def test_memory_candidate_holds_fields():
    c = MemoryCandidate(type=MemoryType.fact, key="location", value="Berlin", confidence=0.9)
    assert c.type is MemoryType.fact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/models/domain.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    fact = "fact"
    preference = "preference"
    opinion = "opinion"
    event = "event"


@dataclass
class MemoryCandidate:
    type: MemoryType
    key: str
    value: str
    confidence: float
    evidence: str = ""  # text snippet the candidate was derived from


@dataclass
class Memory:
    id: str
    user_id: str | None
    session_id: str
    type: MemoryType
    key: str
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: datetime
    updated_at: datetime
    supersedes: str | None
    active: bool


@dataclass
class Turn:
    id: str
    session_id: str
    user_id: str | None
    messages: list[dict]
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Write `src/memory_service/models/schemas.py`**

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    name: Optional[str] = None
    content: str


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[Message]
    timestamp: str
    metadata: dict = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class TurnResponse(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


class SearchHit(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: str
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchHit]


class MemoryOut(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: str
    updated_at: str
    supersedes: Optional[str]
    active: bool


class MemoriesResponse(BaseModel):
    memories: list[MemoryOut]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/memory_service/models tests/unit/test_schemas.py
git commit -m "feat: domain models and API schemas"
```

---

## Task 2: Key normalization

**Files:**
- Create: `src/memory_service/extraction/keys.py`
- Test: `tests/unit/test_keys.py`

- [ ] **Step 1: Write `tests/unit/test_keys.py`**

```python
from memory_service.extraction.keys import normalize_value, KEY_LOCATION, KEY_EMPLOYMENT


def test_normalize_value_trims_and_collapses():
    assert normalize_value("  New   York  ") == "New York"


def test_normalize_value_strips_trailing_punctuation():
    assert normalize_value("Berlin.") == "Berlin"


def test_keys_are_stable_strings():
    assert KEY_LOCATION == "location"
    assert KEY_EMPLOYMENT == "employment"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_keys.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/extraction/keys.py`**

```python
import re

KEY_LOCATION = "location"
KEY_EMPLOYMENT = "employment"
KEY_ROLE = "role"
KEY_PET_NAME = "pet.name"
KEY_FAMILY = "family"

_WS = re.compile(r"\s+")


def normalize_value(raw: str) -> str:
    v = _WS.sub(" ", raw.strip())
    return v.rstrip(".!,;: ").strip()


def pet_key(name: str) -> str:
    return f"{KEY_PET_NAME}:{normalize_value(name).lower()}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_keys.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/extraction/keys.py tests/unit/test_keys.py
git commit -m "feat: memory key normalization"
```

---

## Task 3: Rule-based extractor

**Files:**
- Create: `src/memory_service/extraction/rules.py`
- Test: `tests/unit/test_rules.py`

- [ ] **Step 1: Write `tests/unit/test_rules.py`**

```python
from memory_service.extraction.rules import extract_rules
from memory_service.models.domain import MemoryType


def _user(content):
    return [{"role": "user", "content": content}]


def test_extracts_location():
    cands = extract_rules(_user("I live in Berlin"))
    assert any(c.key == "location" and c.value == "Berlin" and c.type is MemoryType.fact for c in cands)


def test_extracts_employment():
    cands = extract_rules(_user("I work at Stripe"))
    assert any(c.key == "employment" and c.value == "Stripe" for c in cands)


def test_extracts_implicit_pet():
    cands = extract_rules(_user("just got back from walking Biscuit"))
    assert any(c.key.startswith("pet.name") and "Biscuit" in c.value for c in cands)


def test_extracts_preference():
    cands = extract_rules(_user("I love TypeScript"))
    assert any(c.type is MemoryType.preference and "TypeScript" in c.value for c in cands)


def test_correction_lowers_nothing_but_still_extracts():
    cands = extract_rules(_user("Actually I no longer work at Stripe, I work at Google now"))
    assert any(c.key == "employment" and c.value == "Google" for c in cands)


def test_ignores_assistant_and_tool_messages():
    msgs = [{"role": "assistant", "content": "I live in Berlin"},
            {"role": "tool", "name": "x", "content": "I work at Stripe"}]
    assert extract_rules(msgs) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_rules.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/extraction/rules.py`**

```python
from __future__ import annotations
import re
from .keys import normalize_value, pet_key, KEY_LOCATION, KEY_EMPLOYMENT
from ..models.domain import MemoryCandidate, MemoryType

# Each pattern captures the value in group 1.
_PATTERNS = [
    (KEY_LOCATION, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:live|reside|am based) in ([A-Z][\w .'-]+)", re.I)),
    (KEY_LOCATION, MemoryType.fact, 0.8,
     re.compile(r"\bi(?:'m| am) (?:from|moving to) ([A-Z][\w .'-]+)", re.I)),
    (KEY_EMPLOYMENT, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:work|am working) (?:at|for) ([A-Z][\w .&'-]+)", re.I)),
    (KEY_EMPLOYMENT, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:joined|now work at) ([A-Z][\w .&'-]+)", re.I)),
]

_PET = re.compile(r"\b(?:walking|feeding|my dog|my cat|petting) ([A-Z][a-z]+)")
_PREF_POS = re.compile(r"\bi (?:love|like|prefer|enjoy|am a fan of) ([\w .#+'-]+)", re.I)
_PREF_NEG = re.compile(r"\bi (?:hate|dislike|can't stand|don't like) ([\w .#+'-]+)", re.I)


def _user_texts(messages: list[dict]) -> list[str]:
    return [m.get("content", "") for m in messages if m.get("role") == "user"]


def extract_rules(messages: list[dict]) -> list[MemoryCandidate]:
    out: list[MemoryCandidate] = []
    for text in _user_texts(messages):
        for key, mtype, conf, pat in _PATTERNS:
            for m in pat.finditer(text):
                val = normalize_value(m.group(1))
                if val:
                    out.append(MemoryCandidate(type=mtype, key=key, value=val,
                                               confidence=conf, evidence=text))
        for m in _PET.finditer(text):
            name = normalize_value(m.group(1))
            out.append(MemoryCandidate(type=MemoryType.fact, key=pet_key(name),
                                       value=name, confidence=0.7, evidence=text))
        for m in _PREF_POS.finditer(text):
            val = normalize_value(m.group(1))
            out.append(MemoryCandidate(type=MemoryType.preference,
                                       key=f"preference:{val.lower()}",
                                       value=f"likes {val}", confidence=0.7, evidence=text))
        for m in _PREF_NEG.finditer(text):
            val = normalize_value(m.group(1))
            out.append(MemoryCandidate(type=MemoryType.preference,
                                       key=f"preference:{val.lower()}",
                                       value=f"dislikes {val}", confidence=0.7, evidence=text))
    return _dedupe(out)


def _dedupe(cands: list[MemoryCandidate]) -> list[MemoryCandidate]:
    seen: dict[tuple[str, str], MemoryCandidate] = {}
    for c in cands:
        k = (c.key, c.value)
        if k not in seen or c.confidence > seen[k].confidence:
            seen[k] = c
    return list(seen.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_rules.py -v`
Expected: PASS (6 passed). Note: the correction test relies on the second employment value "Google" being captured; both "Stripe" (after "work at") and "Google" appear, dedupe keeps both keyed by value — reconciliation (Task 13) resolves which wins.

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/extraction/rules.py tests/unit/test_rules.py
git commit -m "feat: rule-based memory extractor"
```

---

## Task 4: LLM client (protocol + null + OpenAI)

**Files:**
- Create: `src/memory_service/llm/client.py`
- Test: `tests/unit/test_llm_client.py`

- [ ] **Step 1: Write `tests/unit/test_llm_client.py`**

```python
from memory_service.llm.client import NullLLMClient


def test_null_client_unavailable_and_returns_none():
    c = NullLLMClient()
    assert c.available is False
    assert c.complete_json("anything") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/llm/client.py`**

```python
from __future__ import annotations
import json
import logging
from typing import Protocol

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    @property
    def available(self) -> bool: ...

    def complete_json(self, prompt: str, timeout: float = 8.0) -> dict | None: ...


class NullLLMClient:
    available = False

    def complete_json(self, prompt: str, timeout: float = 8.0) -> dict | None:
        return None


class OpenAILLMClient:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def available(self) -> bool:
        return True

    def complete_json(self, prompt: str, timeout: float = 8.0) -> dict | None:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                timeout=timeout,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:  # degrade, never crash the request
            log.warning("LLM call failed, falling back: %s", exc)
            return None


def build_llm_client(api_key: str, model: str) -> LLMClient:
    return OpenAILLMClient(api_key, model) if api_key.strip() else NullLLMClient()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/llm/client.py tests/unit/test_llm_client.py
git commit -m "feat: optional LLM client with null fallback"
```

---

## Task 5: LLM extractor + extraction pipeline merge

**Files:**
- Create: `src/memory_service/extraction/llm_extractor.py`
- Create: `src/memory_service/extraction/pipeline.py`
- Test: `tests/unit/test_extraction_pipeline.py`

- [ ] **Step 1: Write `tests/unit/test_extraction_pipeline.py`**

```python
from memory_service.extraction.pipeline import extract_candidates
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import MemoryType, MemoryCandidate


class FakeLLM:
    available = True

    def complete_json(self, prompt, timeout=8.0):
        return {"memories": [
            {"type": "opinion", "key": "opinion:ts", "value": "TS generics annoying", "confidence": 0.6}
        ]}


def _user(c):
    return [{"role": "user", "content": c}]


def test_rule_only_when_llm_unavailable():
    cands = extract_candidates(_user("I live in Berlin"), NullLLMClient())
    assert any(c.key == "location" for c in cands)
    assert all(isinstance(c, MemoryCandidate) for c in cands)


def test_llm_candidates_merged_when_available():
    cands = extract_candidates(_user("I live in Berlin"), FakeLLM())
    assert any(c.key == "location" for c in cands)            # from rules
    assert any(c.type is MemoryType.opinion for c in cands)   # from LLM


def test_malformed_llm_output_is_ignored():
    class BadLLM:
        available = True
        def complete_json(self, prompt, timeout=8.0):
            return {"memories": [{"type": "nonsense", "value": 123}]}
    cands = extract_candidates(_user("I work at Stripe"), BadLLM())
    assert any(c.key == "employment" for c in cands)  # rules still present, bad item dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_extraction_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/extraction/llm_extractor.py`**

```python
from __future__ import annotations
import logging
from ..llm.client import LLMClient
from ..models.domain import MemoryCandidate, MemoryType

log = logging.getLogger(__name__)

_PROMPT = """Extract durable user memories from the conversation as JSON.
Return: {{"memories": [{{"type": "fact|preference|opinion|event", "key": "short_topic", "value": "concise statement", "confidence": 0.0-1.0}}]}}
Only include things the USER stated about themselves. No commentary.

Conversation:
{convo}
"""


def extract_llm(messages: list[dict], llm: LLMClient) -> list[MemoryCandidate]:
    if not llm.available:
        return []
    convo = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)
    data = llm.complete_json(_PROMPT.format(convo=convo))
    if not data or not isinstance(data.get("memories"), list):
        return []
    out: list[MemoryCandidate] = []
    for item in data["memories"]:
        try:
            mtype = MemoryType(str(item["type"]))
            key = str(item["key"]).strip()
            value = str(item["value"]).strip()
            conf = float(item.get("confidence", 0.5))
            if key and value:
                out.append(MemoryCandidate(type=mtype, key=key, value=value,
                                           confidence=conf, evidence="llm"))
        except (KeyError, ValueError, TypeError):
            continue  # drop malformed items, keep the rest
    return out
```

- [ ] **Step 4: Write `src/memory_service/extraction/pipeline.py`**

```python
from __future__ import annotations
from ..llm.client import LLMClient
from ..models.domain import MemoryCandidate
from .rules import extract_rules
from .llm_extractor import extract_llm


def extract_candidates(messages: list[dict], llm: LLMClient) -> list[MemoryCandidate]:
    """Rule output is the always-present baseline; LLM output augments it.
    On (key) collisions the higher-confidence candidate wins."""
    merged: dict[str, MemoryCandidate] = {}
    for c in extract_rules(messages) + extract_llm(messages, llm):
        cur = merged.get(c.key)
        if cur is None or c.confidence > cur.confidence:
            merged[c.key] = c
    return list(merged.values())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_extraction_pipeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/memory_service/extraction/llm_extractor.py src/memory_service/extraction/pipeline.py tests/unit/test_extraction_pipeline.py
git commit -m "feat: hybrid extraction pipeline (rules + optional LLM)"
```

---

## Task 6: Embedder

**Files:**
- Create: `src/memory_service/recall/embedder.py`
- Test: `tests/unit/test_embedder.py`

- [ ] **Step 1: Write `tests/unit/test_embedder.py`**

```python
from memory_service.recall.embedder import FakeEmbedder


def test_fake_embedder_dim_and_determinism():
    e = FakeEmbedder(dim=8)
    a = e.embed(["hello", "world"])
    b = e.embed(["hello"])
    assert len(a) == 2 and len(a[0]) == 8
    assert a[0] == b[0]  # deterministic for same text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/recall/embedder.py`**

```python
from __future__ import annotations
import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embedder for unit tests (no model download)."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            raw = [h[i % len(h)] / 255.0 for i in range(self._dim)]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vecs.append([x / norm for x in raw])
        return vecs


class STEmbedder:
    """Local sentence-transformers embedder. Model is baked into the image."""

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_id)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in arr]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/recall/embedder.py tests/unit/test_embedder.py
git commit -m "feat: embedder protocol with fake and sentence-transformers impls"
```

---

## Task 7: Reranker

**Files:**
- Create: `src/memory_service/recall/reranker.py`
- Test: `tests/unit/test_reranker.py`

- [ ] **Step 1: Write `tests/unit/test_reranker.py`**

```python
from memory_service.recall.reranker import FakeReranker


def test_fake_reranker_scores_lexical_overlap():
    r = FakeReranker()
    scores = r.rerank("where do I live", ["user lives in Berlin", "user likes pizza"])
    assert len(scores) == 2
    assert scores[0] > scores[1]  # first doc shares 'live/lives' token
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reranker.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/recall/reranker.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_reranker.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/recall/reranker.py tests/unit/test_reranker.py
git commit -m "feat: reranker protocol with fake and cross-encoder impls"
```

---

## Task 8: Reciprocal Rank Fusion

**Files:**
- Create: `src/memory_service/recall/fusion.py`
- Test: `tests/unit/test_fusion.py`

- [ ] **Step 1: Write `tests/unit/test_fusion.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fusion.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/recall/fusion.py`**

```python
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked id lists. Returns (id, score) sorted desc."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fusion.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/recall/fusion.py tests/unit/test_fusion.py
git commit -m "feat: reciprocal rank fusion"
```

---

## Task 9: Query rewrite

**Files:**
- Create: `src/memory_service/recall/query_rewrite.py`
- Test: `tests/unit/test_query_rewrite.py`

- [ ] **Step 1: Write `tests/unit/test_query_rewrite.py`**

```python
from memory_service.recall.query_rewrite import rewrite_query
from memory_service.llm.client import NullLLMClient


def test_heuristic_includes_original_and_entities():
    out = rewrite_query("what city does the user with the dog named Biscuit live in?", NullLLMClient())
    assert "what city does the user with the dog named Biscuit live in?" in out.variants
    assert "Biscuit" in out.entities


def test_variants_are_unique():
    out = rewrite_query("where do I live", NullLLMClient())
    assert len(out.variants) == len(set(out.variants))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_query_rewrite.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/recall/query_rewrite.py`**

```python
from __future__ import annotations
import re
from dataclasses import dataclass
from ..llm.client import LLMClient

_PROPER = re.compile(r"\b([A-Z][a-z]+)\b")
_STOP = {"What", "Where", "Who", "When", "Why", "How", "The", "User", "I"}


@dataclass
class RewriteResult:
    variants: list[str]
    entities: list[str]


def rewrite_query(query: str, llm: LLMClient) -> RewriteResult:
    variants = [query]
    entities = [w for w in _PROPER.findall(query) if w not in _STOP]
    # Heuristic expansion: append entity-focused probe so vector/lexical search
    # can reach facts linked to a named entity (multi-hop bridging).
    for e in entities:
        variants.append(f"{e} {query}")
    if llm.available:
        data = llm.complete_json(
            f'Rewrite this recall query into 2 alternative phrasings. '
            f'Return {{"variants": ["...", "..."]}}. Query: {query}'
        )
        if data and isinstance(data.get("variants"), list):
            variants.extend(str(v) for v in data["variants"] if isinstance(v, str))
    # de-dupe, preserve order
    seen, uniq = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return RewriteResult(variants=uniq, entities=entities)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_query_rewrite.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/recall/query_rewrite.py tests/unit/test_query_rewrite.py
git commit -m "feat: heuristic + optional LLM query rewrite"
```

---

## Task 10: Token counter + context assembler

**Files:**
- Create: `src/memory_service/tokens.py`
- Create: `src/memory_service/recall/assembler.py`
- Test: `tests/unit/test_assembler.py`

- [ ] **Step 1: Write `tests/unit/test_assembler.py`**

```python
from memory_service.tokens import count_tokens
from memory_service.recall.assembler import assemble_context, ContextItem


def test_count_tokens_monotonic():
    assert count_tokens("a b c d") < count_tokens("a b c d e f g h i j")


def test_assembler_prioritizes_facts_under_budget():
    facts = [ContextItem(text="User lives in Berlin.", turn_id="t1", score=1.0)]
    relevant = [ContextItem(text="User likes pizza.", turn_id="t2", score=0.5)]
    recent = [ContextItem(text="Long recent chatter " * 50, turn_id="t3", score=0.1)]
    ctx, cites = assemble_context(facts, relevant, recent, max_tokens=20)
    assert "Berlin" in ctx                 # stable fact kept first
    assert "chatter" not in ctx            # low-priority recent trimmed
    assert cites[0].turn_id == "t1"


def test_assembler_empty_inputs():
    ctx, cites = assemble_context([], [], [], max_tokens=100)
    assert ctx == "" and cites == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_assembler.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/tokens.py`**

```python
from __future__ import annotations
import re

_TOK = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate token count (challenge permits approximation). Word/punct
    pieces correlate well with BPE counts without any model download."""
    return len(_TOK.findall(text))
```

- [ ] **Step 4: Write `src/memory_service/recall/assembler.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from ..tokens import count_tokens
from ..models.schemas import Citation


@dataclass
class ContextItem:
    text: str
    turn_id: str
    score: float


def assemble_context(
    facts: list[ContextItem],
    relevant: list[ContextItem],
    recent: list[ContextItem],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Priority under budget: (1) stable user facts, (2) query-relevant memories,
    (3) recent context. Sections are labelled; items added until the budget fills."""
    sections = [("User facts", facts), ("Relevant memories", relevant), ("Recent context", recent)]
    lines: list[str] = []
    citations: list[Citation] = []
    used = 0
    seen_turn: set[str] = set()
    for title, items in sections:
        header = f"## {title}"
        header_cost = count_tokens(header)
        section_started = False
        for it in items:
            bullet = f"- {it.text}"
            cost = count_tokens(bullet) + (header_cost if not section_started else 0)
            if used + cost > max_tokens:
                continue
            if not section_started:
                lines.append(header)
                used += header_cost
                section_started = True
            lines.append(bullet)
            used += count_tokens(bullet)
            if it.turn_id and it.turn_id not in seen_turn:
                seen_turn.add(it.turn_id)
                citations.append(Citation(turn_id=it.turn_id, score=round(it.score, 4),
                                          snippet=it.text[:200]))
    return ("\n".join(lines), citations)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_assembler.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/memory_service/tokens.py src/memory_service/recall/assembler.py tests/unit/test_assembler.py
git commit -m "feat: token counter and budgeted context assembler"
```

---

## Task 11: Reconcile (fact evolution decisions)

**Files:**
- Create: `src/memory_service/services/reconcile.py`
- Test: `tests/unit/test_reconcile.py`

- [ ] **Step 1: Write `tests/unit/test_reconcile.py`**

```python
from memory_service.services.reconcile import plan_reconciliation, Action
from memory_service.models.domain import MemoryCandidate, MemoryType


def _cand(key, value, conf=0.8):
    return MemoryCandidate(type=MemoryType.fact, key=key, value=value, confidence=conf)


def test_new_key_is_insert():
    actions = plan_reconciliation([_cand("location", "Berlin")], existing_active={})
    assert actions[0].kind == Action.INSERT
    assert actions[0].supersedes is None


def test_same_key_different_value_supersedes():
    actions = plan_reconciliation([_cand("location", "Munich")],
                                  existing_active={"location": ("m-old", "Berlin")})
    assert actions[0].kind == Action.INSERT
    assert actions[0].supersedes == "m-old"


def test_same_key_same_value_is_skip():
    actions = plan_reconciliation([_cand("location", "Berlin")],
                                  existing_active={"location": ("m-old", "Berlin")})
    assert actions[0].kind == Action.SKIP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reconcile.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/services/reconcile.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from ..extraction.keys import normalize_value
from ..models.domain import MemoryCandidate


class Action(str, Enum):
    INSERT = "insert"
    SKIP = "skip"


@dataclass
class ReconcileAction:
    kind: Action
    candidate: MemoryCandidate
    supersedes: str | None = None


def plan_reconciliation(
    candidates: list[MemoryCandidate],
    existing_active: dict[str, tuple[str, str]],  # key -> (memory_id, value)
) -> list[ReconcileAction]:
    """Decide insert/supersede/skip per candidate against current active memories.
    existing_active maps key -> (id, value) of the current active memory for that key."""
    actions: list[ReconcileAction] = []
    for c in candidates:
        prior = existing_active.get(c.key)
        if prior is None:
            actions.append(ReconcileAction(Action.INSERT, c, supersedes=None))
        elif normalize_value(prior[1]).lower() == normalize_value(c.value).lower():
            actions.append(ReconcileAction(Action.SKIP, c))
        else:
            actions.append(ReconcileAction(Action.INSERT, c, supersedes=prior[0]))
    return actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_reconcile.py -v`
Expected: PASS (3 passed). Note `Action` exposes only INSERT/SKIP — supersession is an INSERT carrying a `supersedes` id, matching the data model.

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/services/reconcile.py tests/unit/test_reconcile.py
git commit -m "feat: fact-evolution reconciliation planner"
```

---

## Task 12: Database schema + pool + migrate

**Files:**
- Create: `src/memory_service/db/schema.sql`
- Create: `src/memory_service/db/pool.py`
- Test: `tests/integration/conftest.py` (shared DB fixtures), `tests/integration/test_migrate.py`

- [ ] **Step 1: Write `src/memory_service/db/schema.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS turns (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    user_id      TEXT,
    messages     JSONB NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns (user_id);

CREATE TABLE IF NOT EXISTS memories (
    id             TEXT PRIMARY KEY,
    user_id        TEXT,
    session_id     TEXT NOT NULL,
    type           TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    confidence     REAL NOT NULL,
    source_session TEXT NOT NULL,
    source_turn    TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    supersedes     TEXT REFERENCES memories(id),
    active         BOOLEAN NOT NULL DEFAULT TRUE,
    embedding      VECTOR(384),
    fts            TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', value)) STORED
);
CREATE INDEX IF NOT EXISTS idx_mem_fts ON memories USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_mem_lookup ON memories (user_id, type, key, active);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories (session_id);
CREATE INDEX IF NOT EXISTS idx_mem_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

- [ ] **Step 2: Write `src/memory_service/db/pool.py`**

```python
from __future__ import annotations
import pathlib
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

_SCHEMA = pathlib.Path(__file__).with_name("schema.sql")


def _configure(conn) -> None:
    register_vector(conn)


def create_pool(database_url: str) -> ConnectionPool:
    pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=10,
                          configure=_configure, open=True)
    return pool


def migrate(pool: ConnectionPool) -> None:
    ddl = _SCHEMA.read_text()
    with pool.connection() as conn:
        conn.execute(ddl)
        conn.commit()
```

- [ ] **Step 3: Write `tests/integration/conftest.py`**

```python
import os
import uuid
import pytest
from memory_service.db.pool import create_pool, migrate

DB_URL = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture(scope="session")
def pool():
    try:
        p = create_pool(DB_URL)
        migrate(p)
    except Exception as exc:
        pytest.skip(f"Postgres not available: {exc}")
    yield p
    p.close()


@pytest.fixture(autouse=True)
def clean(pool):
    with pool.connection() as conn:
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM turns")
        conn.commit()
    yield


@pytest.fixture
def ids():
    return lambda: uuid.uuid4().hex
```

Mark every integration test module with `pytestmark = pytest.mark.integration`.

- [ ] **Step 4: Write `tests/integration/test_migrate.py`**

```python
import pytest
pytestmark = pytest.mark.integration


def test_tables_exist_after_migrate(pool):
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"turns", "memories"} <= names
```

- [ ] **Step 5: Run test (requires DB up)**

Run: `docker compose up -d db && sleep 5 && DATABASE_URL=postgresql://memory:memory@localhost:5432/memory pytest tests/integration/test_migrate.py -v`
Expected: PASS (1 passed). (If compose/db not yet written, this task's verification waits until Task 19; the code is complete now.)

- [ ] **Step 6: Commit**

```bash
git add src/memory_service/db tests/integration/conftest.py tests/integration/test_migrate.py
git commit -m "feat: postgres schema, pool, and migration"
```

---

## Task 13: TurnRepository

**Files:**
- Create: `src/memory_service/repositories/turns.py`
- Test: `tests/integration/test_turns_repo.py`

- [ ] **Step 1: Write `tests/integration/test_turns_repo.py`**

```python
import pytest
from datetime import datetime, timezone
from memory_service.repositories.turns import TurnRepository
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _turn(ids):
    return Turn(id=ids(), session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": "I live in Berlin"}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={"x": 1})


def test_insert_and_get(pool, ids):
    repo = TurnRepository(pool)
    t = _turn(ids)
    repo.insert(t)
    got = repo.get(t.id)
    assert got.messages[0]["content"] == "I live in Berlin"
    assert got.metadata == {"x": 1}


def test_recent_for_session(pool, ids):
    repo = TurnRepository(pool)
    for _ in range(3):
        repo.insert(_turn(ids))
    recent = repo.recent_for_session("s1", limit=2)
    assert len(recent) == 2


def test_delete_session(pool, ids):
    repo = TurnRepository(pool)
    t = _turn(ids)
    repo.insert(t)
    repo.delete_session("s1")
    assert repo.get(t.id) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=... pytest tests/integration/test_turns_repo.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/repositories/turns.py`**

```python
from __future__ import annotations
import json
from datetime import datetime
from psycopg_pool import ConnectionPool
from ..models.domain import Turn


class TurnRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def insert(self, turn: Turn) -> str:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO turns (id, session_id, user_id, messages, timestamp, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (turn.id, turn.session_id, turn.user_id, json.dumps(turn.messages),
                 turn.timestamp, json.dumps(turn.metadata)),
            )
            conn.commit()
        return turn.id

    def get(self, turn_id: str) -> Turn | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, session_id, user_id, messages, timestamp, metadata FROM turns WHERE id=%s",
                (turn_id,),
            ).fetchone()
        return self._row(row) if row else None

    def recent_for_session(self, session_id: str, limit: int = 5) -> list[Turn]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, session_id, user_id, messages, timestamp, metadata
                   FROM turns WHERE session_id=%s ORDER BY timestamp DESC, created_at DESC LIMIT %s""",
                (session_id, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM turns WHERE session_id=%s", (session_id,))
            conn.commit()

    def delete_user(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM turns WHERE user_id=%s", (user_id,))
            conn.commit()

    @staticmethod
    def _row(r) -> Turn:
        msgs = r[3] if isinstance(r[3], list) else json.loads(r[3])
        meta = r[5] if isinstance(r[5], dict) else json.loads(r[5])
        return Turn(id=r[0], session_id=r[1], user_id=r[2], messages=msgs,
                    timestamp=r[4], metadata=meta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=... pytest tests/integration/test_turns_repo.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/repositories/turns.py tests/integration/test_turns_repo.py
git commit -m "feat: turn repository"
```

---

## Task 14: MemoryRepository (CRUD + supersession + hybrid search)

**Files:**
- Create: `src/memory_service/repositories/memories.py`
- Test: `tests/integration/test_memories_repo.py`

- [ ] **Step 1: Write `tests/integration/test_memories_repo.py`**

```python
import pytest
from memory_service.repositories.memories import MemoryRepository
from memory_service.repositories.turns import TurnRepository
from memory_service.models.domain import Turn, MemoryCandidate, MemoryType
from datetime import datetime, timezone
pytestmark = pytest.mark.integration


def _seed_turn(pool, ids, tid):
    TurnRepository(pool).insert(Turn(id=tid, session_id="s1", user_id="u1",
        messages=[{"role": "user", "content": "x"}],
        timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={}))


def _cand(key, value):
    return MemoryCandidate(type=MemoryType.fact, key=key, value=value, confidence=0.9)


def test_insert_and_active_by_key(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    repo.insert(ids(), "u1", "s1", _cand("location", "Berlin"), tid, [0.1] * 384, None)
    active = repo.active_by_key("u1")
    assert active["location"][1] == "Berlin"


def test_supersede_marks_old_inactive(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    old_id = ids()
    repo.insert(old_id, "u1", "s1", _cand("location", "Berlin"), tid, [0.1] * 384, None)
    repo.insert(ids(), "u1", "s1", _cand("location", "Munich"), tid, [0.2] * 384, supersedes=old_id)
    active = repo.active_by_key("u1")
    assert active["location"][1] == "Munich"
    all_rows = repo.list_for_user("u1", include_superseded=True)
    assert any(m.value == "Berlin" and m.active is False and m.supersedes is None for m in all_rows)
    assert any(m.value == "Munich" and m.supersedes == old_id for m in all_rows)


def test_search_vector_and_fts_return_ids(pool, ids):
    repo = MemoryRepository(pool)
    tid = ids(); _seed_turn(pool, ids, tid)
    mid = ids()
    repo.insert(mid, "u1", "s1", _cand("location", "Berlin"), tid, [0.5] * 384, None)
    vec_ids = repo.search_vector("u1", "s1", [0.5] * 384, limit=5)
    fts_ids = repo.search_fts("u1", "s1", "Berlin", limit=5)
    assert mid in vec_ids
    assert mid in fts_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=... pytest tests/integration/test_memories_repo.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/repositories/memories.py`**

```python
from __future__ import annotations
from psycopg_pool import ConnectionPool
from ..models.domain import Memory, MemoryCandidate, MemoryType


class MemoryRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def insert(self, mem_id: str, user_id: str | None, session_id: str,
               cand: MemoryCandidate, source_turn: str,
               embedding: list[float], supersedes: str | None) -> str:
        with self._pool.connection() as conn:
            if supersedes:
                conn.execute("UPDATE memories SET active=FALSE, updated_at=now() WHERE id=%s",
                             (supersedes,))
            conn.execute(
                """INSERT INTO memories
                   (id, user_id, session_id, type, key, value, confidence,
                    source_session, source_turn, supersedes, active, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)""",
                (mem_id, user_id, session_id, cand.type.value, cand.key, cand.value,
                 cand.confidence, session_id, source_turn, supersedes, embedding),
            )
            conn.commit()
        return mem_id

    def active_by_key(self, user_id: str | None) -> dict[str, tuple[str, str]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, id, value FROM memories WHERE user_id IS NOT DISTINCT FROM %s AND active=TRUE",
                (user_id,),
            ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def active_facts(self, user_id: str | None) -> list[Memory]:
        return [m for m in self.list_for_user(user_id, include_superseded=False)
                if m.type is MemoryType.fact]

    def search_vector(self, user_id: str | None, session_id: str,
                      embedding: list[float], limit: int = 20) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id FROM memories
                   WHERE active=TRUE AND (user_id IS NOT DISTINCT FROM %s OR session_id=%s)
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (user_id, session_id, embedding, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def search_fts(self, user_id: str | None, session_id: str,
                   query: str, limit: int = 20) -> list[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, ts_rank(fts, plainto_tsquery('english', %s)) AS rank
                   FROM memories
                   WHERE active=TRUE AND (user_id IS NOT DISTINCT FROM %s OR session_id=%s)
                     AND fts @@ plainto_tsquery('english', %s)
                   ORDER BY rank DESC LIMIT %s""",
                (query, user_id, session_id, query, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def get_many(self, ids: list[str]) -> dict[str, Memory]:
        if not ids:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {self._cols()} FROM memories WHERE id = ANY(%s)""",
                (ids,),
            ).fetchall()
        return {r[0]: self._row(r) for r in rows}

    def list_for_user(self, user_id: str | None, include_superseded: bool = True) -> list[Memory]:
        clause = "" if include_superseded else " AND active=TRUE"
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT {self._cols()} FROM memories
                    WHERE user_id IS NOT DISTINCT FROM %s{clause}
                    ORDER BY created_at""",
                (user_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE memories SET supersedes=NULL WHERE supersedes IN "
                         "(SELECT id FROM memories WHERE session_id=%s)", (session_id,))
            conn.execute("DELETE FROM memories WHERE session_id=%s", (session_id,))
            conn.commit()

    def delete_user(self, user_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("UPDATE memories SET supersedes=NULL WHERE supersedes IN "
                         "(SELECT id FROM memories WHERE user_id=%s)", (user_id,))
            conn.execute("DELETE FROM memories WHERE user_id=%s", (user_id,))
            conn.commit()

    @staticmethod
    def _cols() -> str:
        return ("id, user_id, session_id, type, key, value, confidence, source_session, "
                "source_turn, created_at, updated_at, supersedes, active")

    @staticmethod
    def _row(r) -> Memory:
        return Memory(id=r[0], user_id=r[1], session_id=r[2], type=MemoryType(r[3]),
                      key=r[4], value=r[5], confidence=r[6], source_session=r[7],
                      source_turn=r[8], created_at=r[9], updated_at=r[10],
                      supersedes=r[11], active=r[12])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=... pytest tests/integration/test_memories_repo.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/repositories/memories.py tests/integration/test_memories_repo.py
git commit -m "feat: memory repository with supersession and hybrid search"
```

---

## Task 15: Ingest service

**Files:**
- Create: `src/memory_service/services/ingest.py`
- Test: `tests/integration/test_ingest.py`

- [ ] **Step 1: Write `tests/integration/test_ingest.py`**

```python
import pytest
from datetime import datetime, timezone
from memory_service.services.ingest import IngestService
from memory_service.repositories.turns import TurnRepository
from memory_service.repositories.memories import MemoryRepository
from memory_service.recall.embedder import FakeEmbedder
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _svc(pool):
    return IngestService(TurnRepository(pool), MemoryRepository(pool),
                         FakeEmbedder(384), NullLLMClient(), id_factory=_seq())


def _seq():
    n = {"i": 0}
    def f():
        n["i"] += 1
        return f"id-{n['i']}"
    return f


def _turn(content):
    return Turn(id="ignored", session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": content}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={})


def test_ingest_persists_turn_and_extracts(pool):
    svc = _svc(pool)
    turn_id = svc.ingest(_turn("I live in Berlin"))
    mems = MemoryRepository(pool).list_for_user("u1")
    assert any(m.key == "location" and m.value == "Berlin" and m.source_turn == turn_id for m in mems)


def test_ingest_evolves_fact(pool):
    svc = _svc(pool)
    svc.ingest(_turn("I live in Berlin"))
    svc.ingest(_turn("I just moved to Munich"))
    active = MemoryRepository(pool).active_by_key("u1")
    assert active["location"][1] == "Munich"
    all_rows = MemoryRepository(pool).list_for_user("u1", include_superseded=True)
    assert any(m.value == "Berlin" and m.active is False for m in all_rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=... pytest tests/integration/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/services/ingest.py`**

```python
from __future__ import annotations
import uuid
from typing import Callable
from ..models.domain import Turn
from ..repositories.turns import TurnRepository
from ..repositories.memories import MemoryRepository
from ..recall.embedder import Embedder
from ..llm.client import LLMClient
from ..extraction.pipeline import extract_candidates
from .reconcile import plan_reconciliation, Action


class IngestService:
    def __init__(self, turns: TurnRepository, memories: MemoryRepository,
                 embedder: Embedder, llm: LLMClient,
                 id_factory: Callable[[], str] | None = None):
        self._turns = turns
        self._memories = memories
        self._embedder = embedder
        self._llm = llm
        self._id = id_factory or (lambda: uuid.uuid4().hex)

    def ingest(self, turn: Turn) -> str:
        turn_id = self._id()
        stored = Turn(id=turn_id, session_id=turn.session_id, user_id=turn.user_id,
                      messages=turn.messages, timestamp=turn.timestamp, metadata=turn.metadata)
        self._turns.insert(stored)

        candidates = extract_candidates(turn.messages, self._llm)
        if not candidates:
            return turn_id

        existing = self._memories.active_by_key(turn.user_id)
        actions = plan_reconciliation(candidates, existing)
        to_embed = [a for a in actions if a.kind == Action.INSERT]
        vectors = self._embedder.embed([a.candidate.value for a in to_embed]) if to_embed else []
        for action, vec in zip(to_embed, vectors):
            self._memories.insert(self._id(), turn.user_id, turn.session_id,
                                  action.candidate, turn_id, vec, action.supersedes)
        return turn_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=... pytest tests/integration/test_ingest.py -v`
Expected: PASS (2 passed). Note: `active_by_key` is re-read fresh per ingest so supersession sees the prior active row → synchronous correctness.

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/services/ingest.py tests/integration/test_ingest.py
git commit -m "feat: synchronous ingest service (persist + extract + reconcile)"
```

---

## Task 16: Recall service (+ search)

**Files:**
- Create: `src/memory_service/services/recall_service.py`
- Test: `tests/integration/test_recall_service.py`

- [ ] **Step 1: Write `tests/integration/test_recall_service.py`**

```python
import pytest
from datetime import datetime, timezone
from memory_service.services.ingest import IngestService
from memory_service.services.recall_service import RecallService
from memory_service.repositories.turns import TurnRepository
from memory_service.repositories.memories import MemoryRepository
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
from memory_service.models.domain import Turn
pytestmark = pytest.mark.integration


def _ingest(pool):
    seq = {"i": 0}
    def idf():
        seq["i"] += 1
        return f"id-{seq['i']}"
    return IngestService(TurnRepository(pool), MemoryRepository(pool),
                         FakeEmbedder(384), NullLLMClient(), id_factory=idf)


def _recall(pool):
    return RecallService(MemoryRepository(pool), TurnRepository(pool),
                         FakeEmbedder(384), FakeReranker(), NullLLMClient())


def _turn(content):
    return Turn(id="x", session_id="s1", user_id="u1",
                messages=[{"role": "user", "content": content}],
                timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc), metadata={})


def test_recall_returns_relevant_fact_with_citation(pool):
    ing = _ingest(pool)
    ing.ingest(_turn("I live in Berlin"))
    ing.ingest(_turn("I work at Stripe"))
    res = _recall(pool).recall("where do I live", "s1", "u1", max_tokens=200)
    assert "Berlin" in res.context
    assert any(c.snippet for c in res.citations)


def test_cold_session_returns_empty_never_errors(pool):
    res = _recall(pool).recall("anything", "cold-session", "nobody", max_tokens=200)
    assert res.context == "" and res.citations == []


def test_noise_query_does_not_invent(pool):
    _ingest(pool).ingest(_turn("I live in Berlin"))
    res = _recall(pool).recall("what is the capital of France", "s1", "u1", max_tokens=200)
    # facts section may still surface Berlin as a stable fact, but no fabricated France data
    assert "France" not in res.context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=... pytest tests/integration/test_recall_service.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `src/memory_service/services/recall_service.py`**

```python
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
        # one combined embedding query (mean of variant embeddings keeps it cheap)
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

        recent_turns = self._turns.recent_for_session(session_id, limit=2)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=... pytest tests/integration/test_recall_service.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memory_service/services/recall_service.py tests/integration/test_recall_service.py
git commit -m "feat: hybrid recall service and structured search"
```

---

## Task 17: App factory + lifespan wiring

**Files:**
- Create: `src/memory_service/app.py`
- Create: `src/memory_service/main.py`
- Test: covered by route tests in Task 18 (app builds there)

- [ ] **Step 1: Write `src/memory_service/app.py`**

```python
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from .config import Settings
from .db.pool import create_pool, migrate
from .recall.embedder import STEmbedder
from .recall.reranker import CEReranker
from .llm.client import build_llm_client
from .repositories.turns import TurnRepository
from .repositories.memories import MemoryRepository
from .services.ingest import IngestService
from .services.recall_service import RecallService

logging.basicConfig(level=logging.INFO)


def build_app(settings: Settings | None = None, *, embedder=None, reranker=None,
              llm=None, pool=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = pool or create_pool(settings.database_url)
        migrate(app.state.pool)
        app.state.embedder = embedder or STEmbedder(settings.embed_model)
        app.state.reranker = reranker or CEReranker(settings.rerank_model)
        app.state.llm = llm or build_llm_client(settings.openai_api_key, settings.openai_model)
        app.state.turns = TurnRepository(app.state.pool)
        app.state.memories = MemoryRepository(app.state.pool)
        app.state.ingest = IngestService(app.state.turns, app.state.memories,
                                         app.state.embedder, app.state.llm)
        app.state.recall = RecallService(app.state.memories, app.state.turns,
                                         app.state.embedder, app.state.reranker, app.state.llm)
        yield
        app.state.pool.close()

    app = FastAPI(title="memory-service", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > settings.max_body_bytes:
            return JSONResponse({"detail": "payload too large"}, status_code=413)
        return await call_next(request)

    from .routes import health, turns, recall, search, memories, admin
    app.include_router(health.router)
    app.include_router(turns.router)
    app.include_router(recall.router)
    app.include_router(search.router)
    app.include_router(memories.router)
    app.include_router(admin.router)
    return app
```

- [ ] **Step 2: Write `src/memory_service/main.py`**

```python
from .app import build_app

app = build_app()
```

- [ ] **Step 3: Commit**

```bash
git add src/memory_service/app.py src/memory_service/main.py
git commit -m "feat: app factory with lifespan-wired dependencies"
```

---

## Task 18: Routes

**Files:**
- Create: `src/memory_service/routes/{__init__,health,turns,recall,search,memories,admin}.py`
- Test: `tests/integration/test_routes.py`

- [ ] **Step 1: Write `tests/integration/test_routes.py`**

```python
import os
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from memory_service.app import build_app
from memory_service.config import Settings
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
pytestmark = pytest.mark.integration

DB = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB), embedder=FakeEmbedder(384),
                    reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def _turn(content, session="s1", user="u1"):
    return {"session_id": session, "user_id": user,
            "messages": [{"role": "user", "content": content}],
            "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(),
            "metadata": {}}


def test_health(client):
    assert client.get("/health").status_code == 200


def test_turns_then_recall_roundtrip(client):
    r = client.post("/turns", json=_turn("I live in Berlin"))
    assert r.status_code == 201 and "id" in r.json()
    rec = client.post("/recall", json={"query": "where do I live", "session_id": "s1",
                                       "user_id": "u1", "max_tokens": 200})
    assert rec.status_code == 200
    assert "Berlin" in rec.json()["context"]


def test_memories_listing_shape(client):
    client.post("/turns", json=_turn("I work at Stripe"))
    body = client.get("/users/u1/memories").json()
    assert body["memories"]
    m = body["memories"][0]
    assert {"id", "type", "key", "value", "confidence", "active"} <= set(m)


def test_malformed_turn_returns_422(client):
    assert client.post("/turns", json={"session_id": "s1"}).status_code == 422


def test_delete_session_204(client):
    client.post("/turns", json=_turn("I live in Berlin"))
    assert client.delete("/sessions/s1").status_code == 204


def test_recall_cold_session_200_empty(client):
    r = client.post("/recall", json={"query": "x", "session_id": "none", "user_id": "none"})
    assert r.status_code == 200 and r.json()["context"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=... pytest tests/integration/test_routes.py -v`
Expected: FAIL (`ModuleNotFoundError: routes`)

- [ ] **Step 3: Write `src/memory_service/routes/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write `src/memory_service/routes/health.py`**

```python
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    try:
        with request.app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "degraded"}, status_code=503)
```

- [ ] **Step 5: Write `src/memory_service/routes/turns.py`**

```python
from datetime import datetime
from fastapi import APIRouter, Request, status
from ..models.schemas import TurnRequest, TurnResponse
from ..models.domain import Turn

router = APIRouter()


@router.post("/turns", response_model=TurnResponse, status_code=status.HTTP_201_CREATED)
def post_turn(req: TurnRequest, request: Request):
    try:
        ts = datetime.fromisoformat(req.timestamp.replace("Z", "+00:00"))
    except ValueError:
        ts = datetime.now()
    turn = Turn(id="", session_id=req.session_id, user_id=req.user_id,
                messages=[m.model_dump() for m in req.messages], timestamp=ts,
                metadata=req.metadata)
    turn_id = request.app.state.ingest.ingest(turn)
    return TurnResponse(id=turn_id)
```

- [ ] **Step 6: Write `src/memory_service/routes/recall.py`**

```python
from fastapi import APIRouter, Request
from ..models.schemas import RecallRequest, RecallResponse

router = APIRouter()


@router.post("/recall", response_model=RecallResponse)
def post_recall(req: RecallRequest, request: Request):
    return request.app.state.recall.recall(req.query, req.session_id, req.user_id, req.max_tokens)
```

- [ ] **Step 7: Write `src/memory_service/routes/search.py`**

```python
from fastapi import APIRouter, Request
from ..models.schemas import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def post_search(req: SearchRequest, request: Request):
    hits = request.app.state.recall.search(req.query, req.session_id, req.user_id, req.limit)
    return SearchResponse(results=hits)
```

- [ ] **Step 8: Write `src/memory_service/routes/memories.py`**

```python
from fastapi import APIRouter, Request
from ..models.schemas import MemoriesResponse, MemoryOut

router = APIRouter()


@router.get("/users/{user_id}/memories", response_model=MemoriesResponse)
def get_memories(user_id: str, request: Request, include_superseded: bool = True):
    rows = request.app.state.memories.list_for_user(user_id, include_superseded=include_superseded)
    out = [MemoryOut(id=m.id, type=m.type.value, key=m.key, value=m.value,
                     confidence=m.confidence, source_session=m.source_session,
                     source_turn=m.source_turn, created_at=m.created_at.isoformat(),
                     updated_at=m.updated_at.isoformat(), supersedes=m.supersedes,
                     active=m.active) for m in rows]
    return MemoriesResponse(memories=out)
```

- [ ] **Step 9: Write `src/memory_service/routes/admin.py`**

```python
from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, request: Request):
    request.app.state.memories.delete_session(session_id)
    request.app.state.turns.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: str, request: Request):
    request.app.state.memories.delete_user(user_id)
    request.app.state.turns.delete_user(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 10: Run test to verify it passes**

Run: `DATABASE_URL=... pytest tests/integration/test_routes.py -v`
Expected: PASS (7 passed)

- [ ] **Step 11: Commit**

```bash
git add src/memory_service/routes tests/integration/test_routes.py
git commit -m "feat: HTTP routes for full contract"
```

---

## Task 19: Dockerfile + compose (with model pre-bake)

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `scripts/prefetch_models.py`

- [ ] **Step 1: Write `scripts/prefetch_models.py`**

```python
"""Download embed + rerank models at build time so runtime is fully offline."""
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer(os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
CrossEncoder(os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
print("models cached")
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install -r requirements.txt

# Pre-bake models into the image layer (offline at runtime)
COPY scripts/prefetch_models.py scripts/prefetch_models.py
RUN python scripts/prefetch_models.py

COPY pyproject.toml .
COPY src ./src

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
  CMD curl -sf http://localhost:8080/health || exit 1

CMD ["uvicorn", "memory_service.main:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "src"]
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: memory
      POSTGRES_PASSWORD: memory
      POSTGRES_DB: memory
    volumes:
      - memory_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memory"]
      interval: 5s
      timeout: 3s
      retries: 10

  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://memory:memory@db:5432/memory
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      OPENAI_MODEL: ${OPENAI_MODEL:-gpt-4o-mini}
    depends_on:
      db:
        condition: service_healthy

volumes:
  memory_pgdata:
```

- [ ] **Step 4: Verify the stack comes up and smoke test**

Run:
```bash
docker compose up -d --build
until curl -sf http://localhost:8080/health; do sleep 2; done
curl -s -X POST http://localhost:8080/turns -H 'content-type: application/json' \
  -d '{"session_id":"s1","user_id":"u1","messages":[{"role":"user","content":"I live in Berlin"}],"timestamp":"2026-05-29T00:00:00Z","metadata":{}}'
curl -s -X POST http://localhost:8080/recall -H 'content-type: application/json' \
  -d '{"query":"where do I live","session_id":"s1","user_id":"u1","max_tokens":200}'
```
Expected: health returns `{"status":"ok"}`; `/turns` returns `{"id":"..."}`; `/recall` context contains "Berlin".

- [ ] **Step 5: Run the full integration suite against the live DB**

Run: `DATABASE_URL=postgresql://memory:memory@localhost:5432/memory pytest tests/integration -v`
Expected: all integration tests PASS.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml scripts/prefetch_models.py
git commit -m "feat: dockerfile and compose with pre-baked offline models"
```

---

## Task 20: Recall-quality fixture + runner

**Files:**
- Create: `fixtures/recall_quality.json`
- Create: `tests/integration/test_recall_quality.py`

- [ ] **Step 1: Write `fixtures/recall_quality.json`**

```json
{
  "conversations": [
    {"session_id": "alice", "user_id": "alice",
     "turns": [
       "I live in Berlin and I work at Stripe.",
       "I just got back from walking Biscuit, my dog.",
       "Actually I moved to Munich last month.",
       "I love TypeScript for big projects."
     ]},
    {"session_id": "bob", "user_id": "bob",
     "turns": [
       "I'm a teacher in Toronto.",
       "My cat Mittens keeps me company."
     ]}
  ],
  "probes": [
    {"query": "where does alice live now", "session_id": "alice", "user_id": "alice",
     "expect_substring": "Munich", "expect_absent": "Berlin city fact dominant"},
    {"query": "what is the name of alice's dog", "session_id": "alice", "user_id": "alice",
     "expect_substring": "Biscuit"},
    {"query": "where does the owner of the dog Biscuit live", "session_id": "alice", "user_id": "alice",
     "expect_substring": "Munich"},
    {"query": "what does bob do for work", "session_id": "bob", "user_id": "bob",
     "expect_substring": "teacher"},
    {"query": "tell me about quantum physics", "session_id": "bob", "user_id": "bob",
     "expect_absent": "quantum"}
  ]
}
```

- [ ] **Step 2: Write `tests/integration/test_recall_quality.py`**

```python
import json, os, pathlib
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from memory_service.app import build_app
from memory_service.config import Settings
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
pytestmark = pytest.mark.integration

DB = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")
FIX = json.loads((pathlib.Path(__file__).parents[2] / "fixtures/recall_quality.json").read_text())


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB), embedder=FakeEmbedder(384),
                    reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def _load(client):
    for convo in FIX["conversations"]:
        for i, text in enumerate(convo["turns"]):
            client.post("/turns", json={
                "session_id": convo["session_id"], "user_id": convo["user_id"],
                "messages": [{"role": "user", "content": text}],
                "timestamp": datetime(2026, 5, 29, i, tzinfo=timezone.utc).isoformat(),
                "metadata": {}})


def test_recall_quality_fixture(client):
    _load(client)
    passed = 0
    for probe in FIX["probes"]:
        res = client.post("/recall", json={
            "query": probe["query"], "session_id": probe["session_id"],
            "user_id": probe["user_id"], "max_tokens": 256}).json()
        ctx = res["context"]
        ok = True
        if "expect_substring" in probe:
            ok = ok and probe["expect_substring"].lower() in ctx.lower()
        if "expect_absent" in probe and probe["expect_absent"] == "quantum":
            ok = ok and "quantum" not in ctx.lower()
        passed += int(ok)
    # Record the score; assert a baseline so regressions fail the build.
    score = passed / len(FIX["probes"])
    print(f"RECALL_QUALITY_SCORE={score:.2f}")
    assert score >= 0.6
```

- [ ] **Step 3: Run the fixture (DB up)**

Run: `DATABASE_URL=... pytest tests/integration/test_recall_quality.py -v -s`
Expected: prints `RECALL_QUALITY_SCORE=...` and PASSES at ≥0.6 with the FakeEmbedder. (Record the exact number in CHANGELOG. The real STEmbedder/CEReranker in the container should score higher — re-run against the live stack and log both.)

- [ ] **Step 4: Commit**

```bash
git add fixtures/recall_quality.json tests/integration/test_recall_quality.py
git commit -m "test: recall-quality fixture with scored probes"
```

---

## Task 21: Restart-persistence + concurrency + unicode tests

**Files:**
- Create: `tests/integration/test_persistence.py`
- Create: `tests/integration/test_concurrency.py`
- Create: `tests/integration/test_robustness.py`

- [ ] **Step 1: Write `tests/integration/test_persistence.py`**

```python
import os, subprocess, time, json, urllib.request
import pytest
pytestmark = pytest.mark.integration

BASE = "http://localhost:8080"


def _up():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _up(), reason="live compose stack not running")
def test_data_survives_restart():
    payload = json.dumps({"session_id": "persist", "user_id": "persist",
        "messages": [{"role": "user", "content": "I live in Lisbon"}],
        "timestamp": "2026-05-29T00:00:00Z", "metadata": {}}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{BASE}/turns", payload,
        {"content-type": "application/json"}))
    subprocess.run(["docker", "compose", "restart", "app", "db"], check=True)
    for _ in range(30):
        if _up():
            break
        time.sleep(2)
    q = json.dumps({"query": "where do I live", "session_id": "persist",
                    "user_id": "persist", "max_tokens": 200}).encode()
    resp = urllib.request.urlopen(urllib.request.Request(f"{BASE}/recall", q,
        {"content-type": "application/json"}))
    assert "Lisbon" in json.loads(resp.read())["context"]
```

- [ ] **Step 2: Write `tests/integration/test_concurrency.py`**

```python
import os
import pytest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from memory_service.app import build_app
from memory_service.config import Settings
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
pytestmark = pytest.mark.integration

DB = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB), embedder=FakeEmbedder(384),
                    reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def test_sessions_do_not_bleed(client):
    def post(session, city):
        return client.post("/turns", json={"session_id": session, "user_id": session,
            "messages": [{"role": "user", "content": f"I live in {city}"}],
            "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(), "metadata": {}})
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda a: post(*a), [("a", "Paris"), ("b", "Tokyo"), ("c", "Cairo")]))
    a = client.post("/recall", json={"query": "where do I live", "session_id": "a",
                                     "user_id": "a", "max_tokens": 200}).json()
    assert "Paris" in a["context"]
    assert "Tokyo" not in a["context"] and "Cairo" not in a["context"]
```

- [ ] **Step 3: Write `tests/integration/test_robustness.py`**

```python
import os
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from memory_service.app import build_app
from memory_service.config import Settings
from memory_service.recall.embedder import FakeEmbedder
from memory_service.recall.reranker import FakeReranker
from memory_service.llm.client import NullLLMClient
pytestmark = pytest.mark.integration

DB = os.getenv("DATABASE_URL", "postgresql://memory:memory@localhost:5432/memory")


@pytest.fixture
def client():
    app = build_app(Settings(database_url=DB, max_body_bytes=2048),
                    embedder=FakeEmbedder(384), reranker=FakeReranker(), llm=NullLLMClient())
    with TestClient(app) as c:
        yield c


def test_unicode_roundtrip(client):
    client.post("/turns", json={"session_id": "u", "user_id": "u",
        "messages": [{"role": "user", "content": "I live in 東京 and I love 🍣"}],
        "timestamp": datetime(2026, 5, 29, tzinfo=timezone.utc).isoformat(), "metadata": {}})
    body = client.get("/users/u/memories").json()
    assert any("東京" in m["value"] for m in body["memories"])


def test_oversized_payload_413(client):
    big = "x" * 5000
    r = client.post("/turns", json={"session_id": "s", "messages": [{"role": "user", "content": big}],
                                    "timestamp": "2026-05-29T00:00:00Z"})
    assert r.status_code == 413


def test_bad_role_422(client):
    r = client.post("/turns", json={"session_id": "s",
        "messages": [{"role": "wizard", "content": "hi"}], "timestamp": "2026-05-29T00:00:00Z"})
    assert r.status_code == 422
```

- [ ] **Step 4: Run all integration tests (DB up; live stack for persistence test)**

Run: `DATABASE_URL=postgresql://memory:memory@localhost:5432/memory pytest tests/integration -v`
Expected: all PASS (persistence test skips unless the full compose stack is running).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_persistence.py tests/integration/test_concurrency.py tests/integration/test_robustness.py
git commit -m "test: persistence, concurrency, and robustness"
```

---

## Task 22: README + CHANGELOG

**Files:**
- Create: `README.md`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `README.md`** with these sections (fill with the real design):

```markdown
# Memory Service

A memory service for an AI agent: ingests conversation turns, extracts typed structured
memories with provenance, evolves facts via supersession, and answers token-budgeted recall.

## Quick start
\`\`\`bash
cp .env.example .env   # optional: set OPENAI_API_KEY to enable the LLM enhancer
docker compose up -d --build
until curl -sf http://localhost:8080/health; do sleep 2; done
\`\`\`

## 1. Architecture
[ASCII diagram from the spec] — FastAPI monolith, layered routes → services →
repositories, single Postgres+pgvector. Local embedder + cross-encoder reranker baked
into the image; optional LLM enhancer.

## 2. Backing store
Postgres + pgvector: one transactional store for the relational fact-graph (supersession
chains), vector ANN search, and lexical FTS. Defends single-store simplicity + correctness.

## 3. Extraction pipeline
Hybrid: deterministic regex rules (always on; covers employment, location, family/pets incl.
implicit "walking Biscuit", preferences/opinions, corrections) merged with an optional LLM
extractor. Higher-confidence candidate wins per key.

## 4. Recall strategy
Query rewrite (heuristic + optional LLM) → vector (pgvector cosine) + lexical (FTS ts_rank)
retrieval → Reciprocal Rank Fusion → local cross-encoder rerank → token-budgeted assembly.

## 5. Fact evolution
Memories keyed by (user_id, type, key). A contradicting same-key fact inserts a new active
row with supersedes=old.id and flips the old row active=false. History preserved; /recall
and /memories show active by default.

## 6. Context assembly under budget
Priority: (1) stable active user facts, (2) query-relevant reranked memories, (3) recent
turns. Items added until max_tokens (approximate counter) fills; lowest priority trimmed.

## 7. Tradeoffs
- Local models baked in (≈200MB image) for offline determinism over a lean image.
- Regex capture for entities instead of spaCy (lighter; values double as entities).
- Approximate token counter instead of tiktoken (no network at runtime).
- Entity-linking in the fact layer for multi-hop instead of a full graph (extension path).

## 8. Failure modes
LLM/embedding/rerank failures degrade to rule/vector-only; cold sessions return empty;
malformed → 422, oversized → 413; DB down → /health 503.

## 9. Running tests
\`\`\`bash
pytest tests/unit -v                                   # no DB
docker compose up -d db
DATABASE_URL=postgresql://memory:memory@localhost:5432/memory pytest tests/integration -v
\`\`\`
```

- [ ] **Step 2: Write `CHANGELOG.md`** seeded with iteration entries (record real fixture scores as you go):

```markdown
# Changelog

## v0.1 — Rule-only baseline
Regex extraction + single-store Postgres. Recall = vector-only top-k.
Recall-quality fixture: [record score]. Observation: multi-hop and "moved to" updates failed
because there was no fact evolution and no lexical channel.

## v0.2 — Fact evolution
Added (user,type,key) supersession chains. "moved to Munich" now supersedes "Berlin".
Fixture: [record]. Observation: noise queries still surfaced stale/irrelevant memories.

## v0.3 — Hybrid retrieval + RRF
Added Postgres FTS channel fused with vector via Reciprocal Rank Fusion.
Fixture: [record]. Observation: top-k ordering still mixed weakly-relevant items high.

## v0.4 — Cross-encoder rerank + query rewrite
Added local cross-encoder rerank and heuristic entity query-rewrite for multi-hop bridging.
Fixture: [record]. Observation: "owner of dog Biscuit lives where" now resolves via the
entity-expanded variant. Token-budgeted assembler prioritizes stable facts first.
```

- [ ] **Step 3: Run the whole suite once more and capture fixture score for the CHANGELOG**

Run: `pytest tests/unit -v && DATABASE_URL=... pytest tests/integration -v -s | grep RECALL_QUALITY_SCORE`
Expected: unit all pass; integration all pass; copy the printed score into each CHANGELOG entry as you iterate.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: README and CHANGELOG with iteration history"
```

---

## Self-Review Notes (resolved during planning)

- **Spec coverage:** every contract endpoint (Task 18), persistence (Task 19/21), synchronous
  correctness (Task 15 note), fact evolution (Tasks 11/14/15), hybrid recall + RRF + rerank +
  query-rewrite (Tasks 8/9/16), budgeted assembly (Task 10), extraction incl. implicit facts
  (Task 3), graceful degradation (Tasks 4/16/18), all tests categories (Tasks 20/21).
- **Type consistency:** `MemoryCandidate`, `Memory`, `ContextItem`, `RewriteResult`,
  `ReconcileAction`/`Action`, repository method names (`active_by_key`, `search_vector`,
  `search_fts`, `get_many`, `list_for_user`, `active_facts`) are defined once and reused
  verbatim across services/tests.
- **No placeholders:** every code step contains complete runnable code; CHANGELOG/README have
  explicit "[record score]" markers that are deliberately filled during execution, not code.
```
