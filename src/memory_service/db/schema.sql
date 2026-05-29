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
