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
