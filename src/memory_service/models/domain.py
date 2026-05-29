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
