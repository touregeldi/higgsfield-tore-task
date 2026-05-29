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
