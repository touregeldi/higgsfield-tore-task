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
