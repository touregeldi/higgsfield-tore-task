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
    (3) recent context. Greedy-fill within a tier; once a tier is truncated for
    budget, lower-priority tiers are not started (strict cross-tier priority)."""
    sections = [("User facts", facts), ("Relevant memories", relevant), ("Recent context", recent)]
    lines: list[str] = []
    citations: list[Citation] = []
    used = 0
    truncated = False
    seen_turn: set[str] = set()
    for title, items in sections:
        if truncated:
            break
        header = f"## {title}"
        header_cost = count_tokens(header)
        section_started = False
        for it in items:
            bullet = f"- {it.text}"
            cost = count_tokens(bullet) + (header_cost if not section_started else 0)
            if used + cost > max_tokens:
                truncated = True
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
