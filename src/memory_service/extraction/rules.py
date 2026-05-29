from __future__ import annotations
import re
from .keys import normalize_value, pet_key, KEY_LOCATION, KEY_EMPLOYMENT
from ..models.domain import MemoryCandidate, MemoryType

# Lookahead to stop value capture at sentence delimiters or trailing context words.
_TRAIL_STOP = (
    r"(?=\s*[,.]"
    r"|\s+(?:now|and|but|so|then|though|because|since|when|where|while|after"
    r"|before|until|unless|if|as)\b"
    r"|\Z)"
)

# Each pattern captures the value in group 1.
_PATTERNS = [
    (KEY_LOCATION, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:live|reside|am based) in ([A-Z][\w .'-]+?)" + _TRAIL_STOP, re.I)),
    (KEY_LOCATION, MemoryType.fact, 0.8,
     re.compile(r"\bi(?:'m| am) (?:from|moving to|moved to) ([A-Z][\w .'-]+?)" + _TRAIL_STOP, re.I)),
    (KEY_LOCATION, MemoryType.fact, 0.8,
     re.compile(r"\bi (?:just )?moved to ([A-Z][\w .'-]+?)" + _TRAIL_STOP, re.I)),
    (KEY_EMPLOYMENT, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:work|am working) (?:at|for) ([A-Z][\w .&'-]+?)" + _TRAIL_STOP, re.I)),
    (KEY_EMPLOYMENT, MemoryType.fact, 0.85,
     re.compile(r"\bi (?:joined|now work at) ([A-Z][\w .&'-]+?)" + _TRAIL_STOP, re.I)),
]

_PET = re.compile(r"\b(?:walking|feeding|my dog|my cat|petting) ([A-Z][a-z]+)")
_PREF_POS = re.compile(r"\bi (?:love|like|prefer|enjoy|am a fan of) ([\w .#+'-]+?)" + _TRAIL_STOP, re.I)
_PREF_NEG = re.compile(r"\bi (?:hate|dislike|can't stand|don't like) ([\w .#+'-]+?)" + _TRAIL_STOP, re.I)


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
