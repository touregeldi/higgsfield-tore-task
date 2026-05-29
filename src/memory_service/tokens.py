from __future__ import annotations
import re

_TOK = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate token count (challenge permits approximation). Word/punct
    pieces correlate well with BPE counts without any model download."""
    return len(_TOK.findall(text))
