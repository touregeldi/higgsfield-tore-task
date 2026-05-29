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
