"""
anthropic_client.py

Drop-in replacement for Democritus's OpenAIChatClient using the native
Anthropic SDK.  Implements the LLMClient Protocol (ask / ask_batch).

Usage (in run_extraction.py):
    from extraction.anthropic_client import AnthropicChatClient
    client = AnthropicChatClient()
    # Patch into Democritus factory:
    import llms.factory as factory
    factory.make_llm_client = lambda **kw: client
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

try:
    import anthropic as _anthropic
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


class AnthropicChatClient:
    """
    Implements Democritus LLMClient Protocol using the Anthropic Messages API.

    Env vars:
        ANTHROPIC_API_KEY     – required
        ANTHROPIC_BASE_URL    – optional proxy URL (for UMass GenAI gateway)
        KAN_PRIMARY_MODEL     – default: claude-sonnet-4-6
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        self.model       = model or os.getenv("KAN_PRIMARY_MODEL", "claude-sonnet-4-6")
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running extraction."
            )

        kwargs: dict = {"api_key": key}
        url = base_url or os.getenv("ANTHROPIC_BASE_URL", "")
        if url:
            kwargs["base_url"] = url

        self._client = _anthropic.Anthropic(**kwargs)
        self._usage: list = []

    # ── internal ──────────────────────────────────────────────────────────────

    def _call_once(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self._usage.append({
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": (
                response.usage.input_tokens  * 3.0 +
                response.usage.output_tokens * 15.0
            ) / 1_000_000,
        })
        return response.content[0].text

    # ── LLMClient Protocol ────────────────────────────────────────────────────

    def ask(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return self._call_once(prompt)
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"AnthropicChatClient.ask failed: {exc}") from exc
        return ""  # unreachable

    def ask_batch(self, prompts: List[str]) -> List[str]:
        results = []
        for i, p in enumerate(prompts):
            results.append(self.ask(p))
            if (i + 1) % 10 == 0:
                print(f"  [Anthropic] {i+1}/{len(prompts)} prompts done"
                      f" — cost so far: ${self.total_cost_usd():.4f}")
        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def total_cost_usd(self) -> float:
        return sum(e["cost_usd"] for e in self._usage)

    def usage_summary(self) -> dict:
        return {
            "calls":          len(self._usage),
            "input_tokens":   sum(e["input_tokens"]  for e in self._usage),
            "output_tokens":  sum(e["output_tokens"] for e in self._usage),
            "total_cost_usd": self.total_cost_usd(),
        }
