"""Optional LLM backend.

The platform's default operation is fully deterministic — dunning wording comes
from ``ar_platform.tools.templates`` and every decision is a rule or an ML
score. ``get_llm()`` therefore returns ``None`` unless a real language model is
configured:

* ``AR_LLM_MODE=claude`` -> real Claude API calls (personalized drafting; the
  upcoming agentic layer builds on this)
* anything else          -> ``None`` (deterministic templates; no API key)

This keeps the boundary honest: nothing deterministic masquerades as AI, and
AI is used only where it adds something rules cannot.
"""

from __future__ import annotations

from ar_platform.config import settings
from ar_platform.llm.interface import LLMClient


def get_llm(mode: str | None = None) -> LLMClient | None:
    """Return the configured LLM backend, or None for deterministic operation."""
    mode = (mode or settings.llm_mode).lower()
    if mode == "claude":
        from ar_platform.llm.claude_client import ClaudeLLM

        return ClaudeLLM()
    return None


__all__ = ["LLMClient", "get_llm"]
