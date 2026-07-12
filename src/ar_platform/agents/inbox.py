"""Inbox agent — reads inbound customer replies and classifies intent.

This is the first genuinely *agentic* step: the input is unstructured free text
that deterministic rules cannot enumerate. With a real LLM backend it performs
true natural-language understanding; without one it falls back to an explicit
keyword classifier (which works because the reply simulator emits templated
text). Either way the extracted intent + terms drive case-based routing.
"""

from __future__ import annotations

from ar_platform.agents.base import AgentContext
from ar_platform.dialogue import ReplyClassification, classify_reply_rules
from ar_platform.models import CustomerReply


class InboxAgent:
    name = "inbox"

    def classify(self, ctx: AgentContext, reply: CustomerReply) -> ReplyClassification:
        if ctx.llm is not None:
            classification = ctx.llm.classify_reply(reply.text)
        else:
            classification = classify_reply_rules(reply.text)

        ctx.audit(
            self.name,
            "classified_reply",
            "reply",
            reply.id,
            detail=f"intent={classification.intent}",
            metadata={
                "intent": classification.intent,
                "extension_days": classification.extension_days,
                "backend": ctx.llm.name if ctx.llm else "rules",
            },
        )
        return classification
