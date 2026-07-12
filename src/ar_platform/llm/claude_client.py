"""Real Claude API backend (opt-in via AR_LLM_MODE=claude).

Imports of the ``anthropic`` SDK are deferred to construction time so the whole
platform still runs deterministically even if the SDK or an API key is absent.
Currently provides personalized email drafting; the agentic layer (reply
understanding, negotiation within business-rule bounds) will extend this class.
"""

from __future__ import annotations

import json
from datetime import timedelta

from ar_platform.config import settings
from ar_platform.dialogue import (
    NegotiationAction,
    NegotiationCase,
    NegotiationProposal,
    NegotiationTerms,
    ReplyClassification,
)
from ar_platform.llm.interface import LLMClient
from ar_platform.tools.templates import CollectionContext, EmailDraft

_EMAIL_SYSTEM = (
    "You are a professional accounts-receivable specialist at an enterprise "
    "finance department. Write concise, courteous, and firm collection emails. "
    "Match the requested severity. Return ONLY JSON: {\"subject\": ..., "
    "\"body\": ...}."
)


class ClaudeLLM(LLMClient):
    name = "claude"

    def __init__(self):
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "AR_LLM_MODE=claude requires ANTHROPIC_API_KEY to be set."
            )
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'anthropic' package is required for AR_LLM_MODE=claude. "
                "Install it with: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    def _complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def draft_collection_email(self, ctx: CollectionContext) -> EmailDraft:
        prompt = (
            f"Customer: {ctx.customer_name}\n"
            f"Invoice: {ctx.invoice_id}\n"
            f"Amount outstanding: ${ctx.outstanding:,.2f}\n"
            f"Days overdue: {ctx.days_overdue}\n"
            f"Original due date: {ctx.due_date}\n"
            f"Payment terms: net-{ctx.payment_terms_days}\n"
            f"Requested severity: {ctx.severity}\n\n"
            "Write the collection email."
        )
        raw = self._complete(_EMAIL_SYSTEM, prompt)
        try:
            data = json.loads(raw)
            return EmailDraft(subject=data["subject"], body=data["body"])
        except (json.JSONDecodeError, KeyError):
            # Fall back to using the raw text as the body if parsing fails.
            return EmailDraft(
                subject=f"Regarding invoice {ctx.invoice_id}", body=raw.strip()
            )

    def classify_reply(self, text: str) -> ReplyClassification:
        system = (
            "You classify a customer's reply to a collection email. Intents: "
            "promise_to_pay, extension_request, dispute, already_paid, "
            "info_request, other. Extract a requested extension in days if any. "
            "Return ONLY JSON: {\"intent\": ..., \"extension_days\": int|null}."
        )
        raw = self._complete(system, f"Customer reply:\n{text}", max_tokens=200)
        try:
            data = json.loads(raw)
            return ReplyClassification(
                intent=data["intent"],
                extension_days=data.get("extension_days"),
                rationale="claude classification",
            )
        except (json.JSONDecodeError, KeyError):
            from ar_platform.dialogue import classify_reply_rules

            return classify_reply_rules(text)

    def propose_negotiation(self, case: NegotiationCase) -> NegotiationProposal:
        from ar_platform.dialogue import NegotiationPolicy

        allowed = NegotiationPolicy().max_days_for(case.risk_band)
        system = (
            "You are an AR collections negotiator. Propose one action: accept, "
            "counter, reject, or escalate. You may grant an extension up to the "
            "stated policy limit; propose escalate for anything beyond it. "
            "Return ONLY JSON: {\"action\": ..., \"extension_days\": int, "
            "\"rationale\": ...}."
        )
        prompt = (
            f"Invoice {case.invoice_id}: ${case.outstanding:,.2f} outstanding, "
            f"{case.days_overdue} days overdue.\n"
            f"Customer risk band: {case.risk_band.value} "
            f"(policy extension limit {allowed} days).\n"
            f"Prior broken promises: {case.prior_broken_promises}.\n"
            f"Customer requested: {case.requested_extension_days or 'unspecified'} days.\n"
            "Decide."
        )
        raw = self._complete(system, prompt, max_tokens=250)
        try:
            data = json.loads(raw)
            action = NegotiationAction(data["action"])
            days = int(data.get("extension_days", 0) or 0)
            return NegotiationProposal(
                action=action,
                terms=NegotiationTerms(
                    extension_days=days,
                    new_due_date=case.as_of + timedelta(days=days),
                    promise_amount=case.outstanding,
                ),
                rationale=data.get("rationale", "claude proposal"),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return NegotiationPolicy().decide(case)
