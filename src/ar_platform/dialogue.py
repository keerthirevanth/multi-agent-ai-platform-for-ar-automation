"""Dialogue reasoning: reply classification types + the negotiation policy.

This module holds the pieces that are shared between the deterministic backend
and the real-LLM backend:

* the data structures they both produce (``ReplyClassification``,
  ``NegotiationProposal``);
* the deterministic **guardrail** — ``NegotiationPolicy`` — which encodes the
  business rules for how much leniency each customer may receive and validates
  any proposal (whether from rules or an LLM) against those bounds.

The design contract of the agentic layer lives here: **the LLM may propose,
but the policy disposes.** An LLM can suggest an extension or plan, but if the
suggestion exceeds the customer's risk-tiered authority it is clamped, and
anything beyond the department's authority is routed to a human. This keeps a
non-deterministic reasoner safe and auditable inside a finance system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

from ar_platform.models import RiskBand


# --- reply classification ----------------------------------------------------
@dataclass
class ReplyClassification:
    """The Inbox agent's read of a customer reply."""

    intent: str                        # a models.ReplyIntent value
    extension_days: int | None = None  # requested days, if the reply names any
    amount: float | None = None        # extracted, if the reply names one
    rationale: str = ""


# --- negotiation -------------------------------------------------------------
class NegotiationAction(StrEnum):
    ACCEPT = "accept"       # grant the request as-is
    COUNTER = "counter"     # grant a policy-bounded alternative
    REJECT = "reject"       # decline (e.g. repeated broken promises)
    ESCALATE = "escalate"   # beyond authority -> human


@dataclass
class NegotiationTerms:
    """Concrete terms of a negotiated outcome."""

    extension_days: int = 0
    new_due_date: date | None = None
    promise_amount: float = 0.0


@dataclass
class NegotiationProposal:
    action: NegotiationAction
    terms: NegotiationTerms = field(default_factory=NegotiationTerms)
    rationale: str = ""


@dataclass
class NegotiationCase:
    """Everything needed to reason about (and bound) one negotiation."""

    invoice_id: str
    customer_id: str
    risk_band: RiskBand
    outstanding: float
    days_overdue: int
    requested_extension_days: int      # from the customer's reply (0 if unknown)
    prior_broken_promises: int
    as_of: date


# Risk-tiered authority. These are the business rules a real AR policy encodes;
# they are deterministic and auditable, and they bound whatever any reasoner
# (rules or LLM) is allowed to grant.
_MAX_EXTENSION_DAYS = {RiskBand.LOW: 30, RiskBand.MEDIUM: 14, RiskBand.HIGH: 7}
# Above this exposure, even an in-bounds concession needs human sign-off.
AUTO_APPROVE_EXPOSURE_CEILING = 25_000.0
# Default extension to offer when the customer names no specific date.
_DEFAULT_REQUEST_DAYS = 21


class NegotiationPolicy:
    """Deterministic guardrail: bounds + validation for negotiated concessions."""

    def __init__(
        self,
        exposure_ceiling: float = AUTO_APPROVE_EXPOSURE_CEILING,
        max_extension_days: dict[RiskBand, int] | None = None,
    ):
        self.exposure_ceiling = exposure_ceiling
        self.max_extension_days = max_extension_days or dict(_MAX_EXTENSION_DAYS)

    def max_days_for(self, risk_band: RiskBand) -> int:
        return self.max_extension_days[risk_band]

    def decide(self, case: NegotiationCase) -> NegotiationProposal:
        """The deterministic decision — also the reference the LLM is bounded to."""
        # Repeated broken promises -> no more self-service concessions.
        if case.prior_broken_promises >= 1:
            return NegotiationProposal(
                NegotiationAction.ESCALATE,
                rationale=f"{case.prior_broken_promises} prior broken promise(s)",
            )
        # Large exposure -> human must sign off even if terms are in-bounds.
        if case.outstanding > self.exposure_ceiling:
            return NegotiationProposal(
                NegotiationAction.ESCALATE,
                rationale=f"exposure ${case.outstanding:,.0f} exceeds auto-approve ceiling",
            )

        requested = case.requested_extension_days or _DEFAULT_REQUEST_DAYS
        allowed = self.max_days_for(case.risk_band)
        if requested <= allowed:
            days = requested
            action = NegotiationAction.ACCEPT
            rationale = f"granted {days}d (within {case.risk_band.value} limit {allowed}d)"
        else:
            days = allowed
            action = NegotiationAction.COUNTER
            rationale = (
                f"countered {days}d (requested {requested}d exceeds "
                f"{case.risk_band.value} limit {allowed}d)"
            )
        new_due = case.as_of + timedelta(days=days)
        return NegotiationProposal(
            action=action,
            terms=NegotiationTerms(
                extension_days=days,
                new_due_date=new_due,
                promise_amount=case.outstanding,
            ),
            rationale=rationale,
        )

    def validate(self, case: NegotiationCase, proposal: NegotiationProposal) -> NegotiationProposal:
        """Clamp an externally-produced (e.g. LLM) proposal into policy bounds.

        ACCEPT/COUNTER granting more than the risk-tiered maximum is clamped down
        to the maximum (and relabelled COUNTER). Cases the policy would escalate
        are escalated regardless of what the proposal said — the LLM can never
        widen its own authority.
        """
        reference = self.decide(case)
        if reference.action == NegotiationAction.ESCALATE:
            return reference
        if proposal.action in (NegotiationAction.ACCEPT, NegotiationAction.COUNTER):
            allowed = self.max_days_for(case.risk_band)
            days = min(proposal.terms.extension_days or 0, allowed)
            if days <= 0:
                return reference
            action = (
                NegotiationAction.ACCEPT
                if days >= (case.requested_extension_days or _DEFAULT_REQUEST_DAYS)
                else NegotiationAction.COUNTER
            )
            return NegotiationProposal(
                action=action,
                terms=NegotiationTerms(
                    extension_days=days,
                    new_due_date=case.as_of + timedelta(days=days),
                    promise_amount=case.outstanding,
                ),
                rationale=f"{proposal.rationale} (policy-bounded to {days}d)".strip(),
            )
        # REJECT / ESCALATE proposals pass through.
        return proposal


# --- deterministic reply classifier -----------------------------------------
# Keyword rules. This is an explicit deterministic tool, NOT a masquerading LLM:
# it works because the reply simulator emits templated text. The Claude backend
# is what handles genuine free-text variation.
_INTENT_KEYWORDS = [
    ("already_paid",
     ("already paid", "payment was sent", "we paid", "paid this", "remittance sent")),
    ("dispute",
     ("dispute", "never received", "incorrect", "wrong amount", "not our", "billing error")),
    ("extension_request",
     ("extension", "more time", "installment", "payment plan", "pay in", "part now")),
    ("promise_to_pay",
     ("will pay", "we'll pay", "expect to pay", "by end of", "settle by", "pay by")),
    ("info_request",
     ("resend", "copy of", "send the invoice", "details", "purchase order")),
]


def _extract_days(text: str) -> int | None:
    m = re.search(r"(\d+)\s*day", text.lower())
    return int(m.group(1)) if m else None


def classify_reply_rules(text: str) -> ReplyClassification:
    """Deterministic keyword classification of a customer reply."""
    low = text.lower()
    for intent, keywords in _INTENT_KEYWORDS:
        if any(k in low for k in keywords):
            return ReplyClassification(
                intent=intent,
                extension_days=_extract_days(text),
                rationale=f"matched '{intent}' keywords",
            )
    return ReplyClassification(intent="other", rationale="no keyword match")
