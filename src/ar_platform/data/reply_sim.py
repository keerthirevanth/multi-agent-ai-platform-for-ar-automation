"""Customer reply simulation — the inbound half of the conversation.

Real AR is a dialogue: after we dun a customer, some write back. This module is
the *environment* generating those free-text replies so the Inbox and
Negotiator agents have something genuine to reason over. It is intentionally
part of the simulated world (the "customer"), not part of the department's
agents.

Replies are templated but written as natural sentences (so a keyword classifier
can handle them and a real LLM has realistic text to parse), and the intent mix
is risk-driven — riskier customers ask for more time or dispute more often,
lower-risk ones occasionally note they've already paid. Fully seeded.
"""

from __future__ import annotations

import random
from datetime import date

from ar_platform.models import Customer, CustomerReply, Invoice, RiskBand

# Probability a contacted, still-open invoice draws a reply (once in its life).
REPLY_PROB = 0.30

# Intent mix by risk band: (intent, weight). Higher-risk customers negotiate and
# dispute more; lower-risk ones mostly note payment or ask for a copy.
_INTENT_MIX = {
    RiskBand.LOW: [
        ("already_paid", 0.45), ("info_request", 0.20),
        ("promise_to_pay", 0.20), ("extension_request", 0.10), ("dispute", 0.05),
    ],
    RiskBand.MEDIUM: [
        ("promise_to_pay", 0.30), ("extension_request", 0.30),
        ("already_paid", 0.15), ("dispute", 0.15), ("info_request", 0.10),
    ],
    RiskBand.HIGH: [
        ("extension_request", 0.40), ("promise_to_pay", 0.25),
        ("dispute", 0.25), ("info_request", 0.05), ("already_paid", 0.05),
    ],
}

# Requested extension lengths (days) — some deliberately exceed policy limits so
# the Negotiator has to counter or escalate.
_EXTENSION_CHOICES = [14, 21, 30, 45, 60, 90]

_TEMPLATES = {
    "already_paid": [
        "Hello, our records show we already paid invoice {inv} last week. "
        "Please check your remittance and confirm.",
        "We believe payment was sent for {inv}. Kindly verify on your end.",
    ],
    "info_request": [
        "Could you resend a copy of invoice {inv}? We can't locate the details.",
        "Please send the invoice {inv} and the related purchase order so we can process it.",
    ],
    "promise_to_pay": [
        "Apologies for the delay on {inv}. We will pay the full amount by end of {days} days.",
        "We expect to settle invoice {inv} within {days} days — please bear with us.",
    ],
    "extension_request": [
        "Cash flow is tight this month. Could we have a {days} day extension on {inv}?",
        "We'd like to arrange a payment plan for {inv} — can we pay over {days} days?",
    ],
    "dispute": [
        "We are disputing invoice {inv}: the March shipment was never received.",
        "Invoice {inv} appears to have the wrong amount — this is a billing error on your side.",
    ],
}


def _weighted(rng: random.Random, choices):
    values, weights = zip(*choices, strict=True)
    return rng.choices(values, weights=weights, k=1)[0]


def generate_reply(
    rng: random.Random,
    inv: Invoice,
    cust: Customer,
    sim_date: date,
    seq: int,
) -> CustomerReply:
    """Produce one templated reply for an invoice."""
    intent = _weighted(rng, _INTENT_MIX[cust.risk_band])
    days = _weighted(rng, [(d, 1.0) for d in _EXTENSION_CHOICES])
    template = rng.choice(_TEMPLATES[intent])
    text = template.format(inv=inv.id, days=days)
    return CustomerReply(
        id=f"RLY-{seq:06d}",
        invoice_id=inv.id,
        customer_id=cust.id,
        sim_date=sim_date.isoformat(),
        text=text,
    )
