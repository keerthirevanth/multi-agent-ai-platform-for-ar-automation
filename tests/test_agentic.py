"""Tests for the agentic layer: classification, negotiation guardrails, loop."""

from __future__ import annotations

from datetime import date

import pytest

from ar_platform.agents.base import AgentContext
from ar_platform.agents.negotiator import NegotiatorAgent
from ar_platform.data.generator import generate_base_case
from ar_platform.dialogue import (
    NegotiationAction,
    NegotiationCase,
    NegotiationPolicy,
    NegotiationProposal,
    NegotiationTerms,
    classify_reply_rules,
)
from ar_platform.llm import get_llm
from ar_platform.models import (
    InvoiceStatus,
    RiskBand,
)
from ar_platform.simulation import Simulation
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool
from ar_platform.tools.ml_risk import train_risk_model


# --- classification ----------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("We already paid invoice INV-1 last week", "already_paid"),
        ("Could we have a 30 day extension?", "extension_request"),
        ("We are disputing this: never received the shipment", "dispute"),
        ("We will pay by end of 21 days", "promise_to_pay"),
        ("Please resend a copy of the invoice", "info_request"),
        ("hello", "other"),
    ],
)
def test_reply_classifier(text, expected):
    assert classify_reply_rules(text).intent == expected


def test_classifier_extracts_extension_days():
    assert classify_reply_rules("can we get a 45 day extension").extension_days == 45


# --- negotiation policy (the guardrail) --------------------------------------
def _case(risk, requested, outstanding=5000.0, broken=0):
    return NegotiationCase(
        invoice_id="INV-1", customer_id="C1", risk_band=risk,
        outstanding=outstanding, days_overdue=20,
        requested_extension_days=requested, prior_broken_promises=broken,
        as_of=date(2026, 1, 15),
    )


def test_policy_accepts_within_limit():
    p = NegotiationPolicy().decide(_case(RiskBand.LOW, 30))
    assert p.action == NegotiationAction.ACCEPT and p.terms.extension_days == 30


def test_policy_counters_beyond_limit():
    p = NegotiationPolicy().decide(_case(RiskBand.HIGH, 30))
    assert p.action == NegotiationAction.COUNTER and p.terms.extension_days == 7


def test_policy_escalates_large_exposure():
    p = NegotiationPolicy().decide(_case(RiskBand.LOW, 10, outstanding=40_000))
    assert p.action == NegotiationAction.ESCALATE


def test_policy_escalates_repeat_broken_promises():
    p = NegotiationPolicy().decide(_case(RiskBand.LOW, 10, broken=1))
    assert p.action == NegotiationAction.ESCALATE


def test_policy_clamps_out_of_bounds_llm_proposal():
    # An LLM proposing a 90-day grant to a high-risk customer gets clamped.
    case = _case(RiskBand.HIGH, 90)
    rogue = NegotiationProposal(
        NegotiationAction.ACCEPT, NegotiationTerms(extension_days=90), "too generous"
    )
    bounded = NegotiationPolicy().validate(case, rogue)
    assert bounded.terms.extension_days <= 7


# --- the loop end-to-end -----------------------------------------------------
@pytest.fixture
def store(fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=50, n_invoices=800, seed=41
    )
    s = fresh_store()
    for c in customers:
        s.upsert_customer(c)
    for i in invoices:
        s.upsert_invoice(i)
    for p in payments:
        s.add_payment(p)
    s.commit()
    return s


def test_negotiator_creates_promise_and_extends_due(store):
    inv = next(i for i in store.get_open_invoices() if i.status != InvoiceStatus.DISPUTED)
    cust = store.get_customer(inv.customer_id)
    model = train_risk_model(
        store.get_customers(), store.get_invoices(), store.get_payments()
    )
    ctx = AgentContext(
        store=store, llm=None, email=EmailTool(),
        erp=ERPTool(store), risk_model=model, sim_date=date(2026, 1, 15),
    )
    from ar_platform.dialogue import ReplyClassification

    old_due = inv.due_date
    action = NegotiatorAgent().handle(
        ctx, inv, cust, ReplyClassification(intent="extension_request", extension_days=14)
    )
    store.commit()
    if action.startswith(("accept", "counter")):
        assert store.has_active_promise(inv.id)
        refreshed = next(i for i in store.get_invoices() if i.id == inv.id)
        assert refreshed.due_date > old_due
    else:
        assert action in ("escalated", "rejected")


def test_full_agentic_loop_engages(store):
    sim = Simulation(store, seed=3)
    sim.run(ticks=4, days_per_tick=7)
    replies = store.get_replies()
    assert len(replies) > 0
    assert all(r.status == "handled" for r in replies)
    # Some negotiation should have produced promises.
    promises = store.get_promises()
    assert len(promises) > 0


def test_disputed_invoice_not_chased(store):
    # Force a dispute and confirm Monitor skips it.
    from ar_platform.agents.monitor import MonitorAgent

    inv = next(i for i in store.get_open_invoices() if i.days_overdue(date(2026, 1, 15)) > 0)
    inv.status = InvoiceStatus.DISPUTED
    store.upsert_invoice(inv)
    store.commit()
    model = train_risk_model(
        store.get_customers(), store.get_invoices(), store.get_payments()
    )
    ctx = AgentContext(
        store=store, llm=None, email=EmailTool(),
        erp=ERPTool(store), risk_model=model, sim_date=date(2026, 1, 15),
    )
    flagged = MonitorAgent().run(ctx).items
    assert inv.id not in {w.invoice_id for w in flagged}


def test_deterministic_mode_still_has_no_llm():
    assert get_llm("rules") is None
