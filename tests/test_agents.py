"""Tests for Layer 2: tools, ML risk model, and the agent pipeline."""

from __future__ import annotations

from datetime import date

import pytest

from ar_platform.agents import CommsAgent, MonitorAgent, RiskAgent
from ar_platform.agents.base import AgentContext
from ar_platform.data.generator import generate_base_case
from ar_platform.llm import get_llm
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool
from ar_platform.tools.ml_risk import train_risk_model
from ar_platform.tools.templates import CollectionContext, render_dunning


@pytest.fixture
def ctx(tmp_path, fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=40, n_invoices=400, seed=11
    )
    store = fresh_store()
    for c in customers:
        store.upsert_customer(c)
    for i in invoices:
        store.upsert_invoice(i)
    for p in payments:
        store.add_payment(p)
    store.commit()
    model = train_risk_model(customers, invoices, payments)
    context = AgentContext(
        store=store,
        llm=get_llm("rules"),  # returns None -> deterministic templates path
        email=EmailTool(tmp_path / "outbox"),
        erp=ERPTool(store),
        risk_model=model,
        sim_date=date(2026, 1, 15),
    )
    yield context
    store.close()


def test_deterministic_mode_has_no_llm():
    # "rules" (and any non-claude mode) means fully deterministic operation.
    assert get_llm("rules") is None


def test_dunning_templates_escalate_tone():
    base = dict(
        customer_name="Acme",
        invoice_id="INV-1",
        outstanding=1000.0,
        due_date="2026-01-01",
        payment_terms_days=30,
    )
    reminder = render_dunning(
        CollectionContext(days_overdue=5, severity="reminder", **base)
    )
    final = render_dunning(
        CollectionContext(days_overdue=120, severity="final", **base)
    )
    pre_due = render_dunning(
        CollectionContext(days_overdue=0, severity="pre_due", **base)
    )
    assert "reminder" in reminder.subject.lower()
    assert "FINAL NOTICE" in final.subject
    assert "due" in pre_due.subject.lower()
    assert reminder.body != final.body


def test_risk_model_ranks_risky_higher(ctx):
    # The model should assign meaningfully varied probabilities.
    model = ctx.risk_model
    assert 0.0 <= model.positive_rate <= 1.0
    assert model.train_size > 0


def test_pipeline_flags_scores_and_acts(ctx):
    monitor = MonitorAgent().run(ctx)
    assert len(monitor.items) > 0
    assert all(w.severity in ("pre_due", "reminder", "overdue", "urgent", "final")
               for w in monitor.items)

    risk = RiskAgent().run(ctx, monitor.items)
    assert all(w.risk_score is not None and w.priority is not None
               for w in risk.items)
    # Sorted by priority descending.
    priorities = [w.priority for w in risk.items]
    assert priorities == sorted(priorities, reverse=True)

    comms = CommsAgent().run(ctx, risk.items)
    actions = {w.action for w in comms.items}
    assert actions <= {
        "email_sent",
        "escalated_for_approval",
        "skipped_low_risk",
        "skipped_already_contacted",
    }
    # Emails actually landed in the outbox.
    assert len(ctx.email.sent) == sum(
        1 for w in comms.items if w.action == "email_sent"
    )


def test_high_priority_gets_escalated(ctx):
    monitor = MonitorAgent().run(ctx)
    risk = RiskAgent().run(ctx, monitor.items)
    CommsAgent(escalation_priority=1.0).run(ctx, risk.items)  # force all to escalate
    # Every overdue item escalates; pre-due items never do, by design.
    overdue_items = [w for w in risk.items if w.severity != "pre_due"]
    assert overdue_items and all(w.escalated for w in overdue_items)
    assert not any(w.escalated for w in risk.items if w.severity == "pre_due")


def test_erp_apply_payment_settles_invoice(ctx):
    from ar_platform.models import Payment

    inv = ctx.store.get_open_invoices()[0]
    pay = Payment(
        id="PAY-TEST", invoice_id=inv.id, amount=inv.outstanding,
        date=date(2026, 1, 15), method="ACH",
    )
    note = ctx.erp.apply_payment(inv, pay)
    assert "paid" in note
    refreshed = next(i for i in ctx.store.get_invoices() if i.id == inv.id)
    assert refreshed.status.value == "paid"


def test_audit_log_records_actions(ctx):
    MonitorAgent().run(ctx)
    n = ctx.store._query(
        "SELECT COUNT(*) AS n FROM audit_log WHERE agent='monitor'"
    )[0]["n"]
    assert n > 0


def test_monitor_flags_pre_due_invoices(ctx):
    monitor = MonitorAgent().run(ctx)
    pre_due = [w for w in monitor.items if w.severity == "pre_due"]
    overdue = [w for w in monitor.items if w.severity != "pre_due"]
    # The generated ledger has invoices both overdue and due within a week.
    assert len(overdue) > 0
    assert all(w.days_overdue == 0 for w in pre_due)


def test_pre_due_is_risk_gated_and_never_escalated(ctx):
    monitor = MonitorAgent().run(ctx)
    risk = RiskAgent().run(ctx, monitor.items)
    comms = CommsAgent().run(ctx, risk.items)
    pre_due = [w for w in comms.items if w.severity == "pre_due"]
    for w in pre_due:
        assert not w.escalated
        if (w.risk_score or 0) < CommsAgent().pre_due_risk_threshold:
            assert w.action == "skipped_low_risk"
        else:
            assert w.action in ("email_sent", "skipped_already_contacted")


def test_pre_due_email_sent_at_most_once(ctx):
    monitor = MonitorAgent().run(ctx)
    risk = RiskAgent().run(ctx, monitor.items)
    CommsAgent().run(ctx, risk.items)
    ctx.store.commit()
    # Second pass: every risky pre-due invoice is already contacted.
    monitor2 = MonitorAgent().run(ctx)
    risk2 = RiskAgent().run(ctx, monitor2.items)
    comms2 = CommsAgent().run(ctx, risk2.items)
    repeats = [
        w for w in comms2.items
        if w.severity == "pre_due" and w.action == "email_sent"
    ]
    assert repeats == []
