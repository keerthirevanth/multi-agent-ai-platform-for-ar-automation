"""Tests for Phase B: cash application, disputes, credit management, forecast."""

from __future__ import annotations

from datetime import date

import pytest

from ar_platform.agents.base import AgentContext
from ar_platform.agents.cash_app import CashAppAgent
from ar_platform.agents.credit import CreditAgent
from ar_platform.agents.dispute import DisputeAgent, categorize_dispute, resolve_dispute_case
from ar_platform.data.generator import generate_base_case
from ar_platform.models import (
    DisputeStatus,
    InvoiceStatus,
    Remittance,
    RemittanceStatus,
)
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool

AS_OF = date(2026, 1, 15)


@pytest.fixture
def ctx(tmp_path, fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=40, n_invoices=500, seed=51
    )
    store = fresh_store()
    for c in customers:
        store.upsert_customer(c)
    for i in invoices:
        store.upsert_invoice(i)
    for p in payments:
        store.add_payment(p)
    store.commit()
    context = AgentContext(
        store=store, llm=None, email=EmailTool(tmp_path / "outbox"),
        erp=ERPTool(store), risk_model=None, sim_date=AS_OF,
    )
    yield context
    store.close()


def _make_remittance(rid, inv, cust, amount, reference, payer=None):
    return Remittance(
        id=rid, date=AS_OF.isoformat(), payer_name=payer or cust.name,
        amount=amount, reference_text=reference,
        customer_id=cust.id, intended_invoice_id=inv.id,
    )


def _open_invoice(ctx, min_outstanding=100.0):
    return next(
        i for i in ctx.store.get_open_invoices()
        if i.status != InvoiceStatus.DISPUTED and i.outstanding >= min_outstanding
    )


# --- cash application ---------------------------------------------------------
def test_cash_app_matches_by_reference(ctx):
    inv = _open_invoice(ctx)
    cust = ctx.store.get_customer(inv.customer_id)
    ctx.store.add_remittance(
        _make_remittance("REM-T1", inv, cust, inv.outstanding, f"payment {inv.id} thanks")
    )
    report = CashAppAgent().run(ctx)
    assert report.matched == 1 and report.by_method["reference"] == 1
    rem = ctx.store.get_remittances(RemittanceStatus.MATCHED)[0]
    assert rem.matched_invoice_id == inv.id


def test_cash_app_matches_by_exact_amount(ctx):
    inv = _open_invoice(ctx)
    cust = ctx.store.get_customer(inv.customer_id)
    # No usable reference; payer name in shouting case. Amount must identify it.
    ctx.store.add_remittance(
        _make_remittance(
            "REM-T2", inv, cust, inv.outstanding, "payment on account",
            payer=cust.name.upper(),
        )
    )
    report = CashAppAgent().run(ctx)
    matched = ctx.store.get_remittances(RemittanceStatus.MATCHED)
    if report.matched:  # ambiguity possible if two invoices share the amount
        assert matched[0].matched_invoice_id == inv.id
        assert matched[0].match_method in ("amount", "single_open")
    else:
        assert ctx.store.get_remittances(RemittanceStatus.SUSPENSE)


def test_cash_app_unmatchable_goes_to_suspense_and_big_ones_escalate(ctx):
    inv = _open_invoice(ctx)
    cust = ctx.store.get_customer(inv.customer_id)
    ctx.store.add_remittance(
        _make_remittance(
            "REM-T3", inv, cust, 15_000.0, "po 55555", payer="Totally Unknown Payer GmbH"
        )
    )
    report = CashAppAgent().run(ctx)
    assert report.to_suspense == 1
    assert ctx.store.has_open_escalation("REM-T3")  # >= $10K -> human


# --- disputes -------------------------------------------------------------------
def test_dispute_categorization():
    assert categorize_dispute("the shipment was never received") == "delivery"
    assert categorize_dispute("this invoice has the wrong amount") == "billing_error"
    assert categorize_dispute("items arrived damaged and defective") == "quality"
    assert categorize_dispute("we refuse") == "unknown"


def test_dispute_lifecycle_valid_billing_error_full_credit(ctx):
    inv = _open_invoice(ctx)
    cust = ctx.store.get_customer(inv.customer_id)
    dispute = DisputeAgent().open_case(
        ctx, inv, cust, "this invoice has the wrong amount - billing error"
    )
    assert dispute is not None and dispute.reason_category == "billing_error"
    refreshed = next(i for i in ctx.store.get_invoices() if i.id == inv.id)
    assert refreshed.status == InvoiceStatus.DISPUTED

    outcome = resolve_dispute_case(ctx, dispute, valid=True, note="verified")
    assert "credited" in outcome
    final = next(i for i in ctx.store.get_invoices() if i.id == inv.id)
    assert final.status == InvoiceStatus.PAID  # full credit for billing errors
    assert ctx.store.get_disputes(DisputeStatus.RESOLVED_VALID)


def test_dispute_invalid_resumes_dunning_with_email(ctx):
    inv = _open_invoice(ctx)
    cust = ctx.store.get_customer(inv.customer_id)
    dispute = DisputeAgent().open_case(ctx, inv, cust, "shipment never received")
    outcome = resolve_dispute_case(ctx, dispute, valid=False, note="delivery confirmed")
    assert "resumed" in outcome
    final = next(i for i in ctx.store.get_invoices() if i.id == inv.id)
    assert final.status != InvoiceStatus.DISPUTED
    assert any("Dispute review outcome" in e.subject for e in ctx.email.sent)


# --- credit management ------------------------------------------------------------
def test_credit_hold_and_release(ctx):
    # Pick a customer with open exposure and shrink their limit below it.
    inv = _open_invoice(ctx, min_outstanding=500.0)
    cust = ctx.store.get_customer(inv.customer_id)
    cust.credit_limit = inv.outstanding * 0.5  # force utilization > 90%
    ctx.store.upsert_customer(cust)

    held, released = CreditAgent().run(ctx)
    assert held >= 1
    assert cust.id in {h.customer_id for h in ctx.store.active_credit_holds()}

    # Pay everything off -> utilization 0 -> release on next run.
    from ar_platform.models import Payment

    for i in ctx.store.get_open_invoices():
        if i.customer_id == cust.id and i.status != InvoiceStatus.DISPUTED:
            ctx.erp.apply_payment(
                i,
                Payment(id=f"PAY-X{i.id[-5:]}", invoice_id=i.id,
                        amount=i.outstanding, date=AS_OF, method="wire"),
            )
    held2, released2 = CreditAgent().run(ctx)
    assert cust.id not in {h.customer_id for h in ctx.store.active_credit_holds()}


# --- forecast ---------------------------------------------------------------------
def test_forecaster_trains_and_curve_reconciles(ctx):
    from ar_platform.ml.forecast import expected_cash_curve, train_payment_forecaster

    customers = ctx.store.get_customers()
    invoices = ctx.store.get_invoices()
    payments = ctx.store.get_payments()
    fc = train_payment_forecaster(customers, invoices, payments, as_of=AS_OF)
    assert fc.train_size > 0

    curve = expected_cash_curve(fc, customers, invoices, as_of=AS_OF, weeks=8)
    total_curve = sum(w["expected_cash"] for w in curve)
    open_undisputed = sum(
        i.outstanding for i in invoices
        if i.is_open() and i.status != InvoiceStatus.DISPUTED
    )
    # Without risk weighting the curve must account for all open (undisputed) AR.
    assert abs(total_curve - open_undisputed) < 1.0
    assert all(w["expected_cash"] >= 0 for w in curve)
