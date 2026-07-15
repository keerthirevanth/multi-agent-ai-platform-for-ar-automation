"""Dispute agent — formal dispute/deduction management.

When a customer contests an invoice, the agent categorizes the claim from the
reply text, opens a tracked dispute case, puts dunning on hold, and routes the
case to the human queue. Resolution is a human decision (approve = claim valid,
reject = claim invalid), applied by ``resolve_dispute_case``:

* **valid** — a credit memo adjusts the books: full credit for billing errors,
  partial (50%) for delivery/quality claims where some liability usually
  remains; any remainder returns to normal dunning.
* **invalid** — the hold lifts, the customer gets an explanation email, and
  dunning resumes where it left off.
"""

from __future__ import annotations

import uuid

from ar_platform.agents.base import AgentContext
from ar_platform.models import (
    Customer,
    Dispute,
    DisputeStatus,
    Invoice,
    InvoiceStatus,
    Payment,
)

# Claim category -> share of the outstanding amount credited when upheld.
_CREDIT_RATIO = {"billing_error": 1.0, "delivery": 0.5, "quality": 0.5, "unknown": 0.5}

_CATEGORY_KEYWORDS = [
    ("delivery", ("never received", "not received", "shipment", "not delivered")),
    ("billing_error", ("wrong amount", "billing error", "incorrect", "overcharged")),
    ("quality", ("defective", "damaged", "quality", "faulty")),
]


def categorize_dispute(text: str) -> str:
    low = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(k in low for k in keywords):
            return category
    return "unknown"


class DisputeAgent:
    name = "dispute"

    def open_case(
        self, ctx: AgentContext, inv: Invoice, cust: Customer, reason_text: str
    ) -> Dispute | None:
        """Open a dispute for an invoice (idempotent: one open case per invoice)."""
        if ctx.store.get_open_dispute(inv.id) is not None:
            return None
        category = categorize_dispute(reason_text)
        dispute = Dispute(
            id=f"DSP-{uuid.uuid4().hex[:8]}",
            invoice_id=inv.id,
            customer_id=cust.id,
            opened_date=ctx.sim_date.isoformat(),
            reason_category=category,
            reason_text=reason_text[:1000],
        )
        ctx.store.add_dispute(dispute)
        inv.status = InvoiceStatus.DISPUTED
        ctx.store.upsert_invoice(inv)
        ctx.audit(
            self.name, "opened_dispute", "invoice", inv.id,
            detail=f"category={category}",
            metadata={"dispute_id": dispute.id, "category": category},
        )
        return dispute


def _post_dispute_status(inv: Invoice, as_of) -> InvoiceStatus:
    """Where an invoice returns after its dispute hold lifts."""
    return InvoiceStatus.OVERDUE if inv.due_date < as_of else InvoiceStatus.OPEN


def resolve_dispute_case(
    ctx: AgentContext, dispute: Dispute, valid: bool, note: str = ""
) -> str:
    """Apply a human's dispute verdict (called from escalation resolution)."""
    inv = next(
        (i for i in ctx.store.get_invoices() if i.id == dispute.invoice_id), None
    )
    if inv is None:
        return "stale"

    if valid:
        ratio = _CREDIT_RATIO.get(dispute.reason_category, 0.5)
        credit = round(inv.outstanding * ratio, 2)
        if credit > 0:
            ctx.erp.apply_payment(
                inv,
                Payment(
                    id=f"PAY-C{uuid.uuid4().hex[:6]}",
                    invoice_id=inv.id,
                    amount=credit,
                    date=ctx.sim_date,
                    method="credit_memo",
                ),
            )
        # Whatever wasn't credited goes back to normal collection.
        if inv.outstanding > 0.01 and inv.status == InvoiceStatus.DISPUTED:
            inv.status = _post_dispute_status(inv, ctx.sim_date)
            ctx.store.upsert_invoice(inv)
        ctx.store.resolve_dispute(
            dispute.id, DisputeStatus.RESOLVED_VALID, ctx.sim_date.isoformat(), note
        )
        outcome = f"valid: credited ${credit:,.2f} ({dispute.reason_category})"
    else:
        if inv.status == InvoiceStatus.DISPUTED:
            inv.status = _post_dispute_status(inv, ctx.sim_date)
            ctx.store.upsert_invoice(inv)
        cust = ctx.store.get_customer(dispute.customer_id)
        if cust:
            ctx.email.send(
                to=cust.email,
                subject=f"Dispute review outcome: invoice {inv.id}",
                body=(
                    f"Dear {cust.name},\n\nWe have completed our review of your "
                    f"dispute on invoice {inv.id}. Our records support the "
                    "original charge, so the invoice remains payable. Please "
                    "arrange payment or contact us to discuss.\n\nRegards,\n"
                    "Accounts Receivable Team"
                ),
                invoice_id=inv.id,
                sim_date=ctx.sim_date.isoformat(),
            )
        ctx.store.resolve_dispute(
            dispute.id, DisputeStatus.RESOLVED_INVALID, ctx.sim_date.isoformat(), note
        )
        outcome = "invalid: dunning resumed"

    ctx.audit(
        "dispute", "resolved_dispute", "invoice", inv.id,
        detail=outcome, metadata={"dispute_id": dispute.id, "valid": valid},
    )
    return outcome
