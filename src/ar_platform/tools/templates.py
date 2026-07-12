"""Deterministic dunning-letter templates.

This is intentionally NOT an "AI" component. Collection correspondence at each
severity follows fixed, compliance-reviewed wording — exactly how enterprise AR
departments operate. The optional Claude backend (``AR_LLM_MODE=claude``) can
*replace* these templates with generative drafting for personalization; the
platform's decisions never depend on which one is active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollectionContext:
    """Everything needed to render a collection message for one invoice."""

    customer_name: str
    invoice_id: str
    outstanding: float
    days_overdue: int
    due_date: str
    severity: str          # pre_due | reminder | overdue | urgent | final
    payment_terms_days: int


@dataclass
class EmailDraft:
    subject: str
    body: str


# Severity -> (subject, opening line, closing line). Tone escalates from a
# courteous pre-due nudge to a final notice.
_TEMPLATES = {
    "pre_due": (
        "Upcoming payment: invoice {invoice_id} due {due_date}",
        "We hope all is well. A quick note that invoice {invoice_id} for "
        "${outstanding:,.2f} falls due on {due_date}.",
        "If everything is already scheduled, please disregard this note. If "
        "anything stands in the way of payment, reply and we'll sort it out "
        "together before the due date.",
    ),
    "reminder": (
        "Friendly reminder: invoice {invoice_id}",
        "We hope you're doing well. This is a courtesy reminder that invoice "
        "{invoice_id} for ${outstanding:,.2f} is now due.",
        "If you've already sent payment, please disregard this note. Thank you "
        "for your business.",
    ),
    "overdue": (
        "Overdue: invoice {invoice_id} ({days_overdue} days past due)",
        "Our records show invoice {invoice_id} for ${outstanding:,.2f} is now "
        "{days_overdue} days past due (due {due_date}).",
        "Please arrange payment at your earliest convenience, or reply to let us "
        "know if there is an issue we can help resolve.",
    ),
    "urgent": (
        "URGENT: invoice {invoice_id} is {days_overdue} days overdue",
        "Invoice {invoice_id} for ${outstanding:,.2f} is seriously overdue "
        "({days_overdue} days past the due date of {due_date}).",
        "To avoid further collection action, please remit payment within 7 days "
        "or contact us immediately to arrange a payment plan.",
    ),
    "final": (
        "FINAL NOTICE: invoice {invoice_id}",
        "This is a FINAL NOTICE regarding invoice {invoice_id} for "
        "${outstanding:,.2f}, which is {days_overdue} days overdue.",
        "This account is being prepared for escalation. Immediate payment is "
        "required. Please contact our accounts-receivable team today.",
    ),
}


def render_dunning(ctx: CollectionContext) -> EmailDraft:
    """Render the fixed-wording collection email for a severity level."""
    subject_t, opening_t, closing_t = _TEMPLATES.get(ctx.severity, _TEMPLATES["overdue"])
    fields = {
        "invoice_id": ctx.invoice_id,
        "outstanding": ctx.outstanding,
        "days_overdue": ctx.days_overdue,
        "due_date": ctx.due_date,
    }
    body = (
        f"Dear {ctx.customer_name},\n\n"
        f"{opening_t.format(**fields)}\n\n"
        f"{closing_t.format(**fields)}\n\n"
        "Regards,\n"
        "Accounts Receivable Team"
    )
    return EmailDraft(subject=subject_t.format(**fields), body=body)
