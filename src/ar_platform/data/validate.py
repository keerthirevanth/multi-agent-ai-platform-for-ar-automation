"""Data-quality validation for the AR ledger.

Two kinds of guarantees:

* **Invariants** — structural facts that must *never* be violated (referential
  integrity, paid invoices fully covered, status/date coherence, ...). A single
  violation means the data is corrupt.
* **Behavioral checks** — softer, statistical properties that confirm the
  generation *rules* actually produced the intended economics (e.g. riskier
  customers really do pay later). These validate the model, not just the schema.

Run as a report::

    python -m ar_platform.data.validate
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import mean

from ar_platform.config import settings
from ar_platform.models import Customer, Invoice, InvoiceStatus, Payment

_EPS = 0.01


def check_invariants(
    customers: list[Customer],
    invoices: list[Invoice],
    payments: list[Payment],
    as_of: date | None = None,
) -> list[str]:
    """Return a list of invariant violations (empty list == valid)."""
    as_of = settings.base_case_today if as_of is None else as_of
    cust = {c.id: c for c in customers}
    inv = {i.id: i for i in invoices}
    problems: list[str] = []

    # Referential integrity.
    for i in invoices:
        if i.customer_id not in cust:
            problems.append(f"{i.id}: references unknown customer {i.customer_id}")
    for p in payments:
        if p.invoice_id not in inv:
            problems.append(f"{p.id}: references unknown invoice {p.invoice_id}")

    # Payment coverage vs. status.
    for i in invoices:
        if i.status == InvoiceStatus.PAID and abs(i.paid_amount - i.amount) > _EPS:
            problems.append(f"{i.id}: PAID but paid_amount {i.paid_amount} != amount {i.amount}")
        if i.status == InvoiceStatus.PARTIAL and not (0 < i.paid_amount < i.amount):
            problems.append(f"{i.id}: PARTIAL but paid_amount {i.paid_amount} out of range")

    # Status vs. date coherence.
    for i in invoices:
        past_due = i.due_date < as_of
        if i.status == InvoiceStatus.OVERDUE and not past_due:
            problems.append(f"{i.id}: OVERDUE but not past due")
        if i.status == InvoiceStatus.OPEN and past_due:
            problems.append(f"{i.id}: OPEN but past due (should be OVERDUE)")

    # Structural sanity.
    for i in invoices:
        c = cust.get(i.customer_id)
        if c and (i.due_date - i.issue_date).days != c.payment_terms_days:
            problems.append(f"{i.id}: due-issue span != customer terms")
        if i.amount <= 0 or not (0 <= i.paid_amount <= i.amount + _EPS):
            problems.append(f"{i.id}: amount/paid_amount out of range")

    for p in payments:
        target = inv.get(p.invoice_id)
        if target and p.date < target.issue_date:
            problems.append(f"{p.id}: paid before invoice issued")

    return problems


def behavioral_report(
    customers: list[Customer],
    invoices: list[Invoice],
    payments: list[Payment],
) -> dict[str, float]:
    """Mean days-late of settled payments, grouped by customer risk band."""
    cust = {c.id: c for c in customers}
    inv = {i.id: i for i in invoices}
    late: dict[str, list[int]] = defaultdict(list)
    for p in payments:
        i = inv.get(p.invoice_id)
        if not i:
            continue
        band = cust[i.customer_id].risk_band.value
        late[band].append((p.date - i.due_date).days)
    return {band: round(mean(v), 1) for band, v in late.items() if v}


def main() -> None:
    from ar_platform.data.store import Store

    with Store() as store:
        store.load_base_case()
        customers = store.get_customers()
        invoices = store.get_invoices()
        payments = store.get_payments()

    problems = check_invariants(customers, invoices, payments)
    print(f"Invariant violations: {len(problems)}")
    for p in problems[:20]:
        print(f"  ! {p}")

    print("\nMean days-late by risk band (should increase low -> high):")
    report = behavioral_report(customers, invoices, payments)
    for band in ("low", "medium", "high"):
        if band in report:
            print(f"  {band:7s} {report[band]:6.1f} days")


if __name__ == "__main__":
    main()
