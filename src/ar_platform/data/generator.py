"""Synthetic AR data generator.

There is no real ERP to pull from, so we *synthesize* a believable accounts-
receivable ledger: customers with credit profiles, invoices spread across the
months leading up to the base-case date, and the payments that have settled so
far. Everything is driven by a single RNG seed, so the generated "base case" is
byte-for-byte reproducible on every machine.

Run directly to (re)generate the committed seed CSVs::

    python -m ar_platform.data.generator
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta

from faker import Faker

from ar_platform.config import SEED_DIR, settings
from ar_platform.models import (
    Customer,
    Invoice,
    InvoiceStatus,
    Payment,
    RiskBand,
    Segment,
)

# Segment mix and the parameters that make each segment behave differently.
_SEGMENT_MIX = [
    (Segment.ENTERPRISE, 0.15),
    (Segment.MIDMARKET, 0.35),
    (Segment.SMB, 0.50),
]
_SEGMENT_PARAMS = {
    #                credit_limit range   terms   invoice amount range
    Segment.ENTERPRISE: ((250_000, 1_000_000), 60, (20_000, 120_000)),
    Segment.MIDMARKET: ((60_000, 250_000), 45, (5_000, 40_000)),
    Segment.SMB: ((10_000, 60_000), 30, (500, 8_000)),
}

# Historical lateness (mean days late) by risk band, and the probability that a
# given invoice has actually been paid by "today".
_RISK_PARAMS = {
    #            delay range   base pay-probability
    RiskBand.LOW: ((0, 5), 0.92),
    RiskBand.MEDIUM: ((5, 20), 0.75),
    RiskBand.HIGH: ((20, 55), 0.50),
}
_RISK_MIX = [(RiskBand.LOW, 0.5), (RiskBand.MEDIUM, 0.35), (RiskBand.HIGH, 0.15)]
_PAYMENT_METHODS = ["ACH", "wire", "credit_card", "check"]


def _weighted_choice(rng: random.Random, choices):
    values, weights = zip(*choices, strict=True)
    return rng.choices(values, weights=weights, k=1)[0]


def _generate_customers(rng: random.Random, fake: Faker, n: int) -> list[Customer]:
    customers: list[Customer] = []
    for i in range(n):
        segment = _weighted_choice(rng, _SEGMENT_MIX)
        risk = _weighted_choice(rng, _RISK_MIX)
        (limit_lo, limit_hi), terms, _ = _SEGMENT_PARAMS[segment]
        (delay_lo, delay_hi), _ = _RISK_PARAMS[risk]
        name = fake.company()
        customers.append(
            Customer(
                id=f"CUST-{i + 1:04d}",
                name=name,
                segment=segment,
                email=f"ar@{fake.domain_name()}",
                credit_limit=float(rng.randint(limit_lo // 1000, limit_hi // 1000) * 1000),
                payment_terms_days=terms,
                historical_delay_avg=round(rng.uniform(delay_lo, delay_hi), 1),
                risk_band=risk,
            )
        )
    return customers


def _settle(
    rng: random.Random,
    invoice: Invoice,
    customer: Customer,
    today: date,
) -> Payment | None:
    """Decide whether/how an invoice has been paid as of ``today``.

    Mutates ``invoice.status`` / ``paid_amount`` and returns a Payment if any
    money changed hands. Payment timing keys off the customer's historical
    lateness so the resulting aging distribution looks organic.
    """
    (_, base_prob) = _RISK_PARAMS[customer.risk_band]

    # Expected settle date = due date + historical lateness (with noise).
    lateness = max(0, int(rng.gauss(customer.historical_delay_avg, 6)))
    settle_date = invoice.due_date + timedelta(days=lateness)

    # A small fraction of open invoices get disputed instead of paid.
    if rng.random() < 0.04 and settle_date > today:
        invoice.status = InvoiceStatus.DISPUTED
        return None

    if settle_date <= today and rng.random() < base_prob:
        # Fully paid.
        invoice.status = InvoiceStatus.PAID
        invoice.paid_amount = invoice.amount
        return Payment(
            id=f"PAY-{invoice.id.split('-')[1]}",
            invoice_id=invoice.id,
            amount=invoice.amount,
            date=settle_date,
            method=_weighted_choice(
                rng, [(m, 1.0) for m in _PAYMENT_METHODS]
            ),
        )

    # Late recovery: an invoice that missed its expected settle window often
    # still gets paid eventually (a collector called, cash freed up). Without
    # this, unpaid invoices accumulate forever and the 90+ bucket dominates a
    # long-history ledger; with it, lateness gets a realistic long tail.
    recovery_date = settle_date + timedelta(days=rng.randint(30, 150))
    if recovery_date <= today and rng.random() < base_prob:
        invoice.status = InvoiceStatus.PAID
        invoice.paid_amount = invoice.amount
        return Payment(
            id=f"PAY-{invoice.id.split('-')[1]}",
            invoice_id=invoice.id,
            amount=invoice.amount,
            date=recovery_date,
            method=_weighted_choice(rng, [(m, 1.0) for m in _PAYMENT_METHODS]),
        )

    # Not settled: partial payment sometimes, otherwise open/overdue.
    if invoice.due_date < today and rng.random() < 0.15:
        part = round(invoice.amount * rng.uniform(0.2, 0.7), 2)
        invoice.status = InvoiceStatus.PARTIAL
        invoice.paid_amount = part
        return Payment(
            id=f"PAY-{invoice.id.split('-')[1]}",
            invoice_id=invoice.id,
            amount=part,
            date=min(today, settle_date),
            method=_weighted_choice(rng, [(m, 1.0) for m in _PAYMENT_METHODS]),
        )

    invoice.status = (
        InvoiceStatus.OVERDUE if invoice.due_date < today else InvoiceStatus.OPEN
    )
    return None


def _generate_invoices(
    rng: random.Random,
    customers: list[Customer],
    n: int,
    today: date,
) -> tuple[list[Invoice], list[Payment]]:
    invoices: list[Invoice] = []
    payments: list[Payment] = []
    by_id = {c.id: c for c in customers}

    for i in range(n):
        customer = by_id[rng.choice(list(by_id))]
        _, terms, (amt_lo, amt_hi) = _SEGMENT_PARAMS[customer.segment]

        # Issue dates spread across the history window before "today": deep
        # enough that customers accumulate real payment track records, while
        # still yielding a full spread of aging buckets.
        issue = today - timedelta(days=rng.randint(0, settings.history_days))
        due = issue + timedelta(days=terms)
        amount = round(rng.uniform(amt_lo, amt_hi), 2)

        inv = Invoice(
            id=f"INV-{i + 1:05d}",
            customer_id=customer.id,
            amount=amount,
            issue_date=issue,
            due_date=due,
        )
        pay = _settle(rng, inv, customer, today)
        invoices.append(inv)
        if pay is not None:
            payments.append(pay)

    return invoices, payments


def generate_base_case(
    n_customers: int | None = None,
    n_invoices: int | None = None,
    seed: int | None = None,
    today: date | None = None,
):
    """Generate the full base-case ledger in memory."""
    seed = settings.seed if seed is None else seed
    today = settings.base_case_today if today is None else today
    n_customers = settings.n_customers if n_customers is None else n_customers
    n_invoices = settings.n_invoices if n_invoices is None else n_invoices

    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)

    customers = _generate_customers(rng, fake, n_customers)
    invoices, payments = _generate_invoices(rng, customers, n_invoices, today)
    return customers, invoices, payments


def _write_csv(path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_seed_csvs(customers, invoices, payments) -> None:
    """Persist the base case to the committed ``data/seed/`` CSV files."""
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(SEED_DIR / "customers.csv", [c.to_row() for c in customers])
    _write_csv(SEED_DIR / "invoices.csv", [i.to_row() for i in invoices])
    _write_csv(SEED_DIR / "payments.csv", [p.to_row() for p in payments])


def main() -> None:
    customers, invoices, payments = generate_base_case()
    write_seed_csvs(customers, invoices, payments)
    print(
        f"Base case written to {SEED_DIR}:\n"
        f"  customers: {len(customers)}\n"
        f"  invoices:  {len(invoices)}\n"
        f"  payments:  {len(payments)}"
    )


if __name__ == "__main__":
    main()
