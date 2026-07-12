"""Tests for the data layer: generation is reproducible, store round-trips."""

from __future__ import annotations

from datetime import date

from ar_platform.data.generator import generate_base_case
from ar_platform.models import Invoice, InvoiceStatus


def test_generation_is_reproducible():
    a = generate_base_case(n_customers=20, n_invoices=100, seed=7)
    b = generate_base_case(n_customers=20, n_invoices=100, seed=7)
    # Same seed -> identical customer ids and invoice amounts.
    assert [c.id for c in a[0]] == [c.id for c in b[0]]
    assert [i.amount for i in a[1]] == [i.amount for i in b[1]]


def test_different_seeds_differ():
    a = generate_base_case(n_customers=20, n_invoices=100, seed=1)
    b = generate_base_case(n_customers=20, n_invoices=100, seed=2)
    assert [i.amount for i in a[1]] != [i.amount for i in b[1]]


def test_invoice_aging_buckets():
    inv = Invoice(
        id="INV-1",
        customer_id="CUST-1",
        amount=1000.0,
        issue_date=date(2026, 1, 1),
        due_date=date(2026, 1, 31),
    )
    assert inv.aging_bucket(date(2026, 1, 31)) == "current"
    assert inv.aging_bucket(date(2026, 2, 15)) == "1-30"
    assert inv.aging_bucket(date(2026, 3, 15)) == "31-60"
    assert inv.days_overdue(date(2026, 2, 10)) == 10


def test_outstanding_reflects_partial_payment():
    inv = Invoice(
        id="INV-2",
        customer_id="CUST-1",
        amount=1000.0,
        issue_date=date(2026, 1, 1),
        due_date=date(2026, 1, 31),
        status=InvoiceStatus.PARTIAL,
        paid_amount=400.0,
    )
    assert inv.outstanding == 600.0


def test_store_roundtrip(fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=10, n_invoices=50, seed=3
    )
    store = fresh_store()
    for c in customers:
        store.upsert_customer(c)
    for i in invoices:
        store.upsert_invoice(i)
    store.commit()
    assert len(store.get_customers()) == 10
    assert len(store.get_invoices()) == 50
    # get_customer returns an equal object
    got = store.get_customer(customers[0].id)
    assert got is not None and got.name == customers[0].name
