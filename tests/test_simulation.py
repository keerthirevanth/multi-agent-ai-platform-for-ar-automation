"""Tests for Layer 3: orchestrator, escalation queue, and the simulation clock."""

from __future__ import annotations

from datetime import timedelta

import pytest

from ar_platform.data.generator import generate_base_case
from ar_platform.models import EscalationStatus
from ar_platform.simulation import Simulation


@pytest.fixture
def store(fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=60, n_invoices=600, seed=21
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


def test_tick_advances_clock(store):
    sim = Simulation(store, seed=1)
    start = sim.sim_date
    report = sim.tick(days=7)
    assert sim.sim_date == start + timedelta(days=7)
    assert report.sim_date == sim.sim_date.isoformat()


def test_simulation_reduces_open_ar(store):
    sim = Simulation(store, seed=1)
    start_ar = sum(i.outstanding for i in store.get_open_invoices())
    reports = sim.run(ticks=6, days_per_tick=7)
    end_ar = reports[-1].open_ar
    # Collections should meaningfully reduce outstanding AR.
    assert end_ar < start_ar
    assert sum(r.payments_amount for r in reports) > 0


def test_escalations_are_filed_and_resolved(store):
    sim = Simulation(store, seed=1, auto_approve_rate=0.5)
    sim.tick(days=7)
    all_esc = store.get_escalations()
    assert len(all_esc) > 0
    # With an auto-approve policy, the queue gets resolved (none left pending
    # from the first tick after the clear step).
    resolved = [
        e for e in all_esc
        if e.status in (EscalationStatus.APPROVED, EscalationStatus.REJECTED)
    ]
    assert len(resolved) > 0


def test_no_duplicate_pending_escalation(store):
    sim = Simulation(store, seed=1, auto_approve_rate=None)  # leave pending
    sim.tick(days=1)
    sim.tick(days=1)
    invoice_ids = [e.invoice_id for e in store.get_escalations(EscalationStatus.PENDING)]
    # No invoice should have two open escalations at once.
    assert len(invoice_ids) == len(set(invoice_ids))


def test_simulation_is_reproducible(fresh_store):
    def run_once():
        customers, invoices, payments = generate_base_case(
            n_customers=40, n_invoices=300, seed=5
        )
        s = fresh_store()
        for c in customers:
            s.upsert_customer(c)
        for i in invoices:
            s.upsert_invoice(i)
        for p in payments:
            s.add_payment(p)
        s.commit()
        sim = Simulation(s, seed=99)
        return [r.open_ar for r in sim.run(ticks=4, days_per_tick=7)]

    assert run_once() == run_once()
