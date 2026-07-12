"""Tests for the A/B uplift experiment."""

from __future__ import annotations

from ar_platform.data.generator import generate_base_case
from ar_platform.experiments.uplift import run_arm, summarize


def _seed_store(fresh_store):
    customers, invoices, payments = generate_base_case(
        n_customers=40, n_invoices=400, seed=31
    )
    store = fresh_store()
    for c in customers:
        store.upsert_customer(c)
    for i in invoices:
        store.upsert_invoice(i)
    for p in payments:
        store.add_payment(p)
    store.commit()
    return store, customers, invoices, payments


def test_control_arm_never_acts(fresh_store, tmp_path):
    store, *_ = _seed_store(fresh_store)
    off = run_arm(
        store, seed=7, ticks=2, days_per_tick=7,
        agents_enabled=False, outbox_dir=tmp_path / "out_off",
    )
    assert off.agents == "off"
    assert off.emails_sent == 0
    assert off.escalations == 0
    assert off.collected > 0  # the world still pays at baseline rates


def test_treatment_arm_acts(fresh_store, tmp_path):
    store, customers, invoices, payments = _seed_store(fresh_store)
    on = run_arm(
        store, seed=7, ticks=2, days_per_tick=7,
        agents_enabled=True, outbox_dir=tmp_path / "out_on",
    )
    assert on.agents == "on"
    assert on.emails_sent > 0
    assert on.escalations > 0


def test_arms_are_reproducible(fresh_store, tmp_path):
    def once(tag):
        store, *_ = _seed_store(fresh_store)
        return run_arm(
            store, seed=11, ticks=2, days_per_tick=7,
            agents_enabled=True, outbox_dir=tmp_path / f"out_{tag}",
        )

    a, b = once("a"), once("b")
    assert (a.collected, a.end_open_ar, a.end_dso) == (
        b.collected, b.end_open_ar, b.end_dso
    )


def test_summarize_pairs_by_seed():
    from ar_platform.experiments.uplift import ArmResult

    rows = [
        ArmResult(1, "off", 2, 7, 100.0, 80.0, 40.0, 30.0, 20.0, 0, 0),
        ArmResult(1, "on", 2, 7, 100.0, 70.0, 30.0, 25.0, 32.0, 5, 1),
        ArmResult(2, "off", 2, 7, 100.0, 82.0, 42.0, 31.0, 18.0, 0, 0),
        ArmResult(2, "on", 2, 7, 100.0, 71.0, 33.0, 26.0, 30.0, 6, 2),
    ]
    s = summarize(rows)
    assert s["collected"]["mean"] == 12.0
    assert s["end_open_ar"]["mean"] == -10.5
    assert s["end_dso"]["mean"] == -5.0
