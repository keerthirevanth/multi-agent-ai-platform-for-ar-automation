"""Tests that enforce data quality on every push.

These turn the one-off 'is the data any good?' question into CI-enforced
guarantees: the base case must satisfy every structural invariant, and the
generation rules must produce the intended risk->lateness economics.
"""

from __future__ import annotations

from ar_platform.data.generator import generate_base_case
from ar_platform.data.validate import behavioral_report, check_invariants


def test_base_case_has_no_invariant_violations():
    customers, invoices, payments = generate_base_case()
    problems = check_invariants(customers, invoices, payments)
    assert problems == [], f"data invariants violated: {problems[:5]}"


def test_smaller_ledgers_also_valid():
    for seed in (1, 2, 3):
        customers, invoices, payments = generate_base_case(
            n_customers=30, n_invoices=200, seed=seed
        )
        assert check_invariants(customers, invoices, payments) == []


def test_risk_band_predicts_lateness():
    customers, invoices, payments = generate_base_case()
    report = behavioral_report(customers, invoices, payments)
    # The core economic assumption of the whole platform: risk means something.
    assert report["low"] < report["medium"] < report["high"]
