"""Tests for the ML subsystem: leakage-free features, temporal split, benchmark."""

from __future__ import annotations

from datetime import date

from ar_platform.data.generator import generate_base_case
from ar_platform.ml.benchmark import run_benchmark, temporal_split
from ar_platform.ml.features import (
    FEATURE_NAMES,
    FeatureBuilder,
    build_dataset,
)
from ar_platform.models import Customer, Invoice, InvoiceStatus, Payment, RiskBand, Segment
from ar_platform.tools.ml_risk import train_risk_model


def _cust(cid="CUST-1"):
    return Customer(
        id=cid, name="Test Co", segment=Segment.SMB, email="ar@test.co",
        credit_limit=50_000.0, payment_terms_days=30,
        historical_delay_avg=10.0, risk_band=RiskBand.MEDIUM,
    )


def _inv(iid, cid, issue, amount=1000.0, status=InvoiceStatus.OPEN, paid=0.0):
    return Invoice(
        id=iid, customer_id=cid, amount=amount,
        issue_date=issue, due_date=date(issue.year, issue.month, issue.day),
        status=status, paid_amount=paid,
    )


def test_features_do_not_see_the_future():
    """The feature vector of an early invoice must be identical whether or not
    later invoices/payments exist — the definition of no temporal leakage."""
    cust = _cust()
    early = _inv("INV-1", cust.id, date(2025, 9, 1))
    late = _inv("INV-2", cust.id, date(2025, 12, 1), status=InvoiceStatus.PAID, paid=1000.0)
    late_pay = Payment(
        id="PAY-2", invoice_id="INV-2", amount=1000.0,
        date=date(2025, 12, 20), method="ACH",
    )

    fb_without = FeatureBuilder([cust], [early], [])
    fb_with = FeatureBuilder([cust], [early, late], [late_pay])

    as_of = early.issue_date
    assert fb_without.features_for(early, cust, as_of) == fb_with.features_for(
        early, cust, as_of
    )


def test_features_reflect_prior_history():
    cust = _cust()
    first = _inv("INV-1", cust.id, date(2025, 9, 1), status=InvoiceStatus.PAID, paid=1000.0)
    first.due_date = date(2025, 10, 1)
    pay = Payment(
        id="PAY-1", invoice_id="INV-1", amount=1000.0,
        date=date(2025, 10, 21), method="wire",  # 20 days late
    )
    second = _inv("INV-2", cust.id, date(2025, 11, 15))

    fb = FeatureBuilder([cust], [first, second], [pay])
    x = dict(zip(FEATURE_NAMES, fb.features_for(second, cust, second.issue_date), strict=True))

    assert x["n_prior_invoices"] == 1.0
    assert x["prior_late_mean"] == 20.0
    assert x["prior_settled_ratio"] == 1.0
    assert x["has_history"] == 1.0
    assert x["days_since_last_payment"] == (date(2025, 11, 15) - date(2025, 10, 21)).days


def test_oracle_field_not_in_features():
    assert "historical_delay_avg" not in FEATURE_NAMES


def test_temporal_split_respects_time_order():
    customers, invoices, payments = generate_base_case(
        n_customers=40, n_invoices=400, seed=13
    )
    ds = build_dataset(customers, invoices, payments)
    _, _, _, _, _, split_date = temporal_split(ds)
    n_train = int(round(len(ds) * 0.7))
    assert max(ds.issue_dates[:n_train]) <= min(ds.issue_dates[n_train:])
    assert split_date >= max(ds.issue_dates[:n_train])


def test_benchmark_quick_beats_chance():
    customers, invoices, payments = generate_base_case(
        n_customers=60, n_invoices=600, seed=17
    )
    report = run_benchmark(
        customers, invoices, payments, seed=17, quick=True, save=False
    )
    assert report.winner in ("logistic_regression", "decision_tree")
    tuned = [r for r in report.results if r.model == report.winner][0]
    # Held-out AUC must clearly beat chance on behavioral features alone.
    assert tuned.test_auc > 0.6
    baselines = {r.model for r in report.results}
    assert "baseline_majority" in baselines
    assert "baseline_prior_lateness_rule" in baselines


def test_runtime_model_trains_and_scores():
    customers, invoices, payments = generate_base_case(
        n_customers=30, n_invoices=300, seed=19
    )
    model = train_risk_model(customers, invoices, payments)
    open_inv = next(i for i in invoices if i.is_open())
    cust = next(c for c in customers if c.id == open_inv.customer_id)
    p = model.predict_default_prob(open_inv, cust)
    assert 0.0 <= p <= 1.0
    assert model.train_size > 0
