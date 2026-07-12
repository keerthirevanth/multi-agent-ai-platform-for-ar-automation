"""Tests for the KPI engine."""

from __future__ import annotations

from datetime import date

from ar_platform.data.generator import generate_base_case
from ar_platform.kpis import compute_kpis
from ar_platform.models import AGING_BUCKETS
from ar_platform.tools.ml_risk import train_risk_model


def test_kpis_basic_shape():
    customers, invoices, payments = generate_base_case(
        n_customers=50, n_invoices=500, seed=8
    )
    kpis = compute_kpis(customers, invoices, as_of=date(2026, 1, 15))
    assert kpis.open_ar > 0
    assert kpis.dso > 0
    assert 0 <= kpis.overdue_pct <= 1
    assert 0 <= kpis.collection_rate <= 1
    # Aging buckets sum to open AR.
    assert abs(sum(kpis.aging.values()) - kpis.open_ar) < 1.0
    assert set(kpis.aging) == set(AGING_BUCKETS)


def test_expected_cashflow_with_model():
    customers, invoices, payments = generate_base_case(
        n_customers=50, n_invoices=500, seed=8
    )
    model = train_risk_model(customers, invoices, payments)
    kpis = compute_kpis(customers, invoices, as_of=date(2026, 1, 15), risk_model=model)
    # collectible + loss should reconcile to open AR (both derived from it).
    assert abs((kpis.expected_collectible + kpis.expected_loss) - kpis.open_ar) < 1.0
    assert kpis.expected_collectible > kpis.expected_loss  # most AR is recoverable


def test_dso_zero_when_no_recent_sales():
    # No invoices in the trailing window -> DSO defined as 0.
    customers, invoices, _ = generate_base_case(
        n_customers=10, n_invoices=50, seed=8
    )
    kpis = compute_kpis(customers, invoices, as_of=date(2030, 1, 1))
    assert kpis.dso == 0.0
