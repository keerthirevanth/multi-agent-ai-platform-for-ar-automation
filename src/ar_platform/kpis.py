"""Executive KPI engine.

Computes the finance-department metrics from the current ledger state: DSO,
aging analysis, collection efficiency, expected cash flow, and status mix. These
feed the dashboard and can also be logged per tick to chart trends over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ar_platform.config import settings
from ar_platform.models import AGING_BUCKETS, Customer, Invoice
from ar_platform.tools.ml_risk import RiskModel


@dataclass
class KPIs:
    as_of: str
    open_ar: float
    dso: float                       # Days Sales Outstanding
    overdue_ar: float
    overdue_pct: float               # share of open AR that is past due
    collection_rate: float           # collected / invoiced (lifetime)
    expected_collectible: float      # risk-adjusted recoverable AR
    expected_loss: float             # risk-weighted AR at risk of non-payment
    aging: dict[str, float] = field(default_factory=dict)   # bucket -> outstanding $
    aging_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _dso(invoices: list[Invoice], open_ar: float, as_of: date, lookback: int = 90) -> float:
    """DSO = open AR / average daily credit sales over a trailing window.

    Average daily sales come from invoice value issued in the last ``lookback``
    days, which keeps the metric responsive as the ledger evolves.
    """
    window_start = as_of - timedelta(days=lookback)
    credit_sales = sum(
        i.amount for i in invoices if window_start <= i.issue_date <= as_of
    )
    if credit_sales <= 0:
        return 0.0
    avg_daily_sales = credit_sales / lookback
    return round(open_ar / avg_daily_sales, 1)


def compute_kpis(
    customers: list[Customer],
    invoices: list[Invoice],
    as_of: date | None = None,
    risk_model: RiskModel | None = None,
) -> KPIs:
    as_of = settings.base_case_today if as_of is None else as_of
    by_cust = {c.id: c for c in customers}

    open_invoices = [i for i in invoices if i.is_open()]
    open_ar = round(sum(i.outstanding for i in open_invoices), 2)

    overdue_ar = round(
        sum(i.outstanding for i in open_invoices if i.days_overdue(as_of) > 0), 2
    )

    total_invoiced = sum(i.amount for i in invoices)
    total_collected = sum(i.paid_amount for i in invoices)
    collection_rate = round(total_collected / total_invoiced, 4) if total_invoiced else 0.0

    # Aging analysis.
    aging = {b: 0.0 for b in AGING_BUCKETS}
    aging_counts = {b: 0 for b in AGING_BUCKETS}
    for i in open_invoices:
        b = i.aging_bucket(as_of)
        aging[b] = round(aging[b] + i.outstanding, 2)
        aging_counts[b] += 1

    # Status mix.
    status_counts: dict[str, int] = {}
    for i in invoices:
        status_counts[i.status.value] = status_counts.get(i.status.value, 0) + 1

    # Risk-adjusted cash-flow expectation.
    expected_collectible = 0.0
    expected_loss = 0.0
    if risk_model is not None:
        for i in open_invoices:
            cust = by_cust.get(i.customer_id)
            if cust is None:
                continue
            p_default = risk_model.predict_default_prob(i, cust)
            expected_collectible += i.outstanding * (1 - p_default)
            expected_loss += i.outstanding * p_default

    return KPIs(
        as_of=as_of.isoformat(),
        open_ar=open_ar,
        dso=_dso(invoices, open_ar, as_of),
        overdue_ar=overdue_ar,
        overdue_pct=round(overdue_ar / open_ar, 4) if open_ar else 0.0,
        collection_rate=collection_rate,
        expected_collectible=round(expected_collectible, 2),
        expected_loss=round(expected_loss, 2),
        aging=aging,
        aging_counts=aging_counts,
        status_counts=status_counts,
    )
