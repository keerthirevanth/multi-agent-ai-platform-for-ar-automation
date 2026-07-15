"""Payment-date prediction and the weekly cash forecast.

The classifier answers *whether* an invoice becomes a problem; this module
answers *when the cash lands*. A gradient-boosting regressor learns
days-from-issue-to-settlement on historically paid invoices (same leakage-free
features as the classifier, temporal MAE holdout for honesty), then every open
invoice's predicted payment date is binned into weeks and weighted by its
collection probability — producing the week-by-week expected-cash curve a
treasurer actually plans against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from ar_platform.config import settings
from ar_platform.ml.features import FeatureBuilder
from ar_platform.models import Customer, Invoice, InvoiceStatus, Payment


@dataclass
class PaymentForecaster:
    pipeline: object
    feature_builder: FeatureBuilder
    as_of: date
    train_size: int
    mae_holdout: float | None   # days; temporal holdout

    def predict_days_to_pay(self, inv: Invoice, cust: Customer) -> float:
        as_of = max(self.as_of, inv.issue_date)
        x = self.feature_builder.features_for(inv, cust, as_of=as_of)
        return float(self.pipeline.predict([x])[0])


def train_payment_forecaster(
    customers: list[Customer],
    invoices: list[Invoice],
    payments: list[Payment],
    as_of: date | None = None,
    seed: int | None = None,
) -> PaymentForecaster:
    """Fit days-to-payment on settled invoices; report temporal-holdout MAE."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    as_of = settings.base_case_today if as_of is None else as_of
    seed = settings.seed if seed is None else seed
    by_cust = {c.id: c for c in customers}
    fb = FeatureBuilder(customers, invoices, payments)

    last_pay: dict[str, date] = {}
    for p in payments:
        d = p.date if isinstance(p.date, date) else date.fromisoformat(str(p.date))
        if p.invoice_id not in last_pay or d > last_pay[p.invoice_id]:
            last_pay[p.invoice_id] = d

    rows = []
    for inv in sorted(invoices, key=lambda i: (i.issue_date, i.id)):
        if inv.status != InvoiceStatus.PAID or inv.id not in last_pay:
            continue
        cust = by_cust.get(inv.customer_id)
        if cust is None:
            continue
        target = (last_pay[inv.id] - inv.issue_date).days
        rows.append((fb.features_for(inv, cust, as_of=inv.issue_date), float(target)))

    X = [r[0] for r in rows]
    y = [r[1] for r in rows]

    # Temporal 80/20 holdout for an honest MAE, then refit on everything.
    cut = int(len(rows) * 0.8)
    mae = None
    if cut and len(rows) - cut >= 20:
        probe = HistGradientBoostingRegressor(random_state=seed)
        probe.fit(X[:cut], y[:cut])
        mae = round(float(mean_absolute_error(y[cut:], probe.predict(X[cut:]))), 2)

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(X, y)
    return PaymentForecaster(
        pipeline=model, feature_builder=fb, as_of=as_of,
        train_size=len(rows), mae_holdout=mae,
    )


def expected_cash_curve(
    forecaster: PaymentForecaster,
    customers: list[Customer],
    invoices: list[Invoice],
    as_of: date | None = None,
    weeks: int = 8,
    risk_model=None,
) -> list[dict]:
    """Weekly expected cash inflow from currently open, undisputed invoices.

    Each invoice contributes ``outstanding × P(collect)`` to the week its
    predicted payment date falls in; predictions already past are treated as
    imminent (next week). The final bucket collects everything beyond the
    horizon.
    """
    as_of = forecaster.as_of if as_of is None else as_of
    by_cust = {c.id: c for c in customers}
    buckets = [0.0] * weeks
    later = 0.0

    for inv in invoices:
        if not inv.is_open() or inv.status == InvoiceStatus.DISPUTED:
            continue
        cust = by_cust.get(inv.customer_id)
        if cust is None:
            continue
        pred_days = forecaster.predict_days_to_pay(inv, cust)
        pred_date = inv.issue_date + timedelta(days=max(0, round(pred_days)))
        if pred_date <= as_of:
            pred_date = as_of + timedelta(days=3)  # overdue prediction -> imminent

        p_collect = 1.0
        if risk_model is not None:
            p_collect = 1.0 - risk_model.predict_default_prob(inv, cust)
        value = inv.outstanding * p_collect

        idx = (pred_date - as_of).days // 7
        if idx < weeks:
            buckets[idx] += value
        else:
            later += value

    curve = [
        {
            "week_start": (as_of + timedelta(days=7 * i)).isoformat(),
            "expected_cash": round(buckets[i], 2),
        }
        for i in range(weeks)
    ]
    curve.append({"week_start": "beyond", "expected_cash": round(later, 2)})
    return curve
