"""Leakage-free behavioral feature engineering for payment-risk models.

Every feature for an invoice is computed **as of that invoice's issue date**,
using only invoices issued and payments received strictly before that moment.
This mirrors production exactly: when an invoice is created, the scorer knows
the customer's transaction history so far — nothing from the future.

Deliberately excluded:

* ``customer.historical_delay_avg`` — an oracle field written by the data
  generator. A real ERP has no truthful "how late is this customer on average"
  column; it must be *derived* from history. Using it would smuggle the
  generator's ground truth into the model.
* Anything downstream of the outcome (days overdue, current status, dunning
  contacts) — those leak the label.

Cold start is explicit: a customer's first invoice has no history, so history
features fall back to neutral values and ``has_history`` flags the fact,
letting the model learn a separate prior for unknown customers.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean, pstdev

from ar_platform.config import settings
from ar_platform.models import Customer, Invoice, Payment, Segment

FEATURE_NAMES = [
    # static / contractual
    "amount",
    "payment_terms_days",
    "credit_limit",
    # behavioral, as-of issue date
    "n_prior_invoices",
    "prior_late_mean",
    "prior_late_shrunk",
    "prior_late_std",
    "prior_late_max",
    "prior_ontime_ratio",
    "prior_settled_ratio",
    "open_exposure",
    "credit_utilization",
    "n_open_overdue",
    "days_since_last_payment",
    "amount_vs_avg",
    "has_history",
    # segment one-hot
    "seg_enterprise",
    "seg_midmarket",
    "seg_smb",
]

_SEGMENTS = [Segment.ENTERPRISE, Segment.MIDMARKET, Segment.SMB]
_NO_PAYMENT_DAYS = 999.0  # sentinel for "never paid us yet"

# Empirical-Bayes shrinkage for the lateness estimate: a customer's observed
# mean lateness is pulled toward a neutral prior, weighted by how much history
# they have. With 2 settled invoices the estimate leans on the prior; with 20
# it is essentially their own mean. Constants (not data-derived) keep it
# leak-free.
_SHRINK_PRIOR_DAYS = 7.0
_SHRINK_STRENGTH = 3.0


@dataclass
class _CustomerLedger:
    """A customer's invoices sorted by issue date, with per-invoice payments."""

    invoices: list[Invoice]
    issue_dates: list[date]
    payments_by_invoice: dict[str, list[Payment]]


class FeatureBuilder:
    """Computes as-of-date features from a ledger snapshot.

    Build once per training/scoring pass; ``features_for`` is then O(prior
    invoices of that customer).
    """

    def __init__(
        self,
        customers: list[Customer],
        invoices: list[Invoice],
        payments: list[Payment],
    ):
        pay_by_inv: dict[str, list[Payment]] = {}
        for p in payments:
            pay_by_inv.setdefault(p.invoice_id, []).append(p)
        for plist in pay_by_inv.values():
            plist.sort(key=lambda p: p.date)

        by_cust: dict[str, list[Invoice]] = {}
        for inv in invoices:
            by_cust.setdefault(inv.customer_id, []).append(inv)

        self._ledgers: dict[str, _CustomerLedger] = {}
        for cust_id, invs in by_cust.items():
            invs.sort(key=lambda i: (i.issue_date, i.id))
            self._ledgers[cust_id] = _CustomerLedger(
                invoices=invs,
                issue_dates=[i.issue_date for i in invs],
                payments_by_invoice={
                    i.id: pay_by_inv.get(i.id, []) for i in invs
                },
            )

    # -- internals -----------------------------------------------------------
    def _paid_before(self, ledger: _CustomerLedger, inv: Invoice, as_of: date) -> float:
        """Amount of ``inv`` settled strictly before ``as_of``."""
        return sum(
            p.amount for p in ledger.payments_by_invoice.get(inv.id, ())
            if p.date < as_of
        )

    def features_for(self, inv: Invoice, cust: Customer, as_of: date) -> list[float]:
        """Feature vector for ``inv``, seeing only history strictly before ``as_of``.

        For training/scoring at issue time, pass ``as_of = inv.issue_date``.
        """
        ledger = self._ledgers.get(cust.id)
        prior = []
        if ledger is not None:
            cut = bisect_left(ledger.issue_dates, as_of)
            prior = [i for i in ledger.invoices[:cut] if i.id != inv.id]

        # Settlement lateness of prior invoices that are fully paid by as_of.
        latenesses: list[float] = []
        settled = 0
        open_exposure = 0.0
        n_open_overdue = 0
        last_payment: date | None = None
        amounts = []

        for pi in prior:
            amounts.append(pi.amount)
            paid = self._paid_before(ledger, pi, as_of)
            plist = [
                p for p in ledger.payments_by_invoice.get(pi.id, ())
                if p.date < as_of
            ]
            if plist:
                last = plist[-1].date
                last_payment = last if last_payment is None else max(last_payment, last)
            if paid >= pi.amount - 0.01:
                settled += 1
                latenesses.append((plist[-1].date - pi.due_date).days)
            else:
                outstanding = pi.amount - paid
                open_exposure += outstanding
                if pi.due_date < as_of:
                    n_open_overdue += 1

        n_prior = len(prior)
        has_history = 1.0 if n_prior > 0 else 0.0
        credit_util = (
            (open_exposure + inv.amount) / cust.credit_limit
            if cust.credit_limit > 0
            else 0.0
        )
        seg_onehot = [1.0 if cust.segment == s else 0.0 for s in _SEGMENTS]

        shrunk = (sum(latenesses) + _SHRINK_STRENGTH * _SHRINK_PRIOR_DAYS) / (
            len(latenesses) + _SHRINK_STRENGTH
        )

        return [
            inv.amount,
            float(cust.payment_terms_days),
            cust.credit_limit,
            float(n_prior),
            mean(latenesses) if latenesses else 0.0,
            shrunk,
            pstdev(latenesses) if len(latenesses) > 1 else 0.0,
            max(latenesses) if latenesses else 0.0,
            (sum(1 for d in latenesses if d <= 0) / len(latenesses))
            if latenesses
            else 0.0,
            (settled / n_prior) if n_prior else 0.0,
            open_exposure,
            credit_util,
            float(n_open_overdue),
            float((as_of - last_payment).days) if last_payment else _NO_PAYMENT_DAYS,
            (inv.amount / mean(amounts)) if amounts else 1.0,
            has_history,
            *seg_onehot,
        ]


def label_for(
    inv: Invoice,
    inv_payments: list[Payment],
    as_of: date,
    maturity_days: int | None = None,
) -> int | None:
    """Maturity-window outcome label.

    ``1`` = collection problem (not fully settled within ``maturity_days`` after
    the due date), ``0`` = clean, ``None`` = the window has not elapsed yet as
    of ``as_of``, so the outcome is not knowable and the invoice must be
    excluded from training.

    This fixes the bias of a naive "overdue right now" label, where recently
    issued invoices look artificially risky simply because they have not had
    time to be paid — the cause of train/test positive-rate drift under
    temporal splits.
    """
    maturity_days = (
        settings.label_maturity_days if maturity_days is None else maturity_days
    )
    deadline = inv.due_date + timedelta(days=maturity_days)
    if as_of <= deadline:
        return None
    paid_by_deadline = sum(p.amount for p in inv_payments if p.date <= deadline)
    return 0 if paid_by_deadline >= inv.amount - 0.01 else 1


@dataclass
class Dataset:
    """A supervised dataset with per-row metadata for temporal splitting."""

    X: list[list[float]]
    y: list[int]
    issue_dates: list[date]
    amounts: list[float]
    invoice_ids: list[str]

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(
    customers: list[Customer],
    invoices: list[Invoice],
    payments: list[Payment],
    as_of: date | None = None,
) -> Dataset:
    """Features at issue time + resolved-outcome labels for every labelable invoice."""
    as_of = settings.base_case_today if as_of is None else as_of
    by_cust = {c.id: c for c in customers}
    fb = FeatureBuilder(customers, invoices, payments)

    pay_by_inv: dict[str, list[Payment]] = {}
    for p in payments:
        pay_by_inv.setdefault(p.invoice_id, []).append(p)

    ds = Dataset(X=[], y=[], issue_dates=[], amounts=[], invoice_ids=[])
    for inv in sorted(invoices, key=lambda i: (i.issue_date, i.id)):
        cust = by_cust.get(inv.customer_id)
        if cust is None:
            continue
        label = label_for(inv, pay_by_inv.get(inv.id, []), as_of)
        if label is None:
            continue
        ds.X.append(fb.features_for(inv, cust, as_of=inv.issue_date))
        ds.y.append(label)
        ds.issue_dates.append(inv.issue_date)
        ds.amounts.append(inv.amount)
        ds.invoice_ids.append(inv.id)
    return ds
