"""Runtime payment-risk scorer used by the Risk agent and KPI engine.

Features come from ``ar_platform.ml.features`` — behavioral, computed as-of
date, leakage-free (see that module for the full rationale; notably the
generator's ``historical_delay_avg`` oracle field is NOT used).

Model selection is benchmark-driven: if ``models/risk_model_meta.json`` exists
(produced by ``python -m ar_platform.ml.benchmark``), training re-fits the
winning model family with its tuned hyperparameters on the current ledger.
Without benchmark artifacts it falls back to a sane logistic-regression
default, so the platform runs out of the box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from ar_platform.config import settings
from ar_platform.ml.features import FEATURE_NAMES, FeatureBuilder, build_dataset
from ar_platform.models import Customer, Invoice, Payment


def _default_estimator(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def _estimator_from_meta(meta: dict, seed: int):
    """Re-create the benchmark winner's estimator with its tuned params."""
    name = meta.get("winner", "")
    params = meta.get("best_params", {})
    try:
        if name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000, solver="saga", penalty="elasticnet",
                            l1_ratio=0.5, random_state=seed,
                        ),
                    ),
                ]
            )
            est.set_params(**params)
            return est, name
        if name == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(random_state=seed, **params), name
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=seed, n_jobs=-1, **params), name
        if name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(random_state=seed, **params), name
        if name == "xgboost":
            from xgboost import XGBClassifier

            return (
                XGBClassifier(
                    random_state=seed, eval_metric="logloss", n_jobs=-1,
                    tree_method="hist", **params,
                ),
                name,
            )
        if name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **params), name
    except ImportError:
        pass
    return _default_estimator(seed), "logistic_regression (fallback)"


@dataclass
class RiskModel:
    """A fitted payment-risk classifier bound to a ledger snapshot.

    ``predict_default_prob`` computes features as of the model's training date
    (``as_of``) from the snapshot's history — refresh via retraining when the
    simulation advances (``Simulation(retrain_each_tick=True)``).
    """

    pipeline: object
    feature_builder: FeatureBuilder
    as_of: date
    model_name: str
    feature_names: list[str]
    train_size: int
    positive_rate: float
    auc: float | None = None  # in-sample sanity signal; honest numbers live in the benchmark

    def predict_default_prob(self, inv: Invoice, cust: Customer) -> float:
        as_of = max(self.as_of, inv.issue_date)
        x = self.feature_builder.features_for(inv, cust, as_of=as_of)
        return float(self.pipeline.predict_proba([x])[0][1])


def train_risk_model(
    customers: list[Customer],
    invoices: list[Invoice],
    payments: list[Payment],
    as_of: date | None = None,
    seed: int | None = None,
) -> RiskModel:
    """Fit the risk model on all resolved invoices in the ledger snapshot.

    Uses the benchmark winner's configuration when artifacts exist; the
    benchmark (not this function) is the source of honest held-out metrics —
    here we fit on everything available, as production would.
    """
    from sklearn.metrics import roc_auc_score

    as_of = settings.base_case_today if as_of is None else as_of
    seed = settings.seed if seed is None else seed

    from ar_platform.ml.benchmark import META_PATH

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        estimator, model_name = _estimator_from_meta(meta, seed)
    else:
        estimator, model_name = _default_estimator(seed), "logistic_regression (default)"

    ds = build_dataset(customers, invoices, payments, as_of=as_of)
    estimator.fit(ds.X, ds.y)

    try:
        scores = [row[1] for row in estimator.predict_proba(ds.X)]
        auc = float(roc_auc_score(ds.y, scores)) if len(set(ds.y)) > 1 else None
    except ValueError:
        auc = None

    return RiskModel(
        pipeline=estimator,
        feature_builder=FeatureBuilder(customers, invoices, payments),
        as_of=as_of,
        model_name=model_name,
        feature_names=list(FEATURE_NAMES),
        train_size=len(ds),
        positive_rate=round(sum(ds.y) / len(ds.y), 4) if ds.y else 0.0,
        auc=auc,
    )
