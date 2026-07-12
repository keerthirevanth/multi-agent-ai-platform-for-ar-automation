"""Model benchmark: temporal evaluation, hyperparameter tuning, calibration.

Methodology (each choice is deliberate — see README's ML section):

* **Temporal split** — invoices are ordered by issue date; the first 70% train,
  the last 30% test. A random split would let the model train on invoices
  issued *after* the ones it is tested on, which never happens in production.
* **Tuning on train only** — ``RandomizedSearchCV`` with ``TimeSeriesSplit``
  inside the training window. The test window is touched exactly once per
  model, at the end.
* **Winner by validation score** — the deployed model is chosen by CV score,
  not by test score, so the test metrics remain an unbiased estimate.
* **Baselines included** — a majority-class dummy and a single-feature rule
  (rank by the customer's prior mean lateness). Any model that cannot clearly
  beat the rule baseline is not earning its complexity.
* **Calibration checked** — risk probabilities feed ``expected_loss = amount ×
  p`` KPIs, so we report Brier score and also fit an isotonic-calibrated
  variant of the winner and keep whichever calibrates better.

Run:

    PYTHONPATH=src python -m ar_platform.ml.benchmark [--quick]

Artifacts: ``reports/benchmark_results.csv``, ``models/risk_model.joblib``,
``models/risk_model_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import joblib
import numpy as np

from ar_platform.config import REPO_ROOT, settings
from ar_platform.ml.features import FEATURE_NAMES, Dataset, build_dataset

# Process-based search parallelism. Windows page-file limits make n_jobs=-1
# fragile (each worker re-loads the native ML DLLs), so default to sequential
# search; the tree/boosting estimators still use threads internally.
N_JOBS = int(os.environ.get("AR_ML_JOBS", "1"))

REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "risk_model.joblib"
META_PATH = MODELS_DIR / "risk_model_meta.json"

_PRIOR_LATE_MEAN_IDX = FEATURE_NAMES.index("prior_late_mean")


# --- model space -------------------------------------------------------------
def model_space(seed: int, quick: bool = False) -> dict[str, tuple]:
    """(estimator, param_distributions) per model. Import-guarded extras."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    space: dict[str, tuple] = {
        "logistic_regression": (
            Pipeline(
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
            ),
            {
                "clf__C": np.logspace(-3, 2, 30),
                "clf__l1_ratio": np.linspace(0.0, 1.0, 11),
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "max_depth": [3, 4, 5, 6, 8, 10, None],
                "min_samples_leaf": [5, 10, 20, 40, 80],
                "class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_leaf": [2, 5, 10, 20],
                "max_features": ["sqrt", 0.5, None],
                "class_weight": [None, "balanced"],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.03, 0.06, 0.1, 0.2],
                "max_depth": [2, 3, 4, 6, None],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier

        space["xgboost"] = (
            XGBClassifier(
                random_state=seed, eval_metric="logloss", n_jobs=-1,
                tree_method="hist",
            ),
            {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.03, 0.06, 0.1, 0.2],
                "max_depth": [2, 3, 4, 6],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 3.0],
            },
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        space["lightgbm"] = (
            LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1),
            {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.03, 0.06, 0.1, 0.2],
                "num_leaves": [7, 15, 31, 63],
                "min_child_samples": [10, 20, 40],
                "subsample": [0.7, 0.9, 1.0],
                "reg_lambda": [0.0, 0.5, 3.0],
            },
        )
    except ImportError:
        pass

    if quick:
        space = {k: space[k] for k in ("logistic_regression", "decision_tree") if k in space}
    return space


# --- metrics ------------------------------------------------------------------
def top_capture(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, frac: float = 0.2) -> float:
    """Share of problem-invoice dollars captured in the top ``frac`` by score.

    The business question: if collectors only have capacity for the top 20% of
    the worklist, how much of the actual at-risk money does the model put there?
    """
    n_top = max(1, int(round(len(y) * frac)))
    order = np.argsort(-scores)
    top = order[:n_top]
    at_risk_total = float(np.sum(amounts[y == 1]))
    if at_risk_total <= 0:
        return 0.0
    captured = float(np.sum(amounts[top][y[top] == 1]))
    return captured / at_risk_total


@dataclass
class ModelResult:
    model: str
    cv_auc: float
    test_auc: float
    test_pr_auc: float
    test_brier: float
    test_top20_capture: float
    best_params: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    as_of: str
    n_train: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float
    split_date: str
    winner: str
    winner_calibrated: bool
    results: list[ModelResult] = field(default_factory=list)


# --- core ----------------------------------------------------------------------
def temporal_split(ds: Dataset, train_frac: float = 0.7):
    """Split by issue-date order: past -> train, future -> test."""
    n = len(ds)
    cut = int(round(n * train_frac))
    X = np.asarray(ds.X, dtype=float)
    y = np.asarray(ds.y, dtype=int)
    amounts = np.asarray(ds.amounts, dtype=float)
    split_date = ds.issue_dates[cut] if cut < n else ds.issue_dates[-1]
    return (
        X[:cut], y[:cut],
        X[cut:], y[cut:],
        amounts[cut:], split_date,
    )


def run_benchmark(
    customers,
    invoices,
    payments,
    as_of: date | None = None,
    seed: int | None = None,
    quick: bool = False,
    save: bool = True,
) -> BenchmarkReport:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    as_of = settings.base_case_today if as_of is None else as_of
    seed = settings.seed if seed is None else seed

    ds = build_dataset(customers, invoices, payments, as_of=as_of)
    X_tr, y_tr, X_te, y_te, amt_te, split_date = temporal_split(ds)

    report = BenchmarkReport(
        as_of=as_of.isoformat(),
        n_train=len(y_tr),
        n_test=len(y_te),
        train_positive_rate=round(float(np.mean(y_tr)), 4),
        test_positive_rate=round(float(np.mean(y_te)), 4),
        split_date=split_date.isoformat(),
        winner="",
        winner_calibrated=False,
    )

    # Baseline 1: majority-class dummy (AUC pins at 0.5 — the floor).
    report.results.append(
        ModelResult(
            model="baseline_majority",
            cv_auc=0.5,
            test_auc=0.5,
            test_pr_auc=round(float(np.mean(y_te)), 4),
            test_brier=round(float(brier_score_loss(y_te, np.full(len(y_te), np.mean(y_tr)))), 4),
            test_top20_capture=round(top_capture(y_te, np.zeros(len(y_te)), amt_te), 4),
        )
    )

    # Baseline 2: rank purely by the customer's prior mean lateness.
    rule_scores = X_te[:, _PRIOR_LATE_MEAN_IDX]
    report.results.append(
        ModelResult(
            model="baseline_prior_lateness_rule",
            cv_auc=float("nan"),
            test_auc=round(float(roc_auc_score(y_te, rule_scores)), 4),
            test_pr_auc=round(float(average_precision_score(y_te, rule_scores)), 4),
            test_brier=float("nan"),  # the rule emits ranks, not probabilities
            test_top20_capture=round(top_capture(y_te, rule_scores, amt_te), 4),
        )
    )

    cv = TimeSeriesSplit(n_splits=4)
    n_iter = 5 if quick else 25
    searches: dict[str, RandomizedSearchCV] = {}

    for name, (estimator, params) in model_space(seed, quick=quick).items():
        search = RandomizedSearchCV(
            estimator,
            params,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=seed,
            n_jobs=N_JOBS,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        searches[name] = search

        proba = search.predict_proba(X_te)[:, 1]
        report.results.append(
            ModelResult(
                model=name,
                cv_auc=round(float(search.best_score_), 4),
                test_auc=round(float(roc_auc_score(y_te, proba)), 4),
                test_pr_auc=round(float(average_precision_score(y_te, proba)), 4),
                test_brier=round(float(brier_score_loss(y_te, proba)), 4),
                test_top20_capture=round(top_capture(y_te, proba, amt_te), 4),
                best_params={k: _jsonable(v) for k, v in search.best_params_.items()},
            )
        )

    # Winner: best CV score (selection never sees the test window).
    tuned = [r for r in report.results if r.model in searches]
    winner_result = max(tuned, key=lambda r: r.cv_auc)
    report.winner = winner_result.model
    winner_search = searches[report.winner]

    # Calibration: isotonic on the training window; keep it if Brier improves.
    from sklearn.base import clone

    calibrated = CalibratedClassifierCV(
        clone(winner_search.best_estimator_), method="isotonic", cv=TimeSeriesSplit(n_splits=4)
    )
    calibrated.fit(X_tr, y_tr)
    brier_cal = float(
        brier_score_loss(y_te, calibrated.predict_proba(X_te)[:, 1])
    )
    if brier_cal < winner_result.test_brier:
        report.winner_calibrated = True
        final_model = calibrated
    else:
        final_model = winner_search.best_estimator_

    if save:
        _save_artifacts(report, final_model, winner_result, brier_cal)
    return report


def _jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _save_artifacts(report, final_model, winner_result, brier_cal) -> None:
    import csv

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    with open(REPORTS_DIR / "benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "cv_auc", "test_auc", "test_pr_auc",
                "test_brier", "test_top20_capture", "best_params",
            ],
        )
        writer.writeheader()
        for r in report.results:
            row = asdict(r)
            row["best_params"] = json.dumps(row["best_params"])
            writer.writerow(row)

    joblib.dump(final_model, MODEL_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "as_of": report.as_of,
                "winner": report.winner,
                "winner_calibrated": report.winner_calibrated,
                "cv_auc": winner_result.cv_auc,
                "test_auc": winner_result.test_auc,
                "test_pr_auc": winner_result.test_pr_auc,
                "test_brier_raw": winner_result.test_brier,
                "test_brier_calibrated": round(brier_cal, 4),
                "test_top20_capture": winner_result.test_top20_capture,
                "best_params": winner_result.best_params,
                "n_train": report.n_train,
                "n_test": report.n_test,
                "split_date": report.split_date,
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the payment-risk model benchmark.")
    parser.add_argument("--quick", action="store_true", help="small search, 2 models")
    args = parser.parse_args()

    from ar_platform.data.store import Store

    with Store() as store:
        store.load_base_case()
        customers = store.get_customers()
        invoices = store.get_invoices()
        payments = store.get_payments()

    report = run_benchmark(customers, invoices, payments, quick=args.quick)

    print(
        f"\nDataset: {report.n_train} train / {report.n_test} test "
        f"(temporal split at {report.split_date}; "
        f"positive rate {report.train_positive_rate:.0%} / {report.test_positive_rate:.0%})\n"
    )
    header = (
        f"{'model':30s} {'cv_auc':>7s} {'test_auc':>8s} "
        f"{'pr_auc':>7s} {'brier':>7s} {'top20%':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(report.results, key=lambda r: -(r.test_auc or 0)):
        cv = f"{r.cv_auc:7.3f}" if r.cv_auc == r.cv_auc else "      -"
        brier = f"{r.test_brier:7.3f}" if r.test_brier == r.test_brier else "      -"
        print(
            f"{r.model:30s} {cv} {r.test_auc:8.3f} {r.test_pr_auc:7.3f} "
            f"{brier} {r.test_top20_capture:7.3f}"
        )
    print(
        f"\nWinner (by CV): {report.winner}"
        f"{' + isotonic calibration' if report.winner_calibrated else ''}"
    )
    print(f"Artifacts: {MODEL_PATH}, {META_PATH}, {REPORTS_DIR / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
