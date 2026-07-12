"""Central configuration.

Everything is driven by environment variables (see ``.env.example``) with safe
defaults so the platform runs offline and reproducibly out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Repo layout anchors ---------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
SEED_DIR = DATA_DIR / "seed"
OUTBOX_DIR = DATA_DIR / "outbox"

# Default MySQL connection. Override with AR_DB_URL (kept out of git via .env).
DEFAULT_DB_URL = "mysql+pymysql://ar_user:ar_password@localhost:3306/ar_platform"


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings resolved from the environment."""

    db_url: str = os.environ.get("AR_DB_URL", DEFAULT_DB_URL)
    llm_mode: str = os.environ.get("AR_LLM_MODE", "rules").lower()
    claude_model: str = os.environ.get("AR_CLAUDE_MODEL", "claude-sonnet-5")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    seed: int = _get_int("AR_SEED", 42)

    # The base case is anchored to a fixed "today" so aging is reproducible
    # regardless of the real wall-clock date. The simulation clock advances
    # forward from here.
    base_case_today: date = date(2026, 1, 15)

    # Synthetic data sizing for the base case. ~30 invoices per customer over
    # ~15 months: deep enough history for behavioral features to converge
    # (see the ML diagnostics in the README — history depth, not row count,
    # drives prediction performance).
    n_customers: int = _get_int("AR_N_CUSTOMERS", 150)
    n_invoices: int = _get_int("AR_N_INVOICES", 4500)
    history_days: int = _get_int("AR_HISTORY_DAYS", 450)

    # An invoice's outcome label is only decided once this many days have
    # passed after its due date ("label maturity window"): problem = not fully
    # settled within the window. Younger invoices are excluded from training.
    label_maturity_days: int = _get_int("AR_LABEL_MATURITY_DAYS", 30)


settings = Settings()
