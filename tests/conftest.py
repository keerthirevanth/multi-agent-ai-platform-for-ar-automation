"""Shared test fixtures.

Tests run against a dedicated ``ar_platform_test`` MySQL database (never the
working ``ar_platform`` DB) and reset it between tests, so each test starts from
a clean, isolated ledger. Override the target with AR_TEST_DB_URL.
"""

from __future__ import annotations

import os

import pytest

from ar_platform.data.store import Store

TEST_DB_URL = os.environ.get(
    "AR_TEST_DB_URL",
    "mysql+pymysql://ar_user:ar_password@localhost:3306/ar_platform_test",
)


@pytest.fixture
def db_url() -> str:
    return TEST_DB_URL


@pytest.fixture
def fresh_store():
    """Factory returning a freshly-reset Store on the test database."""

    def _make() -> Store:
        store = Store(TEST_DB_URL)
        store.reset()
        return store

    return _make
