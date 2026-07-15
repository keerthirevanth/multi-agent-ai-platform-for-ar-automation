"""Persistence layer: MySQL-backed AR ledger (SQLAlchemy).

The committed base case lives as CSVs in ``data/seed/``. On first use we load
those into a MySQL database which becomes the mutable working state the
simulation and agents read and write. This separation keeps the reproducible
base case pristine while letting each run evolve freely.

SQLAlchemy's engine gives us a thread-safe connection pool, so the same Store
can be shared across Streamlit's rerun threads without the single-connection
issues a raw driver would hit.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from ar_platform.config import SEED_DIR, settings
from ar_platform.models import (
    AuditEntry,
    CreditHold,
    Customer,
    CustomerReply,
    Dispute,
    DisputeStatus,
    Escalation,
    EscalationStatus,
    Invoice,
    Payment,
    PromiseStatus,
    PromiseToPay,
    Remittance,
    RemittanceStatus,
)

# MySQL DDL. VARCHAR lengths are required for keys/indexes; money is DECIMAL,
# rates are DOUBLE, audit metadata is native JSON.
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS customers (
        id VARCHAR(20) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        segment VARCHAR(20) NOT NULL,
        email VARCHAR(255) NOT NULL,
        credit_limit DECIMAL(16,2) NOT NULL,
        payment_terms_days INT NOT NULL,
        historical_delay_avg DOUBLE NOT NULL,
        risk_band VARCHAR(10) NOT NULL
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS invoices (
        id VARCHAR(20) PRIMARY KEY,
        customer_id VARCHAR(20) NOT NULL,
        amount DECIMAL(16,2) NOT NULL,
        issue_date DATE NOT NULL,
        due_date DATE NOT NULL,
        status VARCHAR(12) NOT NULL,
        paid_amount DECIMAL(16,2) NOT NULL DEFAULT 0,
        INDEX idx_invoices_customer (customer_id),
        INDEX idx_invoices_status (status),
        CONSTRAINT fk_inv_customer FOREIGN KEY (customer_id)
            REFERENCES customers(id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS payments (
        id VARCHAR(20) PRIMARY KEY,
        invoice_id VARCHAR(20) NOT NULL,
        amount DECIMAL(16,2) NOT NULL,
        date DATE NOT NULL,
        method VARCHAR(20) NOT NULL,
        INDEX idx_payments_invoice (invoice_id),
        CONSTRAINT fk_pay_invoice FOREIGN KEY (invoice_id)
            REFERENCES invoices(id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS escalations (
        id VARCHAR(20) PRIMARY KEY,
        invoice_id VARCHAR(20) NOT NULL,
        customer_id VARCHAR(20) NOT NULL,
        sim_date DATE NOT NULL,
        outstanding DECIMAL(16,2) NOT NULL,
        risk_score DOUBLE NOT NULL,
        priority DECIMAL(16,2) NOT NULL,
        severity VARCHAR(12) NOT NULL,
        reason VARCHAR(255) NOT NULL,
        status VARCHAR(12) NOT NULL DEFAULT 'pending',
        resolved_date DATE NULL,
        resolution_note VARCHAR(255) NULL,
        INDEX idx_esc_status (status),
        INDEX idx_esc_invoice (invoice_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS customer_replies (
        id VARCHAR(24) PRIMARY KEY,
        invoice_id VARCHAR(20) NOT NULL,
        customer_id VARCHAR(20) NOT NULL,
        sim_date DATE NOT NULL,
        text VARCHAR(1024) NOT NULL,
        intent VARCHAR(24) NULL,
        status VARCHAR(12) NOT NULL DEFAULT 'new',
        handled_action VARCHAR(64) NULL,
        INDEX idx_reply_status (status),
        INDEX idx_reply_invoice (invoice_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS promises (
        id VARCHAR(24) PRIMARY KEY,
        invoice_id VARCHAR(20) NOT NULL,
        customer_id VARCHAR(20) NOT NULL,
        created_date DATE NOT NULL,
        amount DECIMAL(16,2) NOT NULL,
        due_date DATE NOT NULL,
        status VARCHAR(12) NOT NULL DEFAULT 'pending',
        INDEX idx_promise_status (status),
        INDEX idx_promise_invoice (invoice_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS remittances (
        id VARCHAR(24) PRIMARY KEY,
        date DATE NOT NULL,
        payer_name VARCHAR(255) NOT NULL,
        amount DECIMAL(16,2) NOT NULL,
        reference_text VARCHAR(255) NOT NULL,
        customer_id VARCHAR(20) NOT NULL,
        intended_invoice_id VARCHAR(20) NOT NULL,
        status VARCHAR(12) NOT NULL DEFAULT 'unmatched',
        matched_invoice_id VARCHAR(20) NULL,
        match_method VARCHAR(20) NULL,
        INDEX idx_rem_status (status)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS disputes (
        id VARCHAR(24) PRIMARY KEY,
        invoice_id VARCHAR(20) NOT NULL,
        customer_id VARCHAR(20) NOT NULL,
        opened_date DATE NOT NULL,
        reason_category VARCHAR(20) NOT NULL,
        reason_text VARCHAR(1024) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'open',
        resolved_date DATE NULL,
        resolution_note VARCHAR(255) NULL,
        INDEX idx_disp_status (status),
        INDEX idx_disp_invoice (invoice_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS credit_holds (
        customer_id VARCHAR(20) PRIMARY KEY,
        held_date DATE NOT NULL,
        utilization_at_hold DOUBLE NOT NULL,
        released_date DATE NULL
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id VARCHAR(40) PRIMARY KEY,
        timestamp VARCHAR(32) NOT NULL,
        sim_date VARCHAR(12) NOT NULL,
        agent VARCHAR(32) NOT NULL,
        action VARCHAR(48) NOT NULL,
        entity_type VARCHAR(24) NOT NULL,
        entity_id VARCHAR(40) NOT NULL,
        detail VARCHAR(512),
        metadata JSON,
        INDEX idx_audit_agent (agent)
    ) ENGINE=InnoDB
    """,
]


@lru_cache(maxsize=8)
def _get_engine(db_url: str) -> Engine:
    """One pooled engine per URL (cached), shared across Store instances."""
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=3600, future=True)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Store:
    """Repository over a MySQL database.

    Use as a context manager::

        with Store() as store:
            store.load_base_case()
            invoices = store.get_open_invoices()
    """

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or settings.db_url
        self.engine = _get_engine(self.db_url)
        self._ensure_schema()

    # -- low-level helpers -------------------------------------------------
    def _ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            for ddl in _DDL:
                conn.execute(text(ddl))

    def _exec(self, sql: str, params: dict | list[dict] | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def _query(self, sql: str, params: dict | None = None) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params or {}).mappings().all()
        return [dict(r) for r in rows]

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # The engine/pool is process-cached and reused; nothing to close here.
        pass

    def is_empty(self) -> bool:
        return self._query("SELECT COUNT(*) AS n FROM customers")[0]["n"] == 0

    def reset(self) -> None:
        """Drop all working state (children first for FK safety)."""
        with self.engine.begin() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            for table in (
                "payments", "invoices", "customers", "escalations",
                "customer_replies", "promises", "remittances", "disputes",
                "credit_holds", "audit_log",
            ):
                conn.execute(text(f"DELETE FROM {table}"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    # -- base case ---------------------------------------------------------
    def load_base_case(self, force: bool = False) -> None:
        """Load the committed seed CSVs into the database.

        No-op if the DB already has data unless ``force=True``. Uses bulk inserts
        inside a single transaction for speed.
        """
        if not self.is_empty() and not force:
            return
        self.reset()

        customers = [Customer.from_row(r).to_row() for r in _read_csv(SEED_DIR / "customers.csv")]
        invoices = [Invoice.from_row(r).to_row() for r in _read_csv(SEED_DIR / "invoices.csv")]
        payments = [Payment.from_row(r).to_row() for r in _read_csv(SEED_DIR / "payments.csv")]

        with self.engine.begin() as conn:
            if customers:
                conn.execute(text(_INSERT_CUSTOMER), customers)
            if invoices:
                conn.execute(text(_INSERT_INVOICE), invoices)
            if payments:
                conn.execute(text(_INSERT_PAYMENT), payments)

    # -- customers ---------------------------------------------------------
    def upsert_customer(self, c: Customer) -> None:
        self._exec(_INSERT_CUSTOMER, c.to_row())

    def get_customer(self, customer_id: str) -> Customer | None:
        rows = self._query("SELECT * FROM customers WHERE id = :id", {"id": customer_id})
        return Customer.from_row(rows[0]) if rows else None

    def get_customers(self) -> list[Customer]:
        return [Customer.from_row(r) for r in self._query("SELECT * FROM customers ORDER BY id")]

    # -- invoices ----------------------------------------------------------
    def upsert_invoice(self, inv: Invoice) -> None:
        self._exec(_INSERT_INVOICE, inv.to_row())

    def get_invoices(self) -> list[Invoice]:
        return [Invoice.from_row(r) for r in self._query("SELECT * FROM invoices ORDER BY id")]

    def get_open_invoices(self) -> list[Invoice]:
        return [
            Invoice.from_row(r)
            for r in self._query("SELECT * FROM invoices WHERE status != 'paid' ORDER BY id")
        ]

    def update_invoice_due_date(self, invoice_id: str, new_due: str) -> None:
        """Extend/reset an invoice's due date (used when a plan is negotiated)."""
        self._exec(
            "UPDATE invoices SET due_date = :d WHERE id = :id",
            {"d": new_due, "id": invoice_id},
        )

    # -- payments ----------------------------------------------------------
    def add_payment(self, p: Payment) -> None:
        self._exec(_INSERT_PAYMENT_IGNORE, p.to_row())

    def get_payments(self) -> list[Payment]:
        return [Payment.from_row(r) for r in self._query("SELECT * FROM payments ORDER BY id")]

    # -- sequences / lookups ----------------------------------------------
    def next_seq(self, table: str, prefix: str) -> int:
        """Largest numeric suffix used in ``table.id`` for ids like PREFIX-00042."""
        rows = self._query(f"SELECT id FROM {table}")  # noqa: S608 (table is internal)
        best = 0
        for r in rows:
            tail = r["id"].rsplit("-", 1)[-1]
            if tail.isdigit():
                best = max(best, int(tail))
        return best + 1

    def contacted_invoice_ids(self) -> set[str]:
        rows = self._query(
            "SELECT DISTINCT entity_id FROM audit_log WHERE action = 'email_sent'"
        )
        return {r["entity_id"] for r in rows}

    # -- escalations -------------------------------------------------------
    def add_escalation(self, e: Escalation) -> None:
        self._exec(_INSERT_ESCALATION, e.to_row())

    def get_escalations(self, status: EscalationStatus | None = None) -> list[Escalation]:
        if status is None:
            rows = self._query("SELECT * FROM escalations ORDER BY id")
        else:
            rows = self._query(
                "SELECT * FROM escalations WHERE status = :s ORDER BY id", {"s": status.value}
            )
        return [Escalation.from_row(r) for r in rows]

    def update_escalation(
        self, esc_id: str, status: EscalationStatus, resolved_date: str, note: str
    ) -> None:
        self._exec(
            """UPDATE escalations
               SET status = :status, resolved_date = :rd, resolution_note = :note
               WHERE id = :id""",
            {"status": status.value, "rd": resolved_date, "note": note, "id": esc_id},
        )

    def has_open_escalation(self, invoice_id: str) -> bool:
        rows = self._query(
            "SELECT 1 AS x FROM escalations "
            "WHERE invoice_id = :iid AND status = 'pending' LIMIT 1",
            {"iid": invoice_id},
        )
        return bool(rows)

    # -- customer replies --------------------------------------------------
    def add_reply(self, r: CustomerReply) -> None:
        self._exec(_INSERT_REPLY, r.to_row())

    def get_new_replies(self) -> list[CustomerReply]:
        rows = self._query(
            "SELECT * FROM customer_replies WHERE status = 'new' ORDER BY id"
        )
        return [CustomerReply.from_row(row) for row in rows]

    def get_replies(self, invoice_id: str | None = None) -> list[CustomerReply]:
        if invoice_id is None:
            rows = self._query("SELECT * FROM customer_replies ORDER BY sim_date DESC")
        else:
            rows = self._query(
                "SELECT * FROM customer_replies WHERE invoice_id = :i", {"i": invoice_id}
            )
        return [CustomerReply.from_row(r) for r in rows]

    def has_reply(self, invoice_id: str) -> bool:
        rows = self._query(
            "SELECT 1 AS x FROM customer_replies WHERE invoice_id = :i LIMIT 1",
            {"i": invoice_id},
        )
        return bool(rows)

    def mark_reply_handled(self, reply_id: str, intent: str, action: str) -> None:
        self._exec(
            """UPDATE customer_replies
               SET status = 'handled', intent = :intent, handled_action = :action
               WHERE id = :id""",
            {"intent": intent, "action": action, "id": reply_id},
        )

    # -- promises ----------------------------------------------------------
    def add_promise(self, p: PromiseToPay) -> None:
        self._exec(_INSERT_PROMISE, p.to_row())

    def get_promises(self, status: PromiseStatus | None = None) -> list[PromiseToPay]:
        if status is None:
            rows = self._query("SELECT * FROM promises ORDER BY id")
        else:
            rows = self._query(
                "SELECT * FROM promises WHERE status = :s ORDER BY id", {"s": status.value}
            )
        return [PromiseToPay.from_row(r) for r in rows]

    def has_active_promise(self, invoice_id: str) -> bool:
        rows = self._query(
            "SELECT 1 AS x FROM promises "
            "WHERE invoice_id = :i AND status = 'pending' LIMIT 1",
            {"i": invoice_id},
        )
        return bool(rows)

    def broken_promise_count(self, invoice_id: str) -> int:
        return self._query(
            "SELECT COUNT(*) AS n FROM promises "
            "WHERE invoice_id = :i AND status = 'broken'",
            {"i": invoice_id},
        )[0]["n"]

    def update_promise_status(self, promise_id: str, status: PromiseStatus) -> None:
        self._exec(
            "UPDATE promises SET status = :s WHERE id = :id",
            {"s": status.value, "id": promise_id},
        )

    # -- remittances (cash application) --------------------------------------
    def add_remittance(self, r: Remittance) -> None:
        self._exec(_INSERT_REMITTANCE, r.to_row())

    def get_remittances(self, status: RemittanceStatus | None = None) -> list[Remittance]:
        if status is None:
            rows = self._query("SELECT * FROM remittances ORDER BY id")
        else:
            rows = self._query(
                "SELECT * FROM remittances WHERE status = :s ORDER BY id", {"s": status.value}
            )
        return [Remittance.from_row(r) for r in rows]

    def update_remittance(
        self, rem_id: str, status: RemittanceStatus,
        matched_invoice_id: str | None = None, match_method: str = "",
    ) -> None:
        self._exec(
            """UPDATE remittances SET status = :s, matched_invoice_id = :inv,
               match_method = :m WHERE id = :id""",
            {"s": status.value, "inv": matched_invoice_id, "m": match_method,
             "id": rem_id},
        )

    def invoices_with_pending_remittance(self) -> set[str]:
        """Invoices the customer has already paid for, cash not yet applied.

        Used by the world model: a customer who has sent a remittance will not
        pay the same invoice again, even though our books still show it open.
        """
        rows = self._query(
            "SELECT DISTINCT intended_invoice_id AS iid FROM remittances "
            "WHERE status != 'matched'"
        )
        return {r["iid"] for r in rows}

    def get_remittance(self, rem_id: str) -> Remittance | None:
        rows = self._query(
            "SELECT * FROM remittances WHERE id = :id", {"id": rem_id}
        )
        return Remittance.from_row(rows[0]) if rows else None

    def cash_applied_since(self, start_date: str) -> float:
        """Total payments applied to the ledger after a date (excl. credit memos)."""
        rows = self._query(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM payments "
            "WHERE date > :d AND method != 'credit_memo'",
            {"d": start_date},
        )
        return float(rows[0]["total"])

    def unapplied_cash(self) -> float:
        rows = self._query(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM remittances "
            "WHERE status != 'matched'"
        )
        return float(rows[0]["total"])

    # -- disputes ------------------------------------------------------------
    def add_dispute(self, d: Dispute) -> None:
        self._exec(_INSERT_DISPUTE, d.to_row())

    def get_disputes(self, status: DisputeStatus | None = None) -> list[Dispute]:
        if status is None:
            rows = self._query("SELECT * FROM disputes ORDER BY id")
        else:
            rows = self._query(
                "SELECT * FROM disputes WHERE status = :s ORDER BY id", {"s": status.value}
            )
        return [Dispute.from_row(r) for r in rows]

    def get_open_dispute(self, invoice_id: str) -> Dispute | None:
        rows = self._query(
            "SELECT * FROM disputes WHERE invoice_id = :i AND status = 'open' LIMIT 1",
            {"i": invoice_id},
        )
        return Dispute.from_row(rows[0]) if rows else None

    def resolve_dispute(
        self, dispute_id: str, status: DisputeStatus, resolved_date: str, note: str
    ) -> None:
        self._exec(
            """UPDATE disputes SET status = :s, resolved_date = :d,
               resolution_note = :n WHERE id = :id""",
            {"s": status.value, "d": resolved_date, "n": note, "id": dispute_id},
        )

    # -- credit holds ----------------------------------------------------------
    def add_credit_hold(self, h: CreditHold) -> None:
        self._exec(_INSERT_HOLD, h.to_row())

    def active_credit_holds(self) -> list[CreditHold]:
        rows = self._query(
            "SELECT * FROM credit_holds WHERE released_date IS NULL ORDER BY customer_id"
        )
        return [CreditHold.from_row(r) for r in rows]

    def release_credit_hold(self, customer_id: str, released_date: str) -> None:
        self._exec(
            "UPDATE credit_holds SET released_date = :d WHERE customer_id = :c",
            {"d": released_date, "c": customer_id},
        )

    # -- audit -------------------------------------------------------------
    def add_audit(self, entry: AuditEntry) -> None:
        self._exec(
            _INSERT_AUDIT,
            {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "sim_date": entry.sim_date,
                "agent": entry.agent,
                "action": entry.action,
                "entity_type": entry.entity_type,
                "entity_id": entry.entity_id,
                "detail": entry.detail,
                "metadata": json.dumps(entry.metadata),
            },
        )

    def recent_audit(self, limit: int = 50) -> list[dict]:
        return self._query(
            "SELECT sim_date, agent, action, entity_id, detail "
            "FROM audit_log ORDER BY id DESC LIMIT :n",
            {"n": limit},
        )

    def commit(self) -> None:
        # Writes auto-commit per statement (each opens its own transaction), so
        # this is a no-op kept for API compatibility with the call sites.
        pass


# --- upsert / insert statements (MySQL dialect) -----------------------------
_INSERT_CUSTOMER = """
INSERT INTO customers
    (id, name, segment, email, credit_limit, payment_terms_days,
     historical_delay_avg, risk_band)
VALUES (:id, :name, :segment, :email, :credit_limit, :payment_terms_days,
        :historical_delay_avg, :risk_band)
ON DUPLICATE KEY UPDATE
    name=VALUES(name), segment=VALUES(segment), email=VALUES(email),
    credit_limit=VALUES(credit_limit), payment_terms_days=VALUES(payment_terms_days),
    historical_delay_avg=VALUES(historical_delay_avg), risk_band=VALUES(risk_band)
"""

_INSERT_INVOICE = """
INSERT INTO invoices
    (id, customer_id, amount, issue_date, due_date, status, paid_amount)
VALUES (:id, :customer_id, :amount, :issue_date, :due_date, :status, :paid_amount)
ON DUPLICATE KEY UPDATE
    status=VALUES(status), paid_amount=VALUES(paid_amount)
"""

_INSERT_PAYMENT = """
INSERT INTO payments (id, invoice_id, amount, date, method)
VALUES (:id, :invoice_id, :amount, :date, :method)
ON DUPLICATE KEY UPDATE amount=VALUES(amount)
"""

_INSERT_PAYMENT_IGNORE = """
INSERT IGNORE INTO payments (id, invoice_id, amount, date, method)
VALUES (:id, :invoice_id, :amount, :date, :method)
"""

_INSERT_ESCALATION = """
INSERT IGNORE INTO escalations
    (id, invoice_id, customer_id, sim_date, outstanding, risk_score, priority,
     severity, reason, status, resolved_date, resolution_note)
VALUES (:id, :invoice_id, :customer_id, :sim_date, :outstanding, :risk_score,
        :priority, :severity, :reason, :status, :resolved_date, :resolution_note)
"""

_INSERT_AUDIT = """
INSERT IGNORE INTO audit_log
    (id, timestamp, sim_date, agent, action, entity_type, entity_id, detail, metadata)
VALUES (:id, :timestamp, :sim_date, :agent, :action, :entity_type, :entity_id,
        :detail, :metadata)
"""

_INSERT_REPLY = """
INSERT IGNORE INTO customer_replies
    (id, invoice_id, customer_id, sim_date, text, intent, status, handled_action)
VALUES (:id, :invoice_id, :customer_id, :sim_date, :text, :intent, :status,
        :handled_action)
"""

_INSERT_PROMISE = """
INSERT IGNORE INTO promises
    (id, invoice_id, customer_id, created_date, amount, due_date, status)
VALUES (:id, :invoice_id, :customer_id, :created_date, :amount, :due_date, :status)
"""

_INSERT_REMITTANCE = """
INSERT IGNORE INTO remittances
    (id, date, payer_name, amount, reference_text, customer_id,
     intended_invoice_id, status, matched_invoice_id, match_method)
VALUES (:id, :date, :payer_name, :amount, :reference_text, :customer_id,
        :intended_invoice_id, :status, :matched_invoice_id, :match_method)
"""

_INSERT_DISPUTE = """
INSERT IGNORE INTO disputes
    (id, invoice_id, customer_id, opened_date, reason_category, reason_text,
     status, resolved_date, resolution_note)
VALUES (:id, :invoice_id, :customer_id, :opened_date, :reason_category,
        :reason_text, :status, :resolved_date, :resolution_note)
"""

_INSERT_HOLD = """
INSERT INTO credit_holds (customer_id, held_date, utilization_at_hold, released_date)
VALUES (:customer_id, :held_date, :utilization_at_hold, :released_date)
ON DUPLICATE KEY UPDATE
    held_date=VALUES(held_date),
    utilization_at_hold=VALUES(utilization_at_hold),
    released_date=VALUES(released_date)
"""
