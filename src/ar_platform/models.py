"""Domain models for the AR ledger.

Plain dataclasses (no ORM) keep the domain layer dependency-free and easy to
serialize to/from CSV and MySQL rows. Business logic that belongs to an entity
(e.g. how overdue an invoice is) lives here as methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


def _as_date(value) -> date:
    """Coerce a DB/CSV value to a date (MySQL returns date objects, CSV strings)."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _as_date_str(value) -> str | None:
    """Coerce a date-or-string-or-None to an ISO date string (or None)."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class Segment(StrEnum):
    """Customer segment — drives credit terms and baseline payment behavior."""

    ENTERPRISE = "enterprise"
    MIDMARKET = "midmarket"
    SMB = "smb"


class RiskBand(StrEnum):
    """Coarse credit-risk classification assigned at onboarding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvoiceStatus(StrEnum):
    OPEN = "open"            # issued, not yet due
    OVERDUE = "overdue"      # past due date, unpaid
    PARTIAL = "partial"      # partially paid
    PAID = "paid"            # settled in full
    DISPUTED = "disputed"    # customer has raised a dispute


# Aging buckets used across KPIs and prioritization (days past due).
AGING_BUCKETS = ["current", "1-30", "31-60", "61-90", "90+"]


@dataclass
class Customer:
    id: str
    name: str
    segment: Segment
    email: str
    credit_limit: float
    payment_terms_days: int          # net-N terms (e.g. 30)
    historical_delay_avg: float      # mean days late, historically
    risk_band: RiskBand

    def to_row(self) -> dict:
        d = self.__dict__.copy()
        d["segment"] = self.segment.value
        d["risk_band"] = self.risk_band.value
        return d

    @classmethod
    def from_row(cls, row: dict) -> Customer:
        return cls(
            id=row["id"],
            name=row["name"],
            segment=Segment(row["segment"]),
            email=row["email"],
            credit_limit=float(row["credit_limit"]),
            payment_terms_days=int(row["payment_terms_days"]),
            historical_delay_avg=float(row["historical_delay_avg"]),
            risk_band=RiskBand(row["risk_band"]),
        )


@dataclass
class Invoice:
    id: str
    customer_id: str
    amount: float
    issue_date: date
    due_date: date
    status: InvoiceStatus = InvoiceStatus.OPEN
    paid_amount: float = 0.0

    @property
    def outstanding(self) -> float:
        return round(self.amount - self.paid_amount, 2)

    def days_overdue(self, as_of: date) -> int:
        """Days past due as of a given date (0 if not yet due)."""
        return max(0, (as_of - self.due_date).days)

    def aging_bucket(self, as_of: date) -> str:
        d = self.days_overdue(as_of)
        if d == 0:
            return "current"
        if d <= 30:
            return "1-30"
        if d <= 60:
            return "31-60"
        if d <= 90:
            return "61-90"
        return "90+"

    def is_open(self) -> bool:
        return self.status in (
            InvoiceStatus.OPEN,
            InvoiceStatus.OVERDUE,
            InvoiceStatus.PARTIAL,
            InvoiceStatus.DISPUTED,
        )

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "amount": self.amount,
            "issue_date": self.issue_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "status": self.status.value,
            "paid_amount": self.paid_amount,
        }

    @classmethod
    def from_row(cls, row: dict) -> Invoice:
        return cls(
            id=row["id"],
            customer_id=row["customer_id"],
            amount=float(row["amount"]),
            issue_date=_as_date(row["issue_date"]),
            due_date=_as_date(row["due_date"]),
            status=InvoiceStatus(row["status"]),
            paid_amount=float(row["paid_amount"]),
        )


@dataclass
class Payment:
    id: str
    invoice_id: str
    amount: float
    date: date
    method: str

    def to_row(self) -> dict:
        d = self.__dict__.copy()
        d["date"] = self.date.isoformat()
        return d

    @classmethod
    def from_row(cls, row: dict) -> Payment:
        return cls(
            id=row["id"],
            invoice_id=row["invoice_id"],
            amount=float(row["amount"]),
            date=_as_date(row["date"]),
            method=row["method"],
        )


class ReplyIntent(StrEnum):
    """What a customer's free-text reply is asking for (set by the Inbox agent)."""

    PROMISE_TO_PAY = "promise_to_pay"      # commits to pay by a date
    EXTENSION_REQUEST = "extension_request"  # asks for more time / a plan
    DISPUTE = "dispute"                     # contests the invoice
    ALREADY_PAID = "already_paid"           # claims payment already made
    INFO_REQUEST = "info_request"           # wants a copy / details
    OTHER = "other"                         # unclear -> human


@dataclass
class CustomerReply:
    """An inbound free-text message from a customer about an invoice."""

    id: str
    invoice_id: str
    customer_id: str
    sim_date: str
    text: str
    intent: str | None = None       # set once classified
    status: str = "new"             # new | handled
    handled_action: str = ""

    def to_row(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_row(cls, row: dict) -> CustomerReply:
        return cls(
            id=row["id"],
            invoice_id=row["invoice_id"],
            customer_id=row["customer_id"],
            sim_date=_as_date_str(row["sim_date"]),
            text=row["text"],
            intent=row["intent"],
            status=row["status"],
            handled_action=row["handled_action"] or "",
        )


class PromiseStatus(StrEnum):
    PENDING = "pending"    # agreed, not yet due
    KEPT = "kept"          # paid in full by the promised date
    BROKEN = "broken"      # promised date passed without full payment


@dataclass
class PromiseToPay:
    """A tracked commitment negotiated with a customer."""

    id: str
    invoice_id: str
    customer_id: str
    created_date: str
    amount: float
    due_date: str          # the promised payment date
    status: PromiseStatus = PromiseStatus.PENDING

    def to_row(self) -> dict:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d

    @classmethod
    def from_row(cls, row: dict) -> PromiseToPay:
        return cls(
            id=row["id"],
            invoice_id=row["invoice_id"],
            customer_id=row["customer_id"],
            created_date=_as_date_str(row["created_date"]),
            amount=float(row["amount"]),
            due_date=_as_date_str(row["due_date"]),
            status=PromiseStatus(row["status"]),
        )


class EscalationStatus(StrEnum):
    PENDING = "pending"      # awaiting a human decision
    APPROVED = "approved"    # human authorized the collection action
    REJECTED = "rejected"    # human declined / put on hold


@dataclass
class Escalation:
    """A high-stakes case routed to a human for approval (HITL guardrail)."""

    id: str
    invoice_id: str
    customer_id: str
    sim_date: str
    outstanding: float
    risk_score: float
    priority: float
    severity: str
    reason: str
    status: EscalationStatus = EscalationStatus.PENDING
    resolved_date: str | None = None
    resolution_note: str = ""

    def to_row(self) -> dict:
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d

    @classmethod
    def from_row(cls, row: dict) -> Escalation:
        return cls(
            id=row["id"],
            invoice_id=row["invoice_id"],
            customer_id=row["customer_id"],
            sim_date=_as_date_str(row["sim_date"]),
            outstanding=float(row["outstanding"]),
            risk_score=float(row["risk_score"]),
            priority=float(row["priority"]),
            severity=row["severity"],
            reason=row["reason"],
            status=EscalationStatus(row["status"]),
            resolved_date=_as_date_str(row["resolved_date"]),
            resolution_note=row["resolution_note"] or "",
        )


@dataclass
class AuditEntry:
    """Append-only record of a single agent action (Layer 3 uses this heavily)."""

    id: str
    timestamp: str          # ISO datetime string
    sim_date: str           # the simulation date the action occurred on
    agent: str
    action: str
    entity_type: str        # "invoice" | "customer" | ...
    entity_id: str
    detail: str = ""
    metadata: dict = field(default_factory=dict)
