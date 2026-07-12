"""Agent foundations: shared context, work items, and the Agent contract."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime

from ar_platform.data.store import Store
from ar_platform.llm.interface import LLMClient
from ar_platform.models import AuditEntry
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool
from ar_platform.tools.ml_risk import RiskModel


@dataclass
class WorkItem:
    """A single invoice moving through the agent pipeline, gaining annotations."""

    invoice_id: str
    customer_id: str
    customer_name: str
    outstanding: float
    days_overdue: int
    severity: str                      # set by Monitor
    aging_bucket: str
    risk_score: float | None = None    # set by Risk
    priority: float | None = None      # set by Risk (exposure x risk)
    action: str | None = None          # set by Comms
    escalated: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Shared dependencies handed to every agent for a run.

    ``llm`` is None in fully deterministic operation (the default); agents fall
    back to fixed templates. A real backend (Claude) upgrades drafting and,
    in the agentic layer, decision-making over unstructured input.
    """

    store: Store
    llm: LLMClient | None
    email: EmailTool
    erp: ERPTool
    risk_model: RiskModel
    sim_date: date

    def audit(
        self,
        agent: str,
        action: str,
        entity_type: str,
        entity_id: str,
        detail: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Append an action to the audit log."""
        self.store.add_audit(
            AuditEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                sim_date=self.sim_date.isoformat(),
                agent=agent,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                detail=detail,
                metadata=metadata or {},
            )
        )


@dataclass
class AgentResult:
    """What an agent returns after a run."""

    agent: str
    summary: str
    items: list[WorkItem] = field(default_factory=list)


class Agent(ABC):
    """Base class enforcing the perceive -> decide -> act shape."""

    name: str = "agent"

    @abstractmethod
    def run(self, ctx: AgentContext, items: list[WorkItem] | None = None) -> AgentResult:
        """Execute one cycle. May consume upstream ``items`` and/or read state."""
