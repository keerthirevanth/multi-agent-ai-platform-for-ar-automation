"""Comms agent — the acting end of the pipeline.

For each prioritized invoice it either:

* **Escalates** to a human-approval queue when the case is high-stakes (large
  expected loss or a 'final notice' severity) — the human-in-the-loop guardrail;
  or
* **Auto-sends** a collection email whose tone matches the severity, drafted by
  the configured LLM backend (rules templates by default, Claude if enabled).

Every action is recorded through the ERP tool and the audit log.
"""

from __future__ import annotations

from ar_platform.agents.base import Agent, AgentContext, AgentResult, WorkItem
from ar_platform.tools.templates import CollectionContext, render_dunning

# A case jumps to human review if expected loss exceeds this, or it's a final
# notice. Tunable knob for the autonomy/oversight tradeoff.
ESCALATION_PRIORITY = 15_000.0

# Pre-due nudges go only to invoices the model considers genuinely risky —
# blanket-reminding every customer before every due date would train them to
# ignore us (and annoy the good payers).
PRE_DUE_RISK_THRESHOLD = 0.5


class CommsAgent(Agent):
    name = "comms"

    def __init__(
        self,
        escalation_priority: float = ESCALATION_PRIORITY,
        pre_due_risk_threshold: float = PRE_DUE_RISK_THRESHOLD,
    ):
        self.escalation_priority = escalation_priority
        self.pre_due_risk_threshold = pre_due_risk_threshold

    def run(self, ctx: AgentContext, items: list[WorkItem] | None = None) -> AgentResult:
        items = items or []
        sent = 0
        escalated = 0
        contacted = ctx.store.contacted_invoice_ids()

        for item in items:
            # Pre-due invoices are not delinquent: never escalate them, nudge
            # at most once, and only when the risk model flags them.
            if item.severity == "pre_due":
                if (item.risk_score or 0) < self.pre_due_risk_threshold:
                    item.action = "skipped_low_risk"
                    continue
                if item.invoice_id in contacted:
                    item.action = "skipped_already_contacted"
                    continue
                # falls through to the email path below

            high_stakes = (item.priority or 0) >= self.escalation_priority
            if item.severity != "pre_due" and (high_stakes or item.severity == "final"):
                item.escalated = True
                item.action = "escalated_for_approval"
                escalated += 1
                ctx.audit(
                    self.name,
                    "escalated",
                    "invoice",
                    item.invoice_id,
                    detail=(
                        f"priority=${item.priority or 0:,.0f}, severity={item.severity}"
                    ),
                    metadata={"reason": "high_stakes" if high_stakes else "final_notice"},
                )
                continue

            cust = ctx.store.get_customer(item.customer_id)
            inv = next(
                (i for i in ctx.store.get_open_invoices() if i.id == item.invoice_id),
                None,
            )
            if cust is None or inv is None:
                continue

            email_ctx = CollectionContext(
                customer_name=cust.name,
                invoice_id=item.invoice_id,
                outstanding=item.outstanding,
                days_overdue=item.days_overdue,
                due_date=inv.due_date.isoformat(),
                severity=item.severity,
                payment_terms_days=cust.payment_terms_days,
            )
            # Deterministic templates by default; a configured LLM upgrades
            # the wording (never the decision to send).
            if ctx.llm is not None:
                draft = ctx.llm.draft_collection_email(email_ctx)
            else:
                draft = render_dunning(email_ctx)
            ctx.email.send(
                to=cust.email,
                subject=draft.subject,
                body=draft.body,
                invoice_id=item.invoice_id,
                sim_date=ctx.sim_date.isoformat(),
            )
            item.action = "email_sent"
            sent += 1
            ctx.audit(
                self.name,
                "email_sent",
                "invoice",
                item.invoice_id,
                detail=f"severity={item.severity} to {cust.email}",
                metadata={
                    "subject": draft.subject,
                    "backend": ctx.llm.name if ctx.llm else "templates",
                },
            )

        return AgentResult(
            agent=self.name,
            summary=f"sent {sent} emails, escalated {escalated} for approval",
            items=items,
        )
