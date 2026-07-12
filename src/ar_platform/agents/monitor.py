"""Monitor agent — the top of the pipeline.

Scans the open ledger, identifies invoices that need attention, and assigns a
*severity* based on how far past due they are. It produces the WorkItems that
the Risk and Comms agents downstream consume.
"""

from __future__ import annotations

from ar_platform.agents.base import Agent, AgentContext, AgentResult, WorkItem
from ar_platform.models import InvoiceStatus

# Days-overdue thresholds -> collection severity (drives Comms tone / escalation).
_SEVERITY_BANDS = [
    (1, 30, "reminder"),
    (31, 60, "overdue"),
    (61, 90, "urgent"),
    (91, 10_000, "final"),
]

# Open invoices due within this many days are flagged "pre_due" so high-risk
# ones can be nudged BEFORE they go late — proactive outreach is where real AR
# teams actually move DSO.
PRE_DUE_WINDOW_DAYS = 7


def _severity_for(days_overdue: int) -> str | None:
    for lo, hi, sev in _SEVERITY_BANDS:
        if lo <= days_overdue <= hi:
            return sev
    return None  # not yet due


class MonitorAgent(Agent):
    name = "monitor"

    def run(self, ctx: AgentContext, items=None) -> AgentResult:
        flagged: list[WorkItem] = []
        for inv in ctx.store.get_open_invoices():
            # Don't chase invoices we've paused: disputes are on hold pending
            # resolution, and invoices under an agreed promise-to-pay have their
            # own timeline. This is the conversation changing our behavior.
            if inv.status == InvoiceStatus.DISPUTED:
                continue
            if ctx.store.has_active_promise(inv.id):
                continue

            days = inv.days_overdue(ctx.sim_date)
            severity = _severity_for(days)
            if severity is None:
                # Not yet due: flag as pre_due if it falls due soon, so the
                # Risk agent can decide whether it deserves a proactive nudge.
                days_to_due = (inv.due_date - ctx.sim_date).days
                if 0 <= days_to_due <= PRE_DUE_WINDOW_DAYS:
                    severity = "pre_due"
                else:
                    continue  # comfortably current — nothing to do
            cust = ctx.store.get_customer(inv.customer_id)
            item = WorkItem(
                invoice_id=inv.id,
                customer_id=inv.customer_id,
                customer_name=cust.name if cust else inv.customer_id,
                outstanding=inv.outstanding,
                days_overdue=days,
                severity=severity,
                aging_bucket=inv.aging_bucket(ctx.sim_date),
            )
            flagged.append(item)
            ctx.audit(
                self.name,
                "flagged_pre_due" if severity == "pre_due" else "flagged_overdue",
                "invoice",
                inv.id,
                detail=(
                    f"due in {(inv.due_date - ctx.sim_date).days}d"
                    if severity == "pre_due"
                    else f"{days}d overdue, severity={severity}"
                ),
                metadata={"outstanding": inv.outstanding, "severity": severity},
            )

        flagged.sort(key=lambda w: w.days_overdue, reverse=True)
        return AgentResult(
            agent=self.name,
            summary=f"flagged {len(flagged)} overdue invoices",
            items=flagged,
        )
