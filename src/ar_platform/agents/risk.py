"""Risk agent — scores and prioritizes flagged invoices.

Uses the ML risk model to estimate each invoice's probability of becoming a
collection problem, then computes a *priority* = dollar exposure x default
probability. This is the number that lets the department focus effort where the
expected loss is largest, rather than just chasing the oldest or biggest
invoice.
"""

from __future__ import annotations

from ar_platform.agents.base import Agent, AgentContext, AgentResult, WorkItem


class RiskAgent(Agent):
    name = "risk"

    def run(self, ctx: AgentContext, items: list[WorkItem] | None = None) -> AgentResult:
        items = items or []
        for item in items:
            inv = next(
                (i for i in ctx.store.get_open_invoices() if i.id == item.invoice_id),
                None,
            )
            cust = ctx.store.get_customer(item.customer_id)
            if inv is None or cust is None:
                continue

            prob = ctx.risk_model.predict_default_prob(inv, cust)
            item.risk_score = round(prob, 4)
            # Expected loss: how much is at stake weighted by likelihood of loss.
            item.priority = round(item.outstanding * prob, 2)
            item.notes.append(f"risk={prob:.2f}")

            ctx.audit(
                self.name,
                "scored_risk",
                "invoice",
                item.invoice_id,
                detail=f"risk={prob:.2f}, priority=${item.priority:,.0f}",
                metadata={"risk_score": item.risk_score, "priority": item.priority},
            )

        # Highest expected-loss first — this is the collection worklist order.
        items.sort(key=lambda w: (w.priority or 0), reverse=True)
        avg = (
            sum(i.risk_score or 0 for i in items) / len(items) if items else 0.0
        )
        return AgentResult(
            agent=self.name,
            summary=f"scored {len(items)} invoices (avg risk {avg:.2f})",
            items=items,
        )
