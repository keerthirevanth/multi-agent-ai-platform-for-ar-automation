"""Negotiator agent — the clearest example of bounded agency in the system.

When a customer asks for more time or proposes a plan, the reasoning is:

1. A **proposer** suggests terms — a real LLM reasoning over the case, or the
   deterministic policy's own recommendation when no LLM is configured.
2. The deterministic **NegotiationPolicy** re-validates the proposal, clamping
   anything beyond the customer's risk-tiered authority and escalating cases
   outside the department's authority (large exposure, repeat broken promises).
3. The bounded outcome is applied: extend the invoice's due date and record a
   tracked promise-to-pay, or route to a human.

This is the "LLM proposes, guardrails dispose, humans handle the edges" pattern
that lets a non-deterministic reasoner operate safely inside finance.
"""

from __future__ import annotations

import uuid

from ar_platform.agents.base import AgentContext
from ar_platform.dialogue import (
    NegotiationAction,
    NegotiationCase,
    NegotiationPolicy,
    ReplyClassification,
)
from ar_platform.models import (
    Customer,
    Escalation,
    EscalationStatus,
    Invoice,
    PromiseToPay,
)


class NegotiatorAgent:
    name = "negotiator"

    def __init__(self, policy: NegotiationPolicy | None = None):
        self.policy = policy or NegotiationPolicy()

    def handle(
        self,
        ctx: AgentContext,
        inv: Invoice,
        cust: Customer,
        classification: ReplyClassification,
    ) -> str:
        case = NegotiationCase(
            invoice_id=inv.id,
            customer_id=cust.id,
            risk_band=cust.risk_band,
            outstanding=inv.outstanding,
            days_overdue=inv.days_overdue(ctx.sim_date),
            requested_extension_days=classification.extension_days or 0,
            prior_broken_promises=ctx.store.broken_promise_count(inv.id),
            as_of=ctx.sim_date,
        )

        # Proposer: LLM reasons, then the policy re-validates (never trusts it
        # blindly). No LLM -> the policy's own decision.
        if ctx.llm is not None:
            proposal = self.policy.validate(case, ctx.llm.propose_negotiation(case))
            proposer = ctx.llm.name
        else:
            proposal = self.policy.decide(case)
            proposer = "policy"

        if proposal.action in (NegotiationAction.ACCEPT, NegotiationAction.COUNTER):
            new_due = proposal.terms.new_due_date
            ctx.store.update_invoice_due_date(inv.id, new_due.isoformat())
            ctx.store.add_promise(
                PromiseToPay(
                    id=f"PRM-{uuid.uuid4().hex[:8]}",
                    invoice_id=inv.id,
                    customer_id=cust.id,
                    created_date=ctx.sim_date.isoformat(),
                    amount=proposal.terms.promise_amount,
                    due_date=new_due.isoformat(),
                )
            )
            action = f"{proposal.action.value}_{proposal.terms.extension_days}d"
        elif proposal.action == NegotiationAction.ESCALATE:
            self._escalate(ctx, inv, cust, proposal.rationale)
            action = "escalated"
        else:  # REJECT
            action = "rejected"

        ctx.audit(
            self.name,
            action,
            "invoice",
            inv.id,
            detail=proposal.rationale,
            metadata={"proposer": proposer, "action": proposal.action.value},
        )
        return action

    def _escalate(self, ctx: AgentContext, inv: Invoice, cust: Customer, reason: str) -> None:
        if ctx.store.has_open_escalation(inv.id):
            return
        risk = ctx.risk_model.predict_default_prob(inv, cust) if ctx.risk_model else 0.0
        ctx.store.add_escalation(
            Escalation(
                id=f"ESC-{uuid.uuid4().hex[:8]}",
                invoice_id=inv.id,
                customer_id=cust.id,
                sim_date=ctx.sim_date.isoformat(),
                outstanding=inv.outstanding,
                risk_score=risk,
                priority=round(inv.outstanding * risk, 2),
                severity="negotiation",
                reason=f"negotiation: {reason}",
                status=EscalationStatus.PENDING,
            )
        )
