"""Workflow orchestration.

The orchestrator owns *agent coordination*: it runs the Monitor -> Risk -> Comms
pipeline for a cycle, then persists any escalated cases into the human-approval
queue (deduplicating against already-open escalations). It is deliberately
separate from the simulation clock — the orchestrator decides *what the agents
do*, the simulation decides *how the world moves*.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ar_platform.agents import CommsAgent, MonitorAgent, RiskAgent
from ar_platform.agents.base import AgentContext, WorkItem
from ar_platform.agents.cash_app import CashAppAgent
from ar_platform.agents.credit import CreditAgent
from ar_platform.agents.dispute import DisputeAgent
from ar_platform.agents.inbox import InboxAgent
from ar_platform.agents.negotiator import NegotiatorAgent
from ar_platform.models import (
    Escalation,
    EscalationStatus,
    InvoiceStatus,
    PromiseStatus,
)


@dataclass
class CycleReport:
    flagged: int = 0
    scored: int = 0
    emails_sent: int = 0
    escalations_new: int = 0
    replies_handled: int = 0
    negotiated: int = 0
    disputes: int = 0
    promises_kept: int = 0
    promises_broken: int = 0
    cash_matched: int = 0
    cash_suspense: int = 0
    credit_holds_new: int = 0
    credit_holds_released: int = 0
    agent_summaries: list[str] = field(default_factory=list)
    top_items: list[WorkItem] = field(default_factory=list)


class Orchestrator:
    """Runs one collection cycle: inbound replies, promises, then dunning."""

    def __init__(
        self,
        monitor: MonitorAgent | None = None,
        risk: RiskAgent | None = None,
        comms: CommsAgent | None = None,
        inbox: InboxAgent | None = None,
        negotiator: NegotiatorAgent | None = None,
        cash_app: CashAppAgent | None = None,
        dispute: DisputeAgent | None = None,
        credit: CreditAgent | None = None,
    ):
        self.monitor = monitor or MonitorAgent()
        self.risk = risk or RiskAgent()
        self.comms = comms or CommsAgent()
        self.inbox = inbox or InboxAgent()
        self.negotiator = negotiator or NegotiatorAgent()
        self.cash_app = cash_app or CashAppAgent()
        self.dispute = dispute or DisputeAgent()
        self.credit = credit or CreditAgent()

    # -- inbound: classify + route customer replies ------------------------
    def _process_replies(self, ctx: AgentContext, report: CycleReport) -> None:
        invoices = {i.id: i for i in ctx.store.get_open_invoices()}
        for reply in ctx.store.get_new_replies():
            inv = invoices.get(reply.invoice_id)
            cust = ctx.store.get_customer(reply.customer_id)
            classification = self.inbox.classify(ctx, reply)
            intent = classification.intent

            if inv is None or cust is None:
                action = "stale_invoice"
            elif intent in ("extension_request", "promise_to_pay"):
                action = self.negotiator.handle(ctx, inv, cust, classification)
                report.negotiated += 1
            elif intent == "dispute":
                self.dispute.open_case(ctx, inv, cust, reply.text)
                self._escalate(ctx, reply.invoice_id, reply.customer_id,
                               inv.outstanding, "dispute", "customer disputed invoice")
                report.disputes += 1
                action = "disputed_hold"
            elif intent == "already_paid":
                self._reply_email(ctx, cust, reply.invoice_id,
                                  "Thanks — we're verifying your remittance",
                                  "Thank you for the note. We are checking our records "
                                  "and will confirm receipt of your payment shortly.")
                action = "ack_payment_claim"
            elif intent == "info_request":
                self._reply_email(ctx, cust, reply.invoice_id,
                                  f"Re-sending invoice {reply.invoice_id}",
                                  "As requested, please find the invoice details "
                                  "attached. Let us know if anything else is needed.")
                action = "resent_invoice"
            else:
                self._escalate(ctx, reply.invoice_id, reply.customer_id,
                               inv.outstanding if inv else 0.0, "review",
                               "unclassified customer reply")
                action = "escalated_other"

            ctx.store.mark_reply_handled(reply.id, intent, action)
            report.replies_handled += 1

    # -- track promises made in prior cycles -------------------------------
    def _track_promises(self, ctx: AgentContext, report: CycleReport) -> None:
        from datetime import date

        invoices = {i.id: i for i in ctx.store.get_invoices()}
        for promise in ctx.store.get_promises(PromiseStatus.PENDING):
            inv = invoices.get(promise.invoice_id)
            if inv is None:
                continue
            if inv.status == InvoiceStatus.PAID:
                ctx.store.update_promise_status(promise.id, PromiseStatus.KEPT)
                report.promises_kept += 1
                ctx.audit("negotiator", "promise_kept", "invoice", inv.id,
                          detail=f"promise {promise.id} settled")
            elif date.fromisoformat(promise.due_date) < ctx.sim_date:
                ctx.store.update_promise_status(promise.id, PromiseStatus.BROKEN)
                report.promises_broken += 1
                cust = ctx.store.get_customer(promise.customer_id)
                if cust:
                    self._escalate(ctx, inv.id, cust.id, inv.outstanding,
                                   "broken", "customer broke payment promise")
                ctx.audit("negotiator", "promise_broken", "invoice", inv.id,
                          detail=f"promise {promise.id} past due, re-escalated")

    def _reply_email(self, ctx, cust, invoice_id, subject, body) -> None:
        ctx.email.send(
            to=cust.email, subject=subject,
            body=f"Dear {cust.name},\n\n{body}\n\nRegards,\nAccounts Receivable Team",
            invoice_id=invoice_id, sim_date=ctx.sim_date.isoformat(),
        )

    def _escalate(self, ctx, invoice_id, customer_id, outstanding, severity, reason) -> None:
        if ctx.store.has_open_escalation(invoice_id):
            return
        ctx.store.add_escalation(
            Escalation(
                id=f"ESC-{uuid.uuid4().hex[:8]}",
                invoice_id=invoice_id,
                customer_id=customer_id,
                sim_date=ctx.sim_date.isoformat(),
                outstanding=outstanding,
                risk_score=0.0,
                priority=outstanding,
                severity=severity,
                reason=reason,
                status=EscalationStatus.PENDING,
            )
        )

    def run_cycle(self, ctx: AgentContext) -> CycleReport:
        report = CycleReport()

        # -1. Cash application first: match incoming remittances so freshly
        #     paid invoices drop out before anyone considers chasing them.
        cash = self.cash_app.run(ctx)
        report.cash_matched = cash.matched
        report.cash_suspense = cash.to_suspense
        report.agent_summaries.append(
            f"cash_app: matched {cash.matched}, suspense {cash.to_suspense}"
        )

        # 0. Inbound conversation: classify + route replies, then reconcile
        #    promises from earlier cycles. Both run before dunning so we don't
        #    chase invoices we've just agreed terms on or disputed.
        self._process_replies(ctx, report)
        self._track_promises(ctx, report)
        report.agent_summaries.append(
            f"inbox: handled {report.replies_handled} replies "
            f"({report.negotiated} negotiated, {report.disputes} disputes)"
        )

        # 0.5 Credit management: exposure check -> holds / releases.
        held, released = self.credit.run(ctx)
        report.credit_holds_new = held
        report.credit_holds_released = released

        # 1. Perceive: Monitor flags overdue invoices.
        monitor_res = self.monitor.run(ctx)
        report.flagged = len(monitor_res.items)
        report.agent_summaries.append(f"monitor: {monitor_res.summary}")

        # Skip invoices already sitting in the approval queue — don't double-act.
        pending = [
            w for w in monitor_res.items if not ctx.store.has_open_escalation(w.invoice_id)
        ]

        # 2. Decide: Risk scores + prioritizes.
        risk_res = self.risk.run(ctx, pending)
        report.scored = len(risk_res.items)
        report.agent_summaries.append(f"risk: {risk_res.summary}")

        # 3. Act: Comms emails or escalates.
        comms_res = self.comms.run(ctx, risk_res.items)
        report.agent_summaries.append(f"comms: {comms_res.summary}")

        # 4. Persist escalations to the human-approval queue.
        for item in comms_res.items:
            if item.action == "email_sent":
                report.emails_sent += 1
            elif item.escalated and not ctx.store.has_open_escalation(item.invoice_id):
                ctx.store.add_escalation(
                    Escalation(
                        id=f"ESC-{uuid.uuid4().hex[:8]}",
                        invoice_id=item.invoice_id,
                        customer_id=item.customer_id,
                        sim_date=ctx.sim_date.isoformat(),
                        outstanding=item.outstanding,
                        risk_score=item.risk_score or 0.0,
                        priority=item.priority or 0.0,
                        severity=item.severity,
                        reason=f"expected loss ${item.priority or 0:,.0f}",
                        status=EscalationStatus.PENDING,
                    )
                )
                report.escalations_new += 1

        ctx.store.commit()
        report.top_items = risk_res.items[:10]
        return report


def resolve_escalation(
    ctx: AgentContext,
    esc: Escalation,
    approve: bool,
    note: str = "",
) -> str:
    """Apply a human decision to an escalation (used by dashboard / auto-policy).

    Approval authorizes a final-notice collection email; rejection puts the case
    on hold. Either way the decision is audited.
    """
    status = EscalationStatus.APPROVED if approve else EscalationStatus.REJECTED
    ctx.store.update_escalation(
        esc.id, status, ctx.sim_date.isoformat(), note
    )

    # Suspense-cash cases: approving means the human worked out where the
    # money belongs (the world's ground truth stands in for that research)
    # and applies it; rejecting leaves it in suspense (e.g. refund pending).
    if esc.severity == "cash_app":
        if approve:
            rem = ctx.store.get_remittance(esc.invoice_id)  # esc holds the REM id
            if rem is not None and rem.status.value != "matched":
                from ar_platform.models import Payment, RemittanceStatus

                inv = next(
                    (i for i in ctx.store.get_open_invoices()
                     if i.id == rem.intended_invoice_id),
                    None,
                )
                if inv is not None:
                    ctx.erp.apply_payment(
                        inv,
                        Payment(
                            id=f"PAY-H{esc.id[-6:]}",
                            invoice_id=inv.id,
                            amount=min(rem.amount, inv.outstanding),
                            date=ctx.sim_date,
                            method="remittance",
                        ),
                    )
                    ctx.store.update_remittance(
                        rem.id, RemittanceStatus.MATCHED, inv.id, "human"
                    )
        ctx.audit(
            "human", "resolved_escalation", "escalation", esc.id,
            detail=f"suspense cash {'applied' if approve else 'held'}: {note}",
        )
        ctx.store.commit()
        return status.value

    # Dispute cases: approve = customer's claim upheld (credit memo),
    # reject = claim denied (dunning resumes). Handled by the dispute module.
    if esc.severity == "dispute":
        from ar_platform.agents.dispute import resolve_dispute_case

        dispute = ctx.store.get_open_dispute(esc.invoice_id)
        if dispute is not None:
            resolve_dispute_case(ctx, dispute, valid=approve, note=note)
        ctx.audit(
            "human", "resolved_escalation", "escalation", esc.id,
            detail=f"dispute {'upheld' if approve else 'denied'}: {note}",
        )
        ctx.store.commit()
        return status.value

    if approve:
        cust = ctx.store.get_customer(esc.customer_id)
        inv = next(
            (i for i in ctx.store.get_open_invoices() if i.id == esc.invoice_id), None
        )
        if cust and inv:
            from ar_platform.tools.templates import CollectionContext, render_dunning

            email_ctx = CollectionContext(
                customer_name=cust.name,
                invoice_id=inv.id,
                outstanding=inv.outstanding,
                days_overdue=inv.days_overdue(ctx.sim_date),
                due_date=inv.due_date.isoformat(),
                severity="final",
                payment_terms_days=cust.payment_terms_days,
            )
            draft = (
                ctx.llm.draft_collection_email(email_ctx)
                if ctx.llm is not None
                else render_dunning(email_ctx)
            )
            ctx.email.send(
                to=cust.email, subject=draft.subject, body=draft.body,
                invoice_id=inv.id, sim_date=ctx.sim_date.isoformat(),
            )
    ctx.audit(
        "human", "resolved_escalation", "escalation", esc.id,
        detail=f"{'approved' if approve else 'rejected'}: {note}",
    )
    ctx.store.commit()
    return status.value
