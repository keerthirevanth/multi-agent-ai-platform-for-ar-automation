"""The simulation clock — this is 'how the base case updates'.

Each ``tick`` advances the calendar by N days and does two things:

1. **World update** (the environment): invoices age into 'overdue', customers
   pay some of what they owe, and fresh invoices are issued. Payment likelihood
   is driven by customer risk *and* whether the collection agents have contacted
   them — so the agents' work has a measurable effect on cash collected.
2. **Agent cycle** (the department): the orchestrator runs Monitor -> Risk ->
   Comms and files escalations.

Everything is seeded, so a given (base case, seed, tick count) is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from ar_platform.agents.base import AgentContext
from ar_platform.config import settings
from ar_platform.data.store import Store
from ar_platform.llm import get_llm
from ar_platform.models import (
    Invoice,
    InvoiceStatus,
    Payment,
    PromiseStatus,
    Remittance,
    RemittanceStatus,
    RiskBand,
)
from ar_platform.orchestrator import CycleReport, Orchestrator, resolve_escalation
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool
from ar_platform.tools.ml_risk import RiskModel, train_risk_model

# Baseline daily probability that an open invoice gets paid, by risk band.
_DAILY_PAY_PROB = {RiskBand.LOW: 0.06, RiskBand.MEDIUM: 0.035, RiskBand.HIGH: 0.02}
# Multiplier applied once a customer has been contacted about an invoice.
_CONTACT_BOOST = 1.8
# Daily probability an invoice is paid BEFORE its due date (some customers pay
# early; a pre-due nudge multiplies this by the contact boost).
_EARLY_PAY_DAILY = 0.008
# Payment-likelihood multiplier once a customer has an agreed promise-to-pay.
_PROMISE_BOOST = 2.2
# Remittance reference quality: share arriving with a clean invoice reference
# (straight-through processed). The rest need cash-application matching.
_CLEAN_REF_RATE = 0.70
# Of the messy remainder: mangled-but-present hints vs. bare name-only wires.
_NAME_NOISE = [
    lambda n: n,                                    # exact name
    lambda n: n.upper(),                            # bank shouting case
    lambda n: n.replace(",", "").replace(".", ""),  # punctuation stripped
    lambda n: n.split(" and ")[0].split(",")[0],    # truncated legal name
]
# Expected new invoices issued per customer per 30 days.
_NEW_INVOICE_RATE = 0.5


@dataclass
class TickReport:
    sim_date: str
    days_advanced: int
    new_invoices: int = 0
    payments_count: int = 0
    payments_amount: float = 0.0
    open_ar: float = 0.0
    cycle: CycleReport | None = None
    approvals: int = 0
    rejections: int = 0
    agent_summaries: list[str] = field(default_factory=list)


class Simulation:
    """Drives the AR world forward and runs the finance department each tick.

    ``auto_approve_rate`` simulates a human clearing the escalation queue: that
    fraction of pending escalations is approved each tick (the rest rejected),
    so the human-in-the-loop path exercises end-to-end without a live user. Set
    to ``None`` to leave escalations pending for a dashboard operator to resolve.
    """

    def __init__(
        self,
        store: Store,
        seed: int | None = None,
        auto_approve_rate: float | None = 0.6,
        retrain_each_tick: bool = False,
        agents_enabled: bool = True,
        email_tool: EmailTool | None = None,
    ):
        self.store = store
        self.rng = random.Random(settings.seed if seed is None else seed)
        self.llm = get_llm()
        self.email = email_tool or EmailTool()
        self.erp = ERPTool(store)
        self.orchestrator = Orchestrator()
        self.auto_approve_rate = auto_approve_rate
        self.retrain_each_tick = retrain_each_tick
        # agents_enabled=False is the control arm of the uplift experiment:
        # the world still moves (aging, baseline payments, new invoices) but
        # no agent ever acts, so no invoice gets the contacted payment boost.
        self.agents_enabled = agents_enabled
        self.sim_date: date = settings.base_case_today
        self.model: RiskModel | None = self._train() if agents_enabled else None

    def _train(self) -> RiskModel:
        return train_risk_model(
            self.store.get_customers(),
            self.store.get_invoices(),
            self.store.get_payments(),
            as_of=self.sim_date,
        )

    def context(self) -> AgentContext:
        """Public accessor for the current agent context (used by the dashboard)."""
        return self._ctx()

    def resolve(self, esc, approve: bool, note: str = "manual") -> str:
        """Apply a human decision to an escalation from an external UI."""
        return resolve_escalation(self._ctx(), esc, approve, note)

    def _ctx(self) -> AgentContext:
        return AgentContext(
            store=self.store,
            llm=self.llm,
            email=self.email,
            erp=self.erp,
            risk_model=self.model,
            sim_date=self.sim_date,
        )

    # -- world update ------------------------------------------------------
    def _age_invoices(self) -> None:
        for inv in self.store.get_open_invoices():
            if inv.status == InvoiceStatus.OPEN and inv.due_date < self.sim_date:
                inv.status = InvoiceStatus.OVERDUE
                self.store.upsert_invoice(inv)
        self.store.commit()

    def _receive_cash(self, inv, cust, amount: float, rem_seq: int) -> float:
        """A customer pays: cash arrives as a bank remittance.

        ~70% carry a clean invoice reference and are straight-through processed
        (applied immediately — a clerk or OCR handles these trivially, so both
        experiment arms get them). The rest arrive with mangled or missing
        references and sit unapplied until the cash-application agent matches
        them. Returns the amount actually applied to the ledger now.
        """
        clean = self.rng.random() < _CLEAN_REF_RATE
        if clean:
            reference = f"payment for {inv.id}"
            payer = cust.name
        else:
            payer = self.rng.choice(_NAME_NOISE)(cust.name)
            reference = self.rng.choice([
                f"inv {inv.id.split('-')[1].lstrip('0')}",   # unparseable hint
                f"po {self.rng.randint(10000, 99999)}",       # wrong reference
                "payment on account",
                "",
            ])
        rem = Remittance(
            id=f"REM-{rem_seq:06d}",
            date=self.sim_date.isoformat(),
            payer_name=payer,
            amount=amount,
            reference_text=reference,
            customer_id=cust.id,
            intended_invoice_id=inv.id,
            status=RemittanceStatus.MATCHED if clean else RemittanceStatus.UNMATCHED,
            matched_invoice_id=inv.id if clean else None,
            match_method="reference" if clean else "",
        )
        self.store.add_remittance(rem)
        if clean:
            pay = Payment(
                id=f"PAY-{rem_seq:06d}R",
                invoice_id=inv.id,
                amount=amount,
                date=self.sim_date,
                method="remittance",
            )
            self.erp.apply_payment(inv, pay)
            return amount
        return 0.0

    def _collect_payments(self, days: int) -> tuple[int, float]:
        contacted = self.store.contacted_invoice_ids()
        # Invoices under an agreed promise-to-pay: the customer committed, so
        # they pay more reliably. This is what makes negotiation worth doing.
        promised = {
            p.invoice_id for p in self.store.get_promises(PromiseStatus.PENDING)
        }
        # Customers who already remitted (cash pending application) won't pay
        # the same invoice twice, even though our books still show it open.
        already_remitted = self.store.invoices_with_pending_remittance()
        customers = {c.id: c for c in self.store.get_customers()}
        seq = self.store.next_seq("remittances", "REM")
        count = 0
        total = 0.0

        for inv in self.store.get_open_invoices():
            cust = customers.get(inv.customer_id)
            if cust is None or inv.status == InvoiceStatus.DISPUTED:
                continue
            if inv.id in already_remitted:
                continue  # they've paid; the cash just isn't applied yet

            if inv.due_date >= self.sim_date and inv.status == InvoiceStatus.OPEN:
                # Not yet due: occasional early settlement, meaningfully more
                # likely if a pre-due nudge has landed (contact boost).
                daily = _EARLY_PAY_DAILY
                if inv.id in contacted:
                    daily = min(0.95, daily * _CONTACT_BOOST)
                if self.rng.random() < 1 - (1 - daily) ** days:
                    total += self._receive_cash(inv, cust, inv.outstanding, seq)
                    seq += 1
                    count += 1
                continue

            daily = _DAILY_PAY_PROB[cust.risk_band]
            if inv.id in contacted:
                daily = min(0.95, daily * _CONTACT_BOOST)
            if inv.id in promised:
                daily = min(0.97, daily * _PROMISE_BOOST)
            # Probability of at least one payment event across the tick window.
            prob = 1 - (1 - daily) ** days
            if self.rng.random() >= prob:
                continue

            # Mostly full settlement; occasionally a partial payment.
            if self.rng.random() < 0.2:
                amount = round(inv.outstanding * self.rng.uniform(0.3, 0.7), 2)
            else:
                amount = inv.outstanding
            total += self._receive_cash(inv, cust, amount, seq)
            seq += 1
            count += 1

        return count, round(total, 2)

    def _issue_new_invoices(self, days: int) -> int:
        # Customers on credit hold get no new credit sales until released.
        held = {h.customer_id for h in self.store.active_credit_holds()}
        customers = [c for c in self.store.get_customers() if c.id not in held]
        if not customers:
            return 0
        seq = self.store.next_seq("invoices", "INV")
        expected = len(customers) * _NEW_INVOICE_RATE * (days / 30.0)
        # Poisson-ish count via a Gaussian approximation (std = sqrt(mean)).
        n_new = max(0, int(round(self.rng.gauss(expected, expected**0.5))))

        from ar_platform.data.generator import _SEGMENT_PARAMS

        created = 0
        for _ in range(n_new):
            cust = self.rng.choice(customers)
            _, terms, (lo, hi) = _SEGMENT_PARAMS[cust.segment]
            inv = Invoice(
                id=f"INV-{seq:05d}",
                customer_id=cust.id,
                amount=round(self.rng.uniform(lo, hi), 2),
                issue_date=self.sim_date,
                due_date=self.sim_date + timedelta(days=terms),
                status=InvoiceStatus.OPEN,
            )
            self.store.upsert_invoice(inv)
            seq += 1
            created += 1
        self.store.commit()
        return created

    def _generate_replies(self) -> int:
        """The customer side: some contacted, still-open invoices write back.

        Only invoices we have actually emailed can produce a reply, and each
        invoice replies at most once in its life — so the control arm (no
        emails) generates no replies, and the agentic loop only activates when
        the department is actually corresponding.
        """
        from ar_platform.data.reply_sim import REPLY_PROB, generate_reply

        contacted = self.store.contacted_invoice_ids()
        if not contacted:
            return 0
        customers = {c.id: c for c in self.store.get_customers()}
        seq = self.store.next_seq("customer_replies", "RLY")
        created = 0
        for inv in self.store.get_open_invoices():
            if inv.id not in contacted:
                continue
            if inv.status == InvoiceStatus.DISPUTED or self.store.has_reply(inv.id):
                continue
            if self.rng.random() >= REPLY_PROB:
                continue
            cust = customers.get(inv.customer_id)
            if cust is None:
                continue
            self.store.add_reply(generate_reply(self.rng, inv, cust, self.sim_date, seq))
            seq += 1
            created += 1
        self.store.commit()
        return created

    def _clear_escalation_queue(self) -> tuple[int, int]:
        if self.auto_approve_rate is None:
            return 0, 0
        from ar_platform.models import EscalationStatus

        approvals = rejections = 0
        # Iterate in a content-deterministic order (invoice/case id), NOT by
        # escalation id: escalation ids are random UUIDs, so id order would
        # assign this loop's RNG draws to different cases run-to-run and break
        # seeded reproducibility. At most one pending escalation exists per
        # case id, so this key is unique.
        pending = sorted(
            self.store.get_escalations(EscalationStatus.PENDING),
            key=lambda e: e.invoice_id,
        )
        for esc in pending:
            approve = self.rng.random() < self.auto_approve_rate
            resolve_escalation(
                self._ctx(), esc, approve,
                note="auto-policy (simulated approver)",
            )
            if approve:
                approvals += 1
            else:
                rejections += 1
        return approvals, rejections

    # -- the tick ----------------------------------------------------------
    def tick(self, days: int = 7) -> TickReport:
        """Advance the world by ``days`` and run one finance-department cycle."""
        self.sim_date = self.sim_date + timedelta(days=days)

        # 1. World moves.
        self._age_invoices()
        pay_count, pay_amount = self._collect_payments(days)
        new_inv = self._issue_new_invoices(days)
        # Customers respond to earlier dunning (only meaningful with agents on).
        if self.agents_enabled:
            self._generate_replies()

        if self.retrain_each_tick and self.agents_enabled:
            self.model = self._train()

        # 2. Department acts (skipped entirely in the control arm).
        if self.agents_enabled:
            cycle = self.orchestrator.run_cycle(self._ctx())
            approvals, rejections = self._clear_escalation_queue()
        else:
            cycle = CycleReport()
            approvals = rejections = 0

        open_ar = round(
            sum(i.outstanding for i in self.store.get_open_invoices()), 2
        )
        return TickReport(
            sim_date=self.sim_date.isoformat(),
            days_advanced=days,
            new_invoices=new_inv,
            payments_count=pay_count,
            payments_amount=pay_amount,
            open_ar=open_ar,
            cycle=cycle,
            approvals=approvals,
            rejections=rejections,
            agent_summaries=cycle.agent_summaries,
        )

    def run(self, ticks: int = 8, days_per_tick: int = 7) -> list[TickReport]:
        return [self.tick(days_per_tick) for _ in range(ticks)]
