"""Cash-application agent — matches incoming bank remittances to invoices.

The money has already arrived; the work is figuring out *which invoice* it
pays. Matching runs three deterministic passes, in decreasing confidence:

1. **reference** — the remittance text contains a parseable invoice id;
2. **amount** — the payer resolves to a customer (fuzzy name match against the
   master file) who has exactly one open invoice with that exact outstanding
   amount;
3. **single_open** — the payer resolves to a customer with exactly one open
   invoice; the cash is applied there (as a partial payment if short).

Anything else goes to **suspense** for a human (large amounts are escalated).
The matcher never reads the remittance's ground-truth ``customer_id`` /
``intended_invoice_id`` fields — those exist only so tests can measure match
accuracy honestly.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher

from ar_platform.agents.base import AgentContext
from ar_platform.models import (
    Escalation,
    EscalationStatus,
    Payment,
    RemittanceStatus,
)

_INV_REF = re.compile(r"INV-\d{5}")
_NAME_MATCH_THRESHOLD = 0.82
# Suspense items above this go straight to the human queue.
SUSPENSE_ESCALATION_AMOUNT = 10_000.0


@dataclass
class CashAppReport:
    matched: int = 0
    to_suspense: int = 0
    by_method: dict | None = None


class CashAppAgent:
    name = "cash_app"

    def _resolve_customer(self, payer_name: str, customers: list) -> object | None:
        """Fuzzy-match the bank payer name to the customer master file."""
        best, best_score = None, 0.0
        low = payer_name.lower().strip()
        for c in customers:
            score = SequenceMatcher(None, low, c.name.lower()).ratio()
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= _NAME_MATCH_THRESHOLD else None

    def run(self, ctx: AgentContext) -> CashAppReport:
        report = CashAppReport(by_method={"reference": 0, "amount": 0, "single_open": 0})
        customers = ctx.store.get_customers()
        open_by_id = {i.id: i for i in ctx.store.get_open_invoices()}

        for rem in ctx.store.get_remittances(RemittanceStatus.UNMATCHED):
            inv, method = None, ""

            # Pass 1: explicit invoice reference in the remittance text.
            m = _INV_REF.search(rem.reference_text or "")
            if m and m.group(0) in open_by_id:
                inv, method = open_by_id[m.group(0)], "reference"

            # Pass 2 / 3: resolve payer -> customer, then match by amount.
            if inv is None:
                cust = self._resolve_customer(rem.payer_name, customers)
                if cust is not None:
                    open_invs = [
                        i for i in open_by_id.values() if i.customer_id == cust.id
                    ]
                    exact = [
                        i for i in open_invs if abs(i.outstanding - rem.amount) < 0.01
                    ]
                    if len(exact) == 1:
                        inv, method = exact[0], "amount"
                    elif len(open_invs) == 1:
                        inv, method = open_invs[0], "single_open"

            if inv is not None:
                pay = Payment(
                    id=f"PAY-R{uuid.uuid4().hex[:6]}",
                    invoice_id=inv.id,
                    amount=min(rem.amount, inv.outstanding),
                    date=ctx.sim_date,
                    method="remittance",
                )
                ctx.erp.apply_payment(inv, pay)
                ctx.store.update_remittance(
                    rem.id, RemittanceStatus.MATCHED, inv.id, method
                )
                if inv.outstanding <= 0.01:  # fully settled -> no longer matchable
                    open_by_id.pop(inv.id, None)
                report.matched += 1
                report.by_method[method] += 1
                ctx.audit(
                    self.name, "matched_remittance", "remittance", rem.id,
                    detail=f"${rem.amount:,.2f} -> {inv.id} via {method}",
                    metadata={"method": method, "invoice": inv.id},
                )
            else:
                ctx.store.update_remittance(rem.id, RemittanceStatus.SUSPENSE)
                report.to_suspense += 1
                ctx.audit(
                    self.name, "remittance_suspense", "remittance", rem.id,
                    detail=f"${rem.amount:,.2f} from '{rem.payer_name}' unmatched",
                )
                needs_human = (
                    rem.amount >= SUSPENSE_ESCALATION_AMOUNT
                    and not ctx.store.has_open_escalation(rem.id)
                )
                if needs_human:
                    ctx.store.add_escalation(
                        Escalation(
                            id=f"ESC-{uuid.uuid4().hex[:8]}",
                            invoice_id=rem.id,  # remittance-level case
                            customer_id=rem.customer_id,
                            sim_date=ctx.sim_date.isoformat(),
                            outstanding=rem.amount,
                            risk_score=0.0,
                            priority=rem.amount,
                            severity="cash_app",
                            reason=(
                                f"unmatched remittance ${rem.amount:,.0f} "
                                f"from '{rem.payer_name}'"
                            ),
                            status=EscalationStatus.PENDING,
                        )
                    )
        return report
