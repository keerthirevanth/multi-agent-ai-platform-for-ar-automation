"""Credit-management agent — continuous exposure monitoring.

Each cycle it recomputes every customer's credit utilization (open exposure /
credit limit). Customers above the hold threshold are placed on **credit hold**
— no new credit sales until they pay down below the release threshold. This is
the standard order-to-cash guardrail: it stops the company from compounding its
exposure to a customer who already can't pay.

Deterministic by design (limits are policy, not judgment), fully audited.
"""

from __future__ import annotations

from ar_platform.agents.base import AgentContext
from ar_platform.models import CreditHold

HOLD_UTILIZATION = 0.90     # place on hold above this
RELEASE_UTILIZATION = 0.70  # release once back below this


class CreditAgent:
    name = "credit"

    def __init__(
        self,
        hold_at: float = HOLD_UTILIZATION,
        release_at: float = RELEASE_UTILIZATION,
    ):
        self.hold_at = hold_at
        self.release_at = release_at

    def run(self, ctx: AgentContext) -> tuple[int, int]:
        """Returns (new holds, releases)."""
        exposure: dict[str, float] = {}
        for inv in ctx.store.get_open_invoices():
            exposure[inv.customer_id] = exposure.get(inv.customer_id, 0.0) + inv.outstanding

        held_now = {h.customer_id for h in ctx.store.active_credit_holds()}
        new_holds = releases = 0

        for cust in ctx.store.get_customers():
            if cust.credit_limit <= 0:
                continue
            util = exposure.get(cust.id, 0.0) / cust.credit_limit

            if cust.id not in held_now and util > self.hold_at:
                ctx.store.add_credit_hold(
                    CreditHold(
                        customer_id=cust.id,
                        held_date=ctx.sim_date.isoformat(),
                        utilization_at_hold=round(util, 4),
                    )
                )
                new_holds += 1
                ctx.audit(
                    self.name, "credit_hold", "customer", cust.id,
                    detail=f"utilization {util:.0%} > {self.hold_at:.0%}; new orders blocked",
                    metadata={"utilization": round(util, 4)},
                )
            elif cust.id in held_now and util < self.release_at:
                ctx.store.release_credit_hold(cust.id, ctx.sim_date.isoformat())
                releases += 1
                ctx.audit(
                    self.name, "credit_release", "customer", cust.id,
                    detail=f"utilization {util:.0%} < {self.release_at:.0%}; orders resume",
                    metadata={"utilization": round(util, 4)},
                )
        return new_holds, releases
