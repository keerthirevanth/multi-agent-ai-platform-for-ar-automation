"""Layer 2 demo: run the Monitor -> Risk -> Comms pipeline once over the base case.

    PYTHONPATH=src python -m ar_platform.demo

This is a manual, single-pass demonstration. The full simulation clock and
orchestrated tick loop arrive in Layer 3; here we just show the agents
collaborating and the ML risk model scoring real invoices.
"""

from __future__ import annotations

from ar_platform.agents import CommsAgent, MonitorAgent, RiskAgent
from ar_platform.agents.base import AgentContext
from ar_platform.config import settings
from ar_platform.data.store import Store
from ar_platform.llm import get_llm
from ar_platform.tools.email import EmailTool
from ar_platform.tools.erp import ERPTool
from ar_platform.tools.ml_risk import train_risk_model


def main() -> None:
    with Store() as store:
        store.load_base_case()

        # Train the ML risk model on the current ledger.
        model = train_risk_model(
            store.get_customers(), store.get_invoices(), store.get_payments()
        )
        print(
            f"Risk model [{model.model_name}] trained on {model.train_size} resolved "
            f"invoices (problem rate {model.positive_rate:.0%}, in-sample AUC {model.auc:.3f})\n"
        )

        ctx = AgentContext(
            store=store,
            llm=get_llm(),
            email=EmailTool(),
            erp=ERPTool(store),
            risk_model=model,
            sim_date=settings.base_case_today,
        )

        # Pipeline: Monitor flags -> Risk scores -> Comms acts.
        monitor = MonitorAgent().run(ctx)
        print(f"[monitor] {monitor.summary}")

        risk = RiskAgent().run(ctx, monitor.items)
        print(f"[risk]    {risk.summary}")

        comms = CommsAgent().run(ctx, risk.items)
        print(f"[comms]   {comms.summary}")
        store.commit()

        print("\nTop 5 collection priorities (expected loss = exposure x risk):")
        print(f"  {'invoice':11s} {'customer':28s} {'over':>5s} {'out$':>10s} "
              f"{'risk':>5s} {'priority$':>11s}  action")
        for w in risk.items[:5]:
            print(
                f"  {w.invoice_id:11s} {w.customer_name[:27]:28s} "
                f"{w.days_overdue:4d}d {w.outstanding:10,.0f} "
                f"{w.risk_score:5.2f} {w.priority:11,.0f}  {w.action}"
            )


if __name__ == "__main__":
    main()
