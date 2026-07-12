"""Run the AR simulation forward over several ticks and print a summary.

    PYTHONPATH=src python -m ar_platform.run_simulation --ticks 8 --days 7 --reset

Starts from the committed base case and advances the finance department week by
week, showing open AR trending down as the agents collect.
"""

from __future__ import annotations

import argparse

from ar_platform.config import settings
from ar_platform.data.store import Store
from ar_platform.simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AR automation simulation.")
    parser.add_argument("--ticks", type=int, default=8, help="number of ticks")
    parser.add_argument("--days", type=int, default=7, help="days per tick")
    parser.add_argument("--reset", action="store_true", help="reload the base case first")
    args = parser.parse_args()

    with Store() as store:
        store.load_base_case(force=args.reset)
        sim = Simulation(store)

        start_ar = round(sum(i.outstanding for i in store.get_open_invoices()), 2)
        print(f"Base case {settings.base_case_today} | open AR ${start_ar:,.0f}\n")
        print(f"  {'date':12s} {'newInv':>6s} {'pays':>5s} {'collected$':>12s} "
              f"{'openAR$':>13s} {'flag':>5s} {'email':>6s} {'reply':>6s} "
              f"{'nego':>5s} {'esc':>4s}")

        reports = sim.run(ticks=args.ticks, days_per_tick=args.days)
        for r in reports:
            c = r.cycle
            print(
                f"  {r.sim_date:12s} {r.new_invoices:6d} {r.payments_count:5d} "
                f"{r.payments_amount:12,.0f} {r.open_ar:13,.0f} "
                f"{c.flagged:5d} {c.emails_sent:6d} {c.replies_handled:6d} "
                f"{c.negotiated:5d} {c.escalations_new:4d}"
            )

        end_ar = reports[-1].open_ar
        total_collected = sum(r.payments_amount for r in reports)
        print(
            f"\nAfter {args.ticks} ticks: open AR ${start_ar:,.0f} -> ${end_ar:,.0f} "
            f"(collected ${total_collected:,.0f})"
        )
        pending = len(store.get_escalations())
        print(f"Escalations filed: {pending} | outbox emails: {len(sim.email.sent)}")


if __name__ == "__main__":
    main()
