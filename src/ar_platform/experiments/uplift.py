"""A/B uplift experiment: agents ON vs. agents OFF on the identical world.

The control arm (OFF) advances the same seeded simulation — invoices age,
customers pay at their baseline risk-driven rates, new invoices arrive — but no
agent ever acts, so no invoice receives the contacted-payment boost. The
treatment arm (ON) runs the full Monitor -> Risk -> Comms cycle with
human-in-the-loop escalations.

Both arms start from the same reproducible base case and use the same seed, so
the *only* systematic difference is the agents' work. Repeating over several
seeds turns anecdote into a measured uplift with spread.

    PYTHONPATH=src python -m ar_platform.experiments.uplift --seeds 3 --ticks 8

Writes ``reports/uplift_results.csv`` and prints a paired summary.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

from ar_platform.config import REPO_ROOT
from ar_platform.data.store import Store
from ar_platform.kpis import compute_kpis
from ar_platform.simulation import Simulation
from ar_platform.tools.email import EmailTool

REPORTS_DIR = REPO_ROOT / "reports"
UPLIFT_CSV = REPORTS_DIR / "uplift_results.csv"


@dataclass
class ArmResult:
    seed: int
    agents: str                 # "on" | "off"
    ticks: int
    days_per_tick: int
    start_open_ar: float
    end_open_ar: float
    end_overdue_ar: float
    end_dso: float
    collected: float
    emails_sent: int
    escalations: int


def run_arm(
    store: Store,
    seed: int,
    ticks: int,
    days_per_tick: int,
    agents_enabled: bool,
    outbox_dir: Path | None = None,
) -> ArmResult:
    """Run one arm from the pristine base case currently expected in ``store``.

    The caller is responsible for resetting ``store`` to the desired starting
    ledger before each arm (both arms must start from identical state).
    """
    outbox = outbox_dir or Path(tempfile.mkdtemp(prefix="ar_uplift_outbox_"))
    sim = Simulation(
        store,
        seed=seed,
        agents_enabled=agents_enabled,
        email_tool=EmailTool(outbox),
    )
    start_open_ar = round(
        sum(i.outstanding for i in store.get_open_invoices()), 2
    )

    reports = sim.run(ticks=ticks, days_per_tick=days_per_tick)

    kpis = compute_kpis(
        store.get_customers(), store.get_invoices(), as_of=sim.sim_date
    )
    return ArmResult(
        seed=seed,
        agents="on" if agents_enabled else "off",
        ticks=ticks,
        days_per_tick=days_per_tick,
        start_open_ar=start_open_ar,
        end_open_ar=reports[-1].open_ar,
        end_overdue_ar=kpis.overdue_ar,
        end_dso=kpis.dso,
        collected=round(sum(r.payments_amount for r in reports), 2),
        emails_sent=len(sim.email.sent),
        escalations=len(store.get_escalations()),
    )


def run_experiment(
    seeds: list[int],
    ticks: int = 8,
    days_per_tick: int = 7,
    store: Store | None = None,
    save: bool = True,
) -> list[ArmResult]:
    """Paired ON/OFF runs per seed, each from a fresh base case."""
    own_store = store is None
    store = store or Store()
    results: list[ArmResult] = []

    try:
        for seed in seeds:
            for agents_enabled in (False, True):
                store.load_base_case(force=True)
                results.append(
                    run_arm(
                        store, seed, ticks, days_per_tick, agents_enabled
                    )
                )
        # Leave the shared ledger back at the pristine base case.
        store.load_base_case(force=True)
    finally:
        if own_store:
            store.close()

    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        with open(UPLIFT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
    return results


def summarize(results: list[ArmResult]) -> dict:
    """Per-seed paired deltas (ON - OFF) and their aggregate."""
    by_seed: dict[int, dict[str, ArmResult]] = {}
    for r in results:
        by_seed.setdefault(r.seed, {})[r.agents] = r

    deltas = {"collected": [], "end_open_ar": [], "end_dso": [], "end_overdue_ar": []}
    for arms in by_seed.values():
        on, off = arms["on"], arms["off"]
        deltas["collected"].append(on.collected - off.collected)
        deltas["end_open_ar"].append(on.end_open_ar - off.end_open_ar)
        deltas["end_dso"].append(on.end_dso - off.end_dso)
        deltas["end_overdue_ar"].append(on.end_overdue_ar - off.end_overdue_ar)

    def agg(xs):
        return {
            "mean": round(mean(xs), 2),
            "std": round(stdev(xs), 2) if len(xs) > 1 else 0.0,
        }

    return {k: agg(v) for k, v in deltas.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agents-on/off uplift experiment.")
    parser.add_argument("--seeds", type=int, default=3, help="number of replication seeds")
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    seeds = [101 + i for i in range(args.seeds)]
    print(f"Running uplift experiment: seeds={seeds}, {args.ticks} ticks x {args.days} days\n")
    results = run_experiment(seeds, ticks=args.ticks, days_per_tick=args.days)

    header = (
        f"{'seed':>5s} {'arm':>4s} {'collected$':>13s} {'end openAR$':>13s} "
        f"{'end DSO':>8s} {'emails':>7s} {'esc':>5s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.seed:5d} {r.agents:>4s} {r.collected:13,.0f} {r.end_open_ar:13,.0f} "
            f"{r.end_dso:8.1f} {r.emails_sent:7d} {r.escalations:5d}"
        )

    s = summarize(results)
    print("\nPaired uplift (agents ON - OFF), mean +/- std across seeds:")
    print(f"  collected:   +${s['collected']['mean']:,.0f} +/- {s['collected']['std']:,.0f}")
    print(f"  end open AR: {s['end_open_ar']['mean']:+,.0f} +/- {s['end_open_ar']['std']:,.0f}")
    print(f"  end DSO:     {s['end_dso']['mean']:+.1f} +/- {s['end_dso']['std']:.1f} days")
    print(
        f"  end overdue: {s['end_overdue_ar']['mean']:+,.0f} "
        f"+/- {s['end_overdue_ar']['std']:,.0f}"
    )
    print(f"\nResults saved to {UPLIFT_CSV}")


if __name__ == "__main__":
    main()
