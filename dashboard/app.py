"""Executive AR dashboard (Streamlit).

Run:

    PYTHONPATH=src streamlit run dashboard/app.py

Provides live KPIs (DSO, aging, cash flow, collection efficiency), a control to
advance the simulation clock, trend charts, the human-in-the-loop escalation
approval queue, and an audit-log viewer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ package importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from ar_platform.config import settings  # noqa: E402
from ar_platform.data.store import Store  # noqa: E402
from ar_platform.kpis import compute_kpis  # noqa: E402
from ar_platform.models import (  # noqa: E402
    AGING_BUCKETS,
    DisputeStatus,
    EscalationStatus,
    PromiseStatus,
    RemittanceStatus,
)
from ar_platform.simulation import Simulation  # noqa: E402

st.set_page_config(page_title="AR Automation Platform", layout="wide")


# --- state ------------------------------------------------------------------
def _init(reset: bool = False):
    store = Store()
    store.load_base_case(force=reset)
    sim = Simulation(store)
    st.session_state.sim = sim
    st.session_state.history = []
    _snapshot()  # record the starting KPIs


def _snapshot():
    sim: Simulation = st.session_state.sim
    kpis = compute_kpis(
        sim.store.get_customers(),
        sim.store.get_invoices(),
        as_of=sim.sim_date,
        risk_model=sim.model,
    )
    st.session_state.history.append(kpis)
    return kpis


if "sim" not in st.session_state or not hasattr(st.session_state.sim.store, "recent_audit"):
    # Always open at the known base case so the demo starts from a clean ledger.
    _init(reset=True)

sim: Simulation = st.session_state.sim
kpis = st.session_state.history[-1]

# --- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    backend = sim.llm.name if sim.llm else "deterministic templates"
    st.caption(f"Drafting backend: **{backend}**  ·  seed {settings.seed}")
    st.metric("Simulation date", sim.sim_date.isoformat())

    days = st.select_slider("Days per tick", options=[1, 7, 14, 30], value=7)
    if st.button("Advance clock", use_container_width=True, type="primary"):
        sim.tick(days=days)
        _snapshot()
        st.rerun()

    n_weeks = st.number_input("Fast-forward weeks", 1, 26, 4)
    if st.button("Fast-forward", use_container_width=True):
        for _ in range(int(n_weeks)):
            sim.tick(days=7)
        _snapshot()
        st.rerun()

    if st.button("Reset to base case", use_container_width=True):
        _init(reset=True)
        st.rerun()

# --- header + KPI cards -----------------------------------------------------
st.title("Enterprise AR Automation Platform")
st.caption(
    "Multi-agent finance department — Monitor · Risk (ML) · Comms · Inbox · "
    "Negotiator · Orchestrator with human-in-the-loop escalation."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open AR", f"${kpis.open_ar:,.0f}")
c2.metric("DSO (days)", f"{kpis.dso:.1f}")
c3.metric("Overdue AR", f"${kpis.overdue_ar:,.0f}", f"{kpis.overdue_pct:.0%} of AR")
c4.metric("Collection rate", f"{kpis.collection_rate:.1%}")
c5.metric("Expected loss", f"${kpis.expected_loss:,.0f}")

st.divider()

# --- charts -----------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Aging analysis")
    aging_df = pd.DataFrame(
        {
            "bucket": AGING_BUCKETS,
            "outstanding": [kpis.aging[b] for b in AGING_BUCKETS],
            "count": [kpis.aging_counts[b] for b in AGING_BUCKETS],
        }
    )
    fig = px.bar(
        aging_df, x="bucket", y="outstanding", text_auto=".2s",
        color="bucket", color_discrete_sequence=px.colors.sequential.Reds,
    )
    fig.update_layout(showlegend=False, height=330, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Open AR & DSO over time")
    if len(st.session_state.history) > 1:
        trend = pd.DataFrame(
            {
                "date": [k.as_of for k in st.session_state.history],
                "Open AR": [k.open_ar for k in st.session_state.history],
                "DSO": [k.dso for k in st.session_state.history],
            }
        )
        fig2 = px.line(trend, x="date", y="Open AR", markers=True)
        fig2.update_layout(height=330, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Advance the clock to build the trend.")

# --- expected cash flow -----------------------------------------------------
st.subheader("Risk-adjusted cash-flow expectation")
cf1, cf2 = st.columns(2)
cf1.metric("Expected collectible", f"${kpis.expected_collectible:,.0f}")
cf2.metric("At-risk (expected loss)", f"${kpis.expected_loss:,.0f}")

st.divider()

# --- escalation approval queue (HITL) ---------------------------------------
st.subheader("Escalation queue — human approval")
pending = sim.store.get_escalations(EscalationStatus.PENDING)
if not pending:
    st.success("No escalations awaiting approval.")
else:
    st.caption(f"{len(pending)} case(s) awaiting a decision (highest exposure first).")
    for esc in sorted(pending, key=lambda e: e.priority, reverse=True)[:15]:
        cust = sim.store.get_customer(esc.customer_id)
        cols = st.columns([3, 2, 2, 1.2, 1.2])
        cols[0].write(f"**{esc.invoice_id}** — {cust.name if cust else esc.customer_id}")
        cols[1].write(f"${esc.outstanding:,.0f} out")
        cols[2].write(f"risk {esc.risk_score:.0%} · pri ${esc.priority:,.0f}")
        if cols[3].button("Approve", key=f"a_{esc.id}", help="Approve: send final notice"):
            sim.resolve(esc, approve=True, note="dashboard approval")
            st.rerun()
        if cols[4].button("Reject", key=f"r_{esc.id}", help="Reject: place on hold"):
            sim.resolve(esc, approve=False, note="dashboard hold")
            st.rerun()

st.divider()

# --- agentic layer: customer conversation -----------------------------------
st.subheader("Customer conversation (agentic layer)")
replies = sim.store.get_replies()
promises = sim.store.get_promises()
kept = sum(1 for p in promises if p.status == PromiseStatus.KEPT)
broken = sum(1 for p in promises if p.status == PromiseStatus.BROKEN)
pending_pr = sum(1 for p in promises if p.status == PromiseStatus.PENDING)

a1, a2, a3, a4 = st.columns(4)
a1.metric("Replies handled", sum(1 for r in replies if r.status == "handled"))
a2.metric("Active promises", pending_pr)
a3.metric("Promises kept", kept)
a4.metric("Promises broken", broken)

if replies:
    with st.expander(f"Recent customer replies ({len(replies)})"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "date": r.sim_date, "invoice": r.invoice_id,
                        "intent": r.intent, "action": r.handled_action,
                        "message": r.text,
                    }
                    for r in replies[:40]
                ]
            ),
            use_container_width=True, hide_index=True,
        )
else:
    st.info("Advance the clock — once dunning emails go out, customers reply.")

st.divider()

# --- Phase B: cash application · disputes · credit --------------------------
st.subheader("Cash application, disputes & credit")
rems = sim.store.get_remittances()
n_matched = sum(1 for r in rems if r.status == RemittanceStatus.MATCHED)
n_suspense = sum(1 for r in rems if r.status == RemittanceStatus.SUSPENSE)
disputes_all = sim.store.get_disputes()
open_disputes = [d for d in disputes_all if d.status == DisputeStatus.OPEN]
holds = sim.store.active_credit_holds()

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Remittances matched", n_matched)
b2.metric("In suspense", n_suspense)
b3.metric("Unapplied cash", f"${sim.store.unapplied_cash():,.0f}")
b4.metric("Disputes (open/total)", f"{len(open_disputes)}/{len(disputes_all)}")
b5.metric("Credit holds active", len(holds))

if holds:
    with st.expander(f"Customers on credit hold ({len(holds)})"):
        hold_rows = []
        for h in holds:
            c = sim.store.get_customer(h.customer_id)
            hold_rows.append(
                {
                    "customer": c.name if c else h.customer_id,
                    "held since": h.held_date,
                    "utilization at hold": f"{h.utilization_at_hold:.0%}",
                }
            )
        st.dataframe(pd.DataFrame(hold_rows), use_container_width=True, hide_index=True)

# --- cash forecast -----------------------------------------------------------
st.subheader("Cash forecast (payment-date model)")
if st.button("Compute 8-week expected-cash curve"):
    from ar_platform.ml.forecast import expected_cash_curve, train_payment_forecaster

    with st.spinner("Training payment-date model..."):
        fc = train_payment_forecaster(
            sim.store.get_customers(), sim.store.get_invoices(),
            sim.store.get_payments(), as_of=sim.sim_date,
        )
        curve = expected_cash_curve(
            fc, sim.store.get_customers(), sim.store.get_invoices(),
            as_of=sim.sim_date, weeks=8, risk_model=sim.model,
        )
    st.session_state.cash_curve = (curve, fc.mae_holdout)

if "cash_curve" in st.session_state:
    curve, mae = st.session_state.cash_curve
    st.caption(f"Payment-date model holdout MAE: {mae} days · risk-weighted amounts")
    curve_df = pd.DataFrame(curve)
    fig3 = px.bar(curve_df, x="week_start", y="expected_cash", text_auto=".2s")
    fig3.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- audit log --------------------------------------------------------------
with st.expander("Audit log (most recent 50 actions)"):
    rows = sim.store.recent_audit(limit=50)
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No actions logged yet.")

# --- outbox -----------------------------------------------------------------
with st.expander(f"Collection outbox ({len(sim.email.sent)} emails)"):
    for em in sim.email.sent[-8:][::-1]:
        st.markdown(f"**{em.subject}** to {em.to}  \n_{em.body[:160]}..._")
