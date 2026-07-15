# Enterprise AI Accounts Receivable (AR) Automation Platform

A **production-style**, multi-agent platform that simulates a corporate finance
department's accounts-receivable function. Specialized AI agents autonomously
monitor invoices, assess payment risk, communicate with customers, and surface
executive KPIs — demonstrating how an **agentic architecture** can automate
enterprise financial workflows, in a realistic *simulated* environment, far
beyond a traditional chatbot.

> **Status:** full AR function — collections, agentic conversation loop, cash
> application, disputes, credit management, cash forecasting — complete and
> tested within the simulated environment (60 unit/integration tests), backed by
> MySQL 8.

> **Scope & honest framing.** This is a self-contained *simulation*, not a
> deployed business system. The ledger is synthetic (no real ERP — chosen for
> privacy and reproducible benchmarking), and every measured result below is
> produced *inside* the simulation, against a deliberately unautomated control
> arm. The rigor (temporal ML evaluation, seeded A/B experiments, deterministic
> guardrails) is real; the magnitudes are only as valid as the simulation's
> modelling assumptions. Read every number as "under this simulation's
> assumptions," not as a claim about a live business.

Run the agent pipeline once, or drive the full week-by-week simulation:

```bash
PYTHONPATH=src python -m ar_platform.demo                              # single agent pass
PYTHONPATH=src python -m ar_platform.run_simulation --ticks 8 --days 7 --reset
```

Example: over 8 simulated weeks the agents drive open AR from ~$16.6M down to
~$5.5M, applying ~$12.9M of cash while negotiating, resolving disputes, and
filing escalations for human approval.

## Measured impact (A/B experiment, not a claim)

The platform's value is quantified by a controlled experiment: the identical
seeded world is run twice — agents ON vs. agents OFF (control: invoices still
age, customers still pay at their baseline rates, new invoices still arrive;
nobody chases) — paired across 3 replication seeds, 8 weeks each:

| Metric (8 weeks, mean ± std across seeds) | Agents ON vs. OFF |
| --- | --- |
| Additional cash applied to the ledger | **+$3.48M ± $1.16M** |
| Ending DSO | **−39.1 ± 14.5 days** |
| Ending open AR | −$4.26M ± $0.93M |
| Cash stuck unapplied (suspense) | **−$3.35M ± $0.02M** |
| Ending overdue AR | −$3.64M ± $1.12M |

Scope note: the control arm is a **deliberately pessimistic baseline** — it has
*no* automation beyond straight-through processing of clean-referenced
remittances (no dunning, no negotiation, no cash-application matching, so its
~$3.9M of messy remittances just sit unapplied). The uplift therefore measures
the full platform against a genuinely *unautomated* back office, which is the
honest comparison for "what does the automation add" but is generous by
construction — a real business already does *some* of this manually. Roughly
$3.35M of the gap is cash application alone; the rest is collections behavior.

The ON arm wins on every metric in every seed. Reproduce it yourself:

```bash
PYTHONPATH=src python -m ar_platform.experiments.uplift --seeds 3 --ticks 8
```

---

## Why this is different from a chatbot

| Chatbot | This platform |
| --- | --- |
| Responds to prompts | Runs autonomously on a simulation clock |
| Single model call | Multiple specialized agents that collaborate |
| No state | Persistent AR ledger + append-only audit log |
| No tools | Agents call tools (email, ERP updates, ML risk scoring) |
| No memory | Tracks promises across turns; broken promises re-escalate |
| One-way | Understands free-text replies and negotiates within policy bounds |
| No oversight | Human-in-the-loop escalation for high-risk cases |

## Architecture

```
Layer 4  Dashboard (Streamlit)   — DSO, aging, cash flow, collection efficiency
Layer 3  Orchestrator            — case routing, negotiation, promises, HITL, audit
Layer 2  Agents                  — Monitor · Risk · Comms · Inbox · Negotiator
Layer 1  Tools + Data            — ledger, email sim, ERP store, ML risk model
```

### The agentic layer — closing the conversation loop

A deterministic dunning engine can only *talk*. Real collections are a
*dialogue*, and this is where genuine agency is required (free-text input that
rules cannot enumerate). Each tick, customers reply to our dunning, and the
department reasons about and acts on those replies:

```
Customer reply (free text)
   -> Inbox agent      classifies intent + extracts terms  (LLM, or keyword fallback)
   -> case routing     dispute · already-paid · info · negotiation
   -> Negotiator agent proposes terms, POLICY validates, applies or escalates
   -> Promise tracking  kept promises clear; broken promises re-escalate
```

The centerpiece is the **Negotiator** and its guardrail. When a customer asks
for an extension or plan, an LLM (or the deterministic policy) *proposes* terms,
but the deterministic [`NegotiationPolicy`](src/ar_platform/dialogue.py)
**validates and bounds** every proposal against risk-tiered authority:

| Customer risk | Max self-service extension | Beyond that / large exposure / repeat broken promise |
| --- | --- | --- |
| Low | 30 days | → **human escalation** |
| Medium | 14 days | → **human escalation** |
| High | 7 days | → **human escalation** |

So an LLM can never grant beyond authority — a 90-day request to a high-risk
account is clamped to 7 days or escalated. **LLM proposes, deterministic policy
disposes, humans handle the edges, everything is audited.** That is what makes a
non-deterministic reasoner safe inside a finance system.

Behavior the conversation changes: disputed invoices are put on hold (dunning
stops), invoices under an agreed promise aren't chased, and broken promises
re-escalate automatically. Runs deterministically for CI (keyword classifier +
policy); `AR_LLM_MODE=claude` swaps in genuine language understanding and
reasoning.

### Full AR function (Phase B): cash application, disputes, credit, forecast

Beyond collections, the platform now covers the rest of the order-to-cash
back office the way real AR products do:

- **Cash application** — customers no longer "pay invoices" directly; they send
  **bank remittances** with noisy references (~70% carry a clean invoice id and
  are straight-through processed; the rest arrive as "payment on account" from
  "RODRIGUEZ FIGUEROA AND SANCHEZ"). The cash-app agent matches them in three
  deterministic passes (invoice reference → fuzzy payer name + exact amount →
  single open invoice), parks failures in **suspense**, and escalates large
  unmatched amounts to a human. Match accuracy is measured against ground truth
  the matcher never sees. Until cash is applied, the invoice wrongly looks
  unpaid — automating this is real money (the control arm's "unapplied cash"
  shows the cost of not having it).
- **Dispute management** — dispute replies open a tracked case with a
  categorized reason (delivery / billing error / quality). Dunning holds while
  the case is open; a human verdict either upholds the claim (**credit memo**:
  full credit for billing errors, partial for delivery/quality, remainder back
  to collection) or denies it (customer gets an explanation email and dunning
  resumes).
- **Credit management** — every cycle the credit agent recomputes each
  customer's utilization (open exposure / credit limit): above 90% → **credit
  hold** (the world stops issuing them new invoices), below 70% → released.
- **Cash forecast** — a payment-*date* regressor (same leakage-free features,
  temporal-holdout MAE reported) predicts when each open invoice will pay;
  binned weekly and weighted by collection probability it yields the
  **8-week expected-cash curve** a treasurer plans against.

### Deterministic core, honestly labeled

The operational pipeline is **deliberately deterministic**: severity thresholds,
escalation dollar limits, dunning wording (fixed compliance-style templates in
[tools/templates.py](src/ar_platform/tools/templates.py)), and ML risk scoring.
That is how real AR platforms work — finance demands decisions that are exact,
auditable, and identical every run. Nothing deterministic here masquerades as
AI.

The **LLM seam** (`AR_LLM_MODE=claude`) is reserved for what rules genuinely
cannot do: today, personalized drafting; next (in progress), the *agentic
layer* — understanding free-text customer replies, negotiating payment plans
within business-rule bounds, and routing disputes. The design principle
throughout: **the LLM proposes, deterministic guardrails validate, humans
handle the edges, and everything is audited.**

The pipeline is both reactive and **proactive**: overdue invoices are chased
with severity-escalating dunning (reminder → overdue → urgent → final, with
high-stakes cases routed to human approval), and open invoices falling due
within 7 days are flagged `pre_due` — if the risk model scores one above
threshold, the customer gets a single gentle upcoming-payment reminder *before*
going late (never escalated, never repeated). Mature AR teams move DSO before
the due date, not after.

## Quickstart

**Fastest path — Docker (brings its own MySQL):**

```bash
docker compose up        # starts MySQL 8 + the dashboard on http://localhost:8501
```

**Local path:**

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Set up the MySQL databases + app user (one time) — see docs/DATABASE.md
#    Creates `ar_platform` and `ar_platform_test` and the `ar_user` account.

# 3. Load the committed base case into MySQL
PYTHONPATH=src python -m ar_platform.data.load_base_case --force

# 4. Run tests / the dashboard
PYTHONPATH=src pytest -q
PYTHONPATH=src streamlit run dashboard/app.py
```

Data is stored in **MySQL 8** via SQLAlchemy. See [docs/DATABASE.md](docs/DATABASE.md)
for setup. The base case itself is committed as CSVs and regenerated with
`python -m ar_platform.data.generator`.

## The base case & how it updates

- **Base case** — a seeded synthetic ledger (150 customers, 4,500 invoices
  spanning ~15 months of history, anchored to 2026-01-15) committed as CSVs in
  [`data/seed/`](data/seed/). Because it is seeded (`AR_SEED=42`), every clone
  regenerates the *identical* ledger. The ~30-invoice-deep history per customer
  is deliberate: the ML diagnostics showed history depth is what makes payment
  behavior learnable.
- **Updates** — the simulation advances on a clock ("ticks"). Each tick ages
  invoices, applies customer payments, injects new invoices, and runs the agent
  cycle. All state changes persist to MySQL and append to the audit log.
  *(Simulation engine arrives with Layer 3.)*

### Base-case profile (seed=42)

| Metric | Value |
| --- | --- |
| Customers | 150 (18 enterprise · 41 midmarket · 91 SMB) |
| Invoices | 4,500 over ~15 months (3,653 paid · 412 open · 349 overdue · 69 partial · 17 disputed) |
| Total open AR | ~$16.6M |
| Risk mix | 80 low · 48 medium · 22 high |
| History depth | ~30 invoices per customer |

## Machine learning: payment-risk model

The Risk agent scores every overdue invoice with the probability that it
becomes a collection problem. The model is selected by a rigorous benchmark —
no assumptions:

```bash
PYTHONPATH=src python -m ar_platform.ml.benchmark          # full run
PYTHONPATH=src python -m ar_platform.ml.benchmark --quick  # fast smoke run
```

**Methodology**

- **Leakage-free behavioral features** ([ml/features.py](src/ar_platform/ml/features.py)):
  every feature is computed *as of the invoice's issue date* from prior
  transaction history only (prior lateness mean/std/max plus an
  empirical-Bayes *shrunk* lateness estimate for thin-history customers,
  on-time ratio, open exposure, credit utilization, days since last payment,
  invoice size vs. the customer's average, cold-start flag, segment). The generator's
  `historical_delay_avg` oracle column is deliberately **excluded** — the model
  must rediscover risk from observable behavior, as in production.
- **Temporal split** — first 70% of invoices by issue date train, last 30%
  test. Random splits leak the future; this one can't.
- **Tuning on train only** — `RandomizedSearchCV` (25 candidates/model) with
  `TimeSeriesSplit` CV inside the training window; the test window is touched
  once per model.
- **Winner chosen by CV score**, so test metrics stay unbiased. Calibration
  (isotonic) is fitted and kept only if it improves the Brier score —
  probabilities feed `expected_loss = amount × p` KPIs, so calibration matters.

- **Maturity-window labels** — an invoice is labeled a *problem* only if it was
  not fully settled within 30 days after its due date, and it enters training
  only once that window has elapsed. A naive "overdue right now" label makes
  recently issued invoices look artificially risky (they simply haven't had
  time to pay) and causes train/test drift under temporal splits; the maturity
  window eliminates it (positive rate is now 22% train / 24% test).

**Results** (base case, seed 42 — regenerate with the command above after
`python -m ar_platform.data.load_base_case --force`):

| Model | CV AUC | Test AUC | PR-AUC | Brier | Top-20% capture |
| --- | --- | --- | --- | --- | --- |
| **Logistic regression (winner by CV)** | **0.789** | 0.791 | 0.672 | 0.130 | 0.638 |
| Hist. gradient boosting | 0.786 | 0.796 | 0.643 | 0.128 | 0.628 |
| XGBoost | 0.779 | 0.794 | 0.664 | 0.129 | 0.583 |
| Random forest | 0.773 | 0.796 | 0.643 | 0.163 | 0.606 |
| LightGBM | 0.760 | 0.801 | 0.663 | 0.129 | 0.564 |
| Decision tree | 0.752 | 0.791 | 0.555 | 0.180 | 0.574 |
| Baseline: prior-lateness rule | – | 0.800 | 0.654 | – | 0.616 |
| Baseline: majority class | 0.500 | 0.500 | 0.235 | 0.180 | 0.265 |

The winner's tuned configuration is stored in `models/risk_model_meta.json`;
the runtime scorer ([tools/ml_risk.py](src/ar_platform/tools/ml_risk.py))
re-fits that configuration on the live ledger, so the platform always deploys
the benchmark-validated model family.

**How we got here — diagnostics, not guesswork.** On the original shallow
ledger (~7 invoices of history per customer) all models plateaued near AUC
0.65–0.69. An oracle-ceiling test (adding the generator's hidden true-risk
fields) showed a ceiling of ~0.76, a flat learning curve showed more *rows*
would not help, and a data-scaling experiment showed deeper *per-customer
history* was the lever (+0.16 AUC). The base case was rebuilt accordingly
(450-day window, ~30 invoices/customer) and labels were re-posed with the
maturity window — lifting held-out AUC to ~0.79–0.80 and cutting Brier from
0.24 to 0.13 (much better-calibrated probabilities for the expected-loss KPIs).

**Honest findings the benchmark surfaces:**

- With deep history, the *single-feature rule* (rank by prior mean lateness)
  reaches 0.80 test AUC by itself — payment behavior is highly persistent, so
  simple signals dominate. The models' added value is calibrated probabilities
  (Brier 0.13 vs. none for a rank-only rule), multi-feature robustness for
  thin-history customers, and better dollar-capture at the top of the worklist.
- The winning model family changed with the data regime (gradient boosting on
  shallow noisy data → logistic regression on deep clean data) — a concrete
  demonstration of why model selection must be re-run when the data changes,
  which is exactly what the benchmark automates.
- On synthetic data a benchmark partly measures the generator's rules;
  validating the same pipeline on a public receivables dataset is on the
  roadmap.

## Project layout

```
src/ar_platform/
├── config.py          # env-driven settings (seed, LLM mode, base-case date)
├── models.py          # domain models: Customer, Invoice, Payment, AuditEntry
├── data/
│   ├── generator.py   # Faker-based synthetic ledger generator (seeded)
│   └── store.py       # MySQL repository (SQLAlchemy); loads base case, evolves state
├── ml/
│   ├── features.py    # leakage-free behavioral features (as-of-date)
│   ├── benchmark.py   # temporal-split model benchmark + tuning + calibration
│   └── forecast.py    # payment-date regression -> weekly expected-cash curve
├── agents/            # Monitor, Risk, Comms, Inbox, Negotiator, CashApp, Dispute, Credit
├── dialogue.py        # reply classifier + NegotiationPolicy (deterministic guardrail)
├── tools/             # email sim, ERP writer, runtime risk scorer, templates
├── experiments/
│   └── uplift.py      # A/B experiment: agents on vs. off, paired by seed
├── data/reply_sim.py  # simulated free-text customer replies (the "customer")
├── orchestrator.py    # case routing + negotiation + promises + HITL escalation
└── simulation.py      # the tick engine (world update + agent cycle)
data/seed/             # committed, reproducible base-case CSVs
reports/               # benchmark results (committed)
models/                # trained model + metadata (binary gitignored)
tests/                 # pytest suite
```

## Performance & known limitations

- **Throughput.** One simulation tick over the full ~4,500-invoice ledger takes
  ~47s (all eight agents, cash-application, credit and forecast steps). The
  bottleneck is deliberate: every write opens its own MySQL transaction for
  clarity and auditability rather than batching. Fine for offline experiments;
  the obvious optimization (per-tick transaction batching) is unimplemented.
- **Synthetic data.** The ledger is generated, not sampled from a real ERP, so a
  benchmark partly measures the generator's own rules. Validating the pipeline
  on a public receivables dataset is the highest-value next step for external
  credibility.
- **Simulated dynamics.** Payment-response magnitudes (contact boost, promise
  boost, reply/dispute rates) are modelling assumptions, not fitted to real
  behavior — see the scope note at the top.

## Roadmap

- [x] **Layer 1** — data models, synthetic generator, MySQL store, base case
- [x] **Layer 2** — agents (Monitor, Risk w/ ML, Comms), pluggable LLM, tools
- [x] **Layer 3** — orchestrator, simulation clock, audit log, HITL escalation
- [x] **Layer 4** — Streamlit executive dashboard (DSO, aging, cash flow)
- [x] **ML rigor** — behavioral features, temporal benchmark, tuning, calibration
- [x] **A/B uplift experiment** — measured: −10.2 days DSO, +$971K collected vs. control
- [x] **Pre-due proactive outreach** — risk-gated upcoming-payment nudges before invoices go late
- [x] **Agentic layer** — simulated customer replies, inbox intent classification,
      negotiation agent (proposes within business-rule bounds), promise-to-pay
      tracking, case-based orchestration
- [x] **Cash application** — noisy remittance matching, suspense queue, accuracy
      measured against hidden ground truth
- [x] **Dispute workflow** — categorized cases, dunning hold, credit-memo
      resolution via the human queue
- [x] **Credit management** — utilization monitoring, automatic holds/releases
- [x] **Cash forecast** — payment-date regression → weekly expected-cash curve
- [ ] FastAPI service layer, config-driven rules, auth/roles (production hardening)
- [ ] SHAP explanations attached to escalations

## License

MIT — see [LICENSE](LICENSE).
