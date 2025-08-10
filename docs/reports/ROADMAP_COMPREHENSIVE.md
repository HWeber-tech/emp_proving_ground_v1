# EMP Roadmap (FIX‑First, Budget‑Tiered)

A comprehensive, practical plan to move from today’s consolidated FIX‑first baseline to the full system vision outlined in the Encyclopaedia. Broker connectivity remains FIX‑only throughout; low‑budget tiers use OpenBB and other low‑cost data sources for research and backtesting only.

## Guiding principles

- FIX‑only for broker connectivity; no OpenAPI paths in runtime or CI.
- Calculations live in the sensory layer; expand technical analysis inside the sensory cortex components.
- Single source of truth for strategy, risk, and evolution surfaces under `src/core/*`.
- Safety first: deterministic offline workflows precede any live activity.
- Every milestone has measurable exit criteria and rollback paths.

## Where we are now (Baseline)

- Clean, FIX‑only codebase; legacy modules removed or shimmed
- Core strategy/risk/evolution consolidated and test‑covered
- Backtest/report pipeline writes artifacts to `docs/reports/`
- Paper/demo FIX path is the default; mock (`EMP_USE_MOCK_FIX=1`) is reserved for CI/unit tests

Artifacts: `docs/reports/CLEANUP_REPORT.md`, README quick start, tests under `tests/current/` all green.

## Budget tiers (platform scope and costs grow with tiers)

Funding model: lower tiers fund the next tier’s build‑out. Each tier must produce measurable value (reports, research outputs, or paper performance) that justifies moving up. Promotion requires meeting KPIs and securing budget from demonstrated outcomes.

- **Tier Bronze (Low budget, OpenBB/Yahoo data; no paid infra)**
  - Goal: Prove signal pipeline, risk controls, and execution model offline
  - Data: OpenBB (macro/yields/equities), Yahoo as fallback; local parquet/duckdb storage
  - Infra: Local dev + optional free cloud storage; no managed services
  - Broker: FIX paper/demo sessions (default); mock only for CI/unit tests
  - Funding output: publish backtest reports and feature studies; baseline paper performance to unlock Silver
- **Tier Silver (Moderate budget)**
  - Goal: Real‑time research pipeline, stateful backtests, and FIX price/trade dry‑runs
  - Data: Same as Bronze + optional premium endpoints where needed
  - Infra: Redis + TimescaleDB (or Postgres) for real‑time/state; basic dashboards
  - Broker: FIX sessions established but restricted to paper/demo
  - Funding output: real‑time dashboards and reproducible stateful backtests; reliability KPIs to unlock Gold
- **Tier Gold (High budget)**
  - Goal: Robustization (HA, observability, secrets, audit), multi‑asset scaling, nightly research
  - Data: Premium feeds as needed, scheduled ingest; model registry
  - Infra: Managed DBs, metrics/alerts, job scheduler, artifacts storage
  - Broker: Gradual live ramp with strict guardrails
  - Funding output: nightly research pipeline, audited experiments, improved paper metrics; SLOs to justify live pilot (Platinum)
- **Tier Platinum (Very high budget / production)**
  - Goal: Production operations with SLOs, DR, compliance, and on‑call
  - Data: Full vendor contracts and redundancy
  - Infra: Multi‑region, infra as code, compliance pipelines
  - Broker: Live with controlled exposure and progressive rollout

## Phased roadmap (outcomes and exit criteria)

### Phase 1 — Bronze: Offline excellence (4–6 weeks)
**Objectives**
- Establish reproducible, low‑cost research loop with OpenBB/Yahoo data
- Validate sensory features (yields/macro/microstructure), risk gates, and sizing offline

**Scope & deliverables**
- OpenBB ingestion adapter (data only), local parquet/duckdb stores
- Sensory cortex extensions for TA features (yields/macro/vol) within sensory layer
- Backtest runner and reports (regime gating, braking, portfolio caps, USD beta)
- CI: tests + lint + FIX‑only guard; artifact publishing of reports

**Exit criteria / KPIs**
- End‑to‑end backtest on sample symbols with report artifacts (PnL/maxDD/regime counts)
- ≥3 WHY/WHAT feature families integrated and unit‑tested
- CI green on merge with generated reports uploaded

### Phase 2 — Silver: Real‑time research loop (6–8 weeks)
**Objectives**
- Introduce stateful, real‑time pipeline with minimal managed services
- Dry‑run FIX price/trade sessions and end‑to‑end monitoring

**Scope & deliverables**
- Redis + TimescaleDB (or Postgres + time series ext) for ticks, features, and states
- Real‑time ingestion (OpenBB scheduling + Yahoo fallback), backfill and replay tools
- Portfolio tracker, metrics endpoints, dashboards for core signals
- FIX sessions: quote/trade connectivity exercised against demo; no live orders

**Exit criteria / KPIs**
- Stable real‑time signal computation < 1s latency at 1–5Hz
- Stateful backtest reproducibility from DB snapshots
- FIX sessions connect reliably; order flow dry‑run completes without errors

### Phase 3 — Gold: Robustization & scale (8–12 weeks)
**Objectives**
- Harden the platform for scale, reliability, and multi‑asset workflows

**Scope & deliverables**
- Observability (metrics, logs, alerts), secrets management, audit logs
- Job scheduler for nightly research runs and rolling backtests
- Parameter registry and experiment tracking; feature store consolidation
- Risk: cross‑asset caps, intraday drawdown controls, exposure scheduler

**Exit criteria / KPIs**
- SLOs met for data freshness and processing latency
- Nightly backtests complete < N hours with artifacts and audit trail
- Recovery playbooks tested (DB restore, process failover)

### Phase 4 — Platinum: Production operations (ongoing)
**Objectives**
- Productionize: compliance, DR, progressive live ramp with strict guardrails

**Scope & deliverables**
- DR/BCP: backups, region failover tests, infra as code
- Compliance hooks: audit trail export, change management, access controls
- Live trading ramp: 0 → small exposure with automatic kill‑switch and circuit breakers

**Exit criteria / KPIs**
- Successful dark‑run, then controlled live ramp with target error budgets
- Post‑incident RCA template exercised; on‑call runbooks in place

## Workstreams and owners (suggested)

- Sensory & features: TA/ML feature design, sensory cortex integrations
- Risk & sizing: per‑asset caps, aggregate VaR/β caps, braking/regime gates
- Data & storage: ingestion adapters, stateful DBs, replayers, snapshot tools
- Execution & FIX: FIX manager, order lifecycle, paper/demo connectivity
- Observability & ops: metrics/logs/alerts, dashboards, playbooks
- Governance & CI: policy enforcement, lint/tests, artifact publishing

## Compliance & policy checklist

- FIX‑only broker connectivity
- CI job blocks OpenAPI/FastAPI deps/references
- Secrets in vault, never in code or logs
- Reports and artifacts stored under `docs/reports/` and tagged by run

## Risk register (top)

- Data vendor constraints (OpenBB rate limits / coverage) → scheduling + caching
- Hidden complexity in live FIX integration → exercise against demo early
- Cost creep in Silver/Gold tiers → strict infra budget caps and teardown scripts

## References & appendices

- Encyclopaedia: `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md`
- Cleanup status: `docs/reports/CLEANUP_REPORT.md`
- FIX guides and architecture notes: `docs/fix_api/*`, `docs/ARCHITECTURE_REALITY.md`
