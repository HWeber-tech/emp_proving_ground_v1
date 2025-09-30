# Modernisation roadmap – Season reset

This reset distils the latest audit, technical debt, and status reports into a
fresh execution plan. It assumes the conceptual architecture mirrors the EMP
Encyclopedia while acknowledging that most subsystems remain scaffolding.

## Current parity snapshot

| Signal | Reality | Evidence |
| --- | --- | --- |
| Architecture | Layered domains and canonical `SystemConfig` definitions are in place, enforcing the core → sensory → thinking → trading → orchestration stack described in the encyclopedia. | 【F:docs/architecture/overview.md†L9-L48】 |
| Delivery state | The codebase is still a development framework: evolution, intelligence, execution, and strategy layers run on mocks; there is no production ingest, risk sizing, or portfolio management. | 【F:docs/DEVELOPMENT_STATUS.md†L7-L35】 |
| Quality posture | CI passes with 76% coverage, but hotspots include operational metrics, position models, and configuration loaders; runtime validation checks still fail. | 【F:docs/ci_baseline_report.md†L8-L27】【F:docs/technical_debt_assessment.md†L31-L112】 |
| Debt hotspots | Hollow risk management, unsupervised async tasks, namespace drift, and deprecated exports continue to surface in audits. | 【F:docs/technical_debt_assessment.md†L33-L80】【F:src/core/__init__.py†L11-L51】 |
| Legacy footprint | Deprecated configs and legacy integration guides remain in the tree, signalling unfinished cleanup. | 【F:src/config/risk_config.py†L1-L8】【F:src/config/evolution_config.py†L1-L8】【F:docs/legacy/README.md†L1-L12】 |

## Gaps to close

- [ ] **Operational data backbone** – Deliver real Timescale/Redis/Kafka services,
  parameterise SQL, and supervise ingest tasks instead of relying on mocks.
- [ ] **Sensory + evolution execution** – Replace HOW/ANOMALY stubs, wire lineage
  telemetry, and prove adaptive strategies against recorded data.
- [ ] **Risk and runtime safety** – Enforce `RiskConfig`, finish the builder rollout,
  adopt supervised async lifecycles, and purge deprecated facades.
- [ ] **Quality and observability** – Expand regression coverage, close the
  documentation gap, and track remediation progress through CI snapshots.
- [ ] **Dead code and duplication** – Triage the 168-file dead-code backlog and
  eliminate shim exports so operators see a single canonical API surface.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】

## Roadmap cadence

### Now (0–30 days)

- [x] **Stabilise runtime entrypoints** – Move all application starts through
  `RuntimeApplication` and register background jobs under a task supervisor to
  eliminate unsupervised `create_task` usage. Runtime CLI invocations and the
  bootstrap sensory loop now run under `TaskSupervisor`, ensuring graceful
  signal/time-based shutdown paths.【F:docs/technical_debt_assessment.md†L33-L56】【F:src/runtime/cli.py†L206-L249】【F:src/runtime/bootstrap_runtime.py†L227-L268】
- [ ] **Security hardening sprint** – Execute the remediation plan’s Phase 0:
  parameterise SQL, remove `eval`, and address blanket exception handlers in
  operational modules.【F:docs/development/remediation_plan.md†L34-L72】
    - *Progress*: Hardened the SQLite-backed real portfolio monitor with managed
      connections, parameterised statements, and narrowed exception handling to
      surface operational failures instead of masking them.【F:src/trading/portfolio/real_portfolio_monitor.py†L1-L572】
- [x] **Context pack refresh** – Replace legacy briefs with the updated context in
  `docs/context/alignment_briefs` so discovery and reviews inherit the same
  narrative reset (this change set).
- [ ] **Coverage guardrails** – Extend the CI baseline to include ingest orchestration
  and risk policy regression tests, lifting coverage beyond the fragile 76% line.
  - *Progress*: Added an end-to-end regression for the real portfolio monitor to
    exercise data writes, analytics, and reporting flows under pytest, closing a
    previously untested gap in the trading surface.【F:tests/trading/test_real_portfolio_monitor.py†L1-L77】

### Next (30–90 days)

- [ ] **Institutional ingest vertical** – Provision managed Timescale/Redis/Kafka
  environments, implement supervised connectors, and document failover drills.
- [ ] **Sensory cortex uplift** – Deliver executable HOW/ANOMALY organs, instrument
  drift telemetry, and expose metrics through runtime summaries and the event
  bus.
- [ ] **Evolution engine foundation** – Seed realistic genomes, wire lineage
  snapshots, and gate adaptive runs behind feature flags until governance reviews
  complete.【F:docs/development/remediation_plan.md†L92-L167】
- [ ] **Risk API enforcement** – Align trading modules with deterministic risk
  interfaces, surface policy violations via telemetry, and add escalation runbooks.

### Later (90+ days)

- [ ] **Operational readiness** – Expand incident response, alert routing, and system
  validation so professional deployments can demonstrate reliability.
- [ ] **Dead-code eradication** – Batch-delete unused modules flagged by the cleanup
  report and tighten import guards to prevent shims from resurfacing.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】
- [ ] **Governance and compliance** – Build the reporting cadence for KYC/AML,
  regulatory telemetry, and audit storage prior to live-broker pilots.【F:docs/technical_debt_assessment.md†L58-L112】

## Actionable to-do tracker

| Status | Task | Owner hint | Linkage |
| --- | --- | --- | --- |
| [ ] | Stand up production-grade ingest slice with parameterised SQL and supervised tasks | Data backbone squad | Now → Operational data backbone |
| [ ] | Deliver executable HOW/ANOMALY organs with lineage telemetry and regression coverage | Sensory cortex squad | Now/Next → Sensory + evolution execution |
| [ ] | Roll out deterministic risk API and supervised runtime builder across execution modules | Execution & risk squad | Now/Next → Risk and runtime safety |
| [ ] | Expand CI to cover ingest orchestration, risk policies, and observability guardrails | Quality guild | Now → Quality and observability |
| [ ] | Purge deprecated shims and close dead-code backlog | Platform hygiene crew | Later → Dead code and duplication |

## Execution guardrails

- Keep policy, lint, types, and pytest checks green on every PR; treat CI failures
  as blockers.
- Update context packs and roadmap status pages alongside significant feature
  work; stale documentation is considered a regression.【F:docs/technical_debt_assessment.md†L90-L112】
- Maintain the truth-first status culture: mock implementations must remain
  labelled and roadmapped until replaced by production-grade systems.【F:docs/DEVELOPMENT_STATUS.md†L7-L35】
