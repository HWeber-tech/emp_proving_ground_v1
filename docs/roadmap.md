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
| Legacy footprint | Canonical risk and evolution configuration now resolve through their source modules, with both legacy shims removed while integration guides still trail reality. | 【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L52】【F:docs/reports/CLEANUP_REPORT.md†L74-L84】【F:docs/legacy/README.md†L1-L12】 |

## Gaps to close

- [ ] **Operational data backbone** – Deliver real Timescale/Redis/Kafka services,
  parameterise SQL, and supervise ingest tasks instead of relying on mocks.
  - *Progress*: Timescale retention telemetry now validates schema/table/timestamp
    identifiers, parameterises retention queries, and documents the contract via
    regression tests so institutional slices cannot inject raw SQL through policy
    definitions.【F:src/operations/retention.py†L42-L195】【F:tests/operations/test_data_retention.py†L1-L118】
  - *Progress*: Timescale reader guards schema/table/column identifiers, normalises
    symbol filters, and parameterises queries before execution so ingest consumers
    cannot smuggle unsafe SQL through configuration, with security regressions
    covering identifier fuzzing.【F:src/data_foundation/persist/timescale_reader.py†L19-L210】【F:tests/data_foundation/test_timescale_reader_security.py†L1-L45】
- [ ] **Sensory + evolution execution** – Replace HOW/ANOMALY stubs, wire lineage
  telemetry, and prove adaptive strategies against recorded data.
  - *Progress*: Ecosystem optimizer now defends against unsafe genomes and
    malformed regime metadata by normalising canonical models, skipping
    non-numeric parameters, and logging adapter failures with pytest coverage on
    each guardrail so evolution runs cannot silently corrupt state.【F:src/ecosystem/optimization/ecosystem_optimizer.py†L59-L230】【F:tests/ecosystem/test_ecosystem_optimizer_hardening.py†L1-L70】
- [ ] **Risk and runtime safety** – Enforce `RiskConfig`, finish the builder rollout,
  adopt supervised async lifecycles, and purge deprecated facades.
  - *Progress*: The trading risk gateway now drives portfolio checks through the
    real risk manager, records liquidity and policy telemetry, and rejects
    intents that breach drawdown, exposure, or liquidity limits, aligning the
    trading stack with the deterministic enforcement promised by the real
    manager.【F:src/trading/risk/risk_gateway.py†L161-L379】【F:tests/current/test_risk_gateway_validation.py†L74-L206】
  - *Progress*: Trading manager initialises its portfolio risk manager via the
    canonical `get_risk_manager` facade, exposes the core engine’s snapshot and
    assessment APIs, and keeps execution telemetry aligned with the deterministic
    risk manager surfaced by the runtime builder.【F:src/trading/trading_manager.py†L105-L147】【F:src/risk/risk_manager_impl.py†L533-L573】
- [ ] **Quality and observability** – Expand regression coverage, close the
  documentation gap, and track remediation progress through CI snapshots.
  - *Progress*: Event bus health publishing now logs unexpected primary-bus
    failures, only falls back to the global bus after emitting warnings, and
    raises on unknown exceptions so telemetry regressions surface instead of
    disappearing behind silent fallbacks, with tests covering the warning and
    escalation paths.【F:src/operations/event_bus_health.py†L234-L281】【F:tests/operations/test_event_bus_health.py†L95-L147】
  - *Progress*: Bootstrap control centre helpers now log champion payload,
    trading-manager method, and formatter failures, keeping operational
    diagnostics visible during bootstrap runs and documenting the logging
    behaviour under pytest.【F:src/operations/bootstrap_control_center.py†L31-L115】【F:tests/current/test_bootstrap_control_center.py†L178-L199】
  - *Progress*: System validation telemetry escalates runtime publish failures
    into warnings, raises on unexpected errors, and regression tests capture the
    fallback handling so readiness dashboards surface degraded validation runs
    instead of missing them.【F:src/operations/system_validation.py†L269-L312】【F:tests/operations/test_system_validation.py†L85-L137】
- [ ] **Dead code and duplication** – Triage the 168-file dead-code backlog and
  eliminate shim exports so operators see a single canonical API surface.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】
  - *Progress*: Removed the deprecated risk and evolution configuration shims so
    consumers converge on the canonical modules without drift when new services
    arrive.【F:docs/reports/CLEANUP_REPORT.md†L74-L84】【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L52】
  - *Progress*: Retired the legacy strategy template package and rewrote the
    canonical mean reversion regression to exercise the modern trading
    strategies API, shrinking the dead-code backlog and aligning tests with the
    production surface.【F:docs/reports/CLEANUP_REPORT.md†L87-L106】【F:tests/current/test_mean_reversion_strategy.py†L1-L80】

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
    - *Progress*: Strategy registry now opens per-operation SQLite connections,
      raises typed errors, and uses parameterised statements so governance writes
      are supervised instead of silently swallowed.【F:src/governance/strategy_registry.py†L1-L347】
    - *Progress*: Data retention telemetry guards schema/table/timestamp
      identifiers and uses SQLAlchemy parameter binding so operators cannot
      inject raw SQL through policy configuration, with tests covering the
      hardened contract.【F:src/operations/retention.py†L42-L195】【F:tests/operations/test_data_retention.py†L1-L118】
    - *Progress*: Yahoo ingest persistence sanitises DuckDB table names, uses
      parameterised deletes, and asserts gateway error handling so bootstrap
      persistence cannot be hijacked by crafted identifiers or silent
      failures.【F:src/data_foundation/ingest/yahoo_ingest.py†L82-L151】【F:tests/data_foundation/test_yahoo_ingest_security.py†L32-L80】【F:tests/data_integration/test_yfinance_gateway_security.py†L12-L56】
    - *Progress*: Hardened the IC Markets operational bridge with classified
      network/message error handling, managed retries, and structured logging so
      FIX connectivity failures surface instead of stalling silent loops.【F:src/operational/icmarkets_robust_application.py†L22-L333】
    - *Progress*: Security posture publishing now warns and falls back to the
      global bus when runtime publishing fails, raises on unexpected errors, and
      documents the error-handling paths under pytest so telemetry outages cannot
      disappear silently.【F:src/operations/security.py†L536-L579】【F:tests/operations/test_security.py†L148-L263】
- [x] **Context pack refresh** – Replace legacy briefs with the updated context in
  `docs/context/alignment_briefs` so discovery and reviews inherit the same
  narrative reset (this change set).
- [ ] **Coverage guardrails** – Extend the CI baseline to include ingest orchestration
  and risk policy regression tests, lifting coverage beyond the fragile 76% line.
  - *Progress*: Added an end-to-end regression for the real portfolio monitor to
    exercise data writes, analytics, and reporting flows under pytest, closing a
    previously untested gap in the trading surface.【F:tests/trading/test_real_portfolio_monitor.py†L1-L77】
  - *Progress*: Added ingest observability and risk policy telemetry regression tests
    so CI surfaces regressions in data backbone snapshots and policy evaluation
    markdown output.【F:tests/data_foundation/test_ingest_observability.py†L1-L190】【F:tests/trading/test_risk_policy_telemetry.py†L1-L124】
  - *Progress*: Coverage telemetry now emits per-domain matrices from the
    coverage XML, with CLI tooling and pytest coverage documenting the JSON/markdown
    contract so dashboards can flag lagging domains without scraping CI logs.【F:tools/telemetry/coverage_matrix.py†L1-L199】【F:tests/tools/test_coverage_matrix.py†L1-L123】【F:docs/status/ci_health.md†L13-L31】
  - *Progress*: CI workflow now fails fast if ingest, operations, trading, and
    governance suites regress by pinning pytest entrypoints and coverage include
    lists to those domains, preventing partial runs from passing unnoticed.【F:.github/workflows/ci.yml†L79-L120】【F:pytest.ini†L1-L14】【F:pyproject.toml†L45-L85】
  - *Progress*: Ingest trend and Kafka readiness publishers now log event bus
    failures and ship regression tests so telemetry gaps raise alerts instead of
    disappearing silently.【F:src/operations/ingest_trends.py†L303-L329】【F:tests/operations/test_ingest_trends.py†L90-L118】【F:src/operations/kafka_readiness.py†L313-L333】【F:tests/operations/test_kafka_readiness.py†L115-L143】
  - *Progress*: Trading position model guardrails now run under pytest,
    asserting timestamp updates, profit recalculations, and close flows so the
    lightweight execution telemetry remains deterministic under CI coverage.【F:tests/trading/test_position_model_guardrails.py†L1-L105】

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
  - *Progress*: Risk gateway wiring now normalises intents, enforces
    drawdown/exposure/liquidity guardrails, and publishes policy decisions so
    trading managers consume the same deterministic risk manager path as the
    runtime builder.【F:src/trading/trading_manager.py†L1-L320】【F:src/trading/risk/risk_gateway.py†L161-L379】【F:tests/current/test_risk_gateway_validation.py†L74-L206】

### Later (90+ days)

- [ ] **Operational readiness** – Expand incident response, alert routing, and system
  validation so professional deployments can demonstrate reliability.
- [ ] **Dead-code eradication** – Batch-delete unused modules flagged by the cleanup
  report and tighten import guards to prevent shims from resurfacing.【F:docs/reports/CLEANUP_REPORT.md†L71-L188】
- [ ] **Governance and compliance** – Build the reporting cadence for KYC/AML,
  regulatory telemetry, and audit storage prior to live-broker pilots.【F:docs/technical_debt_assessment.md†L58-L112】
  - *Progress*: Governance reporting cadence now assembles compliance readiness,
    regulatory telemetry, and Timescale audit evidence into a single artefact,
    publishes the snapshot on the event bus, and trims persisted histories so
    audits inherit deterministic evidence, with pytest covering scheduling,
    publishing, and storage flows.【F:src/operations/governance_reporting.py†L1-L200】【F:tests/operations/test_governance_reporting.py†L1-L152】

## Actionable to-do tracker

| Status | Task | Owner hint | Linkage |
| --- | --- | --- | --- |
| [ ] | Stand up production-grade ingest slice with parameterised SQL and supervised tasks | Data backbone squad | Now → Operational data backbone |
| [ ] | Deliver executable HOW/ANOMALY organs with lineage telemetry and regression coverage | Sensory cortex squad | Now/Next → Sensory + evolution execution |
| [ ] | Roll out deterministic risk API and supervised runtime builder across execution modules | Execution & risk squad | Now/Next → Risk and runtime safety |
| [ ] | Expand CI to cover ingest orchestration, risk policies, and observability guardrails | Quality guild | Now → Quality and observability |
| [ ] | Purge deprecated shims and close dead-code backlog | Platform hygiene crew | Later → Dead code and duplication |

- *Progress*: Risk and evolution configuration now source directly from their
  canonical modules with the legacy shims removed, shrinking the cleanup queue
  and preventing namespace drift.【F:docs/reports/CLEANUP_REPORT.md†L74-L84】【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L52】

## Execution guardrails

- Keep policy, lint, types, and pytest checks green on every PR; treat CI failures
  as blockers.
- Update context packs and roadmap status pages alongside significant feature
  work; stale documentation is considered a regression.【F:docs/technical_debt_assessment.md†L90-L112】
- Maintain the truth-first status culture: mock implementations must remain
  labelled and roadmapped until replaced by production-grade systems.【F:docs/DEVELOPMENT_STATUS.md†L7-L35】
