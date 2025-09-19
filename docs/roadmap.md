# Modernization Roadmap

This roadmap orchestrates the remaining modernization work for the EMP Professional
Predator codebase and reframes the backlog through the lens of the
`EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` concept blueprint. Pair it with the living
[Technical Debt Assessment](technical_debt_assessment.md) when planning discovery
spikes, execution tickets, or milestone reviews.

## Concept alignment context pack

Use these artefacts as the substrate for context-aware planning, ticket grooming,
and code reviews:

- **Concept blueprint** – `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` (truth-first draft of
  the five-layer architecture, compliance posture, and commercialization journeys).
- **Reality snapshot** – [`system_validation_report.json`](../system_validation_report.json)
  and [`docs/technical_debt_assessment.md`](technical_debt_assessment.md) for the
  current engineering baseline.
- **Delta log** – This roadmap plus the quarterly architecture notes capture how the
  implementation is evolving toward the concept claims.

### Promises vs. implementation gaps

| Concept promise | Current state | Roadmap response |
| --- | --- | --- |
| Tiered data backbone with TimescaleDB, Redis, Kafka, and Spark orchestration once we outgrow the 1 GB bootstrap footprint. | Tier‑0 relies on DuckDB/CSV helpers; Yahoo ingest returns placeholders and there is no orchestration for professional data tiers. | Stand up a production-ready ingest slice with TimescaleDB persistence, Redis caching, and Kafka streaming, then wire orchestration that can flip between bootstrap and institutional tiers. |
| 4D+1 sensory cortex delivering superhuman perception with WHY/HOW/WHAT/WHEN/ANOMALY working together. | Only WHY/WHAT/WHEN have lightweight heuristics, ANOMALY is a stub, and HOW is absent from the default runtime. | Build the HOW organ, connect anomaly detection, and upgrade existing sensors with calibrated macro/order-flow data feeds. |
| Evolutionary intelligence that adapts live strategies through genetic evolution and meta-learning. | Tier‑2 evolution mode throws `NotImplementedError`; genomes are stubs and trading decisions remain static thresholds. | Implement population management, realistic genomes, and fitness evaluation; make Tier‑2 executable behind feature flags. |
| Institutional execution, risk, and compliance with OMS/EMS connectivity plus regulatory workflows (MiFID II, Dodd‑Frank, KYC/AML). | Execution is an in-memory simulator, risk checks are minimal, and compliance layers are no-ops. | Deliver FIX/EMS bridges, expand risk controls, and prototype compliance workflows with audit persistence. |
| Operational readiness with monitoring, alerting, security, and disaster recovery suitable for higher tiers. | Bootstrap control center aggregates telemetry but broader ops/security tooling is missing. | Design the monitoring stack, access controls, and backup routines aligned with the concept doc’s operational maturity targets. |
| Commercial roadmap that validates the €250 → institutional journey and 95 % cost savings. | Bootstrap monitor assumes a $100 000 paper account; there is no empirical ROI model or cost tracking. | Instrument cost/fee models, track ROI experiments, and align marketing claims with evidence. |

> **Context engineering tip:** When drafting epics or writing code, pull the relevant
> concept excerpt into the issue description and link the gap table row. This keeps
> discussions anchored to the documented intent.

## Execution rhythm

- **Stage work into reviewable tickets** – Translate bullets into 1–2 day tasks with a
  Definition of Done, clear owner, and explicit validation steps.
- **Run a Now / Next / Later board** – Groom items as soon as they move to "Later" so
  the next engineer can start without another planning meeting.
- **Time-box discovery** – Use 4–8 hour spike tickets when investigation is required
  and close them with a written summary so execution inherits the findings.
- **Weekly sync** – Spend 15 minutes each Friday to capture status, blockers, and any
  resequencing. Update this document and the tracking board immediately afterwards.
- **Keep telemetry fresh** – Refresh formatter progress, coverage deltas, and alerting
  status in [`docs/status/ci_health.md`](status/ci_health.md) so dashboards mirror the
  roadmap.

## Way forward

### 30-day outcomes (Now)
- Draft alignment briefs for the five major gap areas (data backbone, sensory
  cortex, evolution engine, institutional risk/compliance, operational readiness)
  that translate encyclopedia promises into actionable epics and acceptance
  criteria.
- Ship the “concept context pack” – issue templates, pull request checklists, and
  architecture notes that quote `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` sections next
  to current code references so contributors inherit the same narrative.
- Prototype a TimescaleDB-backed ingest slice fed by the existing Yahoo bootstrap
  downloader to exercise migrations, connection pooling, and state-store wiring.
- Add scaffolding tests or notebooks that demonstrate how the new ingest slice will
  be queried by sensors and risk modules once Redis/Kafka are introduced.
- Adjust public-facing docs (README, roadmap preface, launch deck) to clarify that
  v2.3 is a concept draft with hypotheses awaiting validation.

### 60-day outcomes (Next)
- Land the first production-grade ingest vertical: TimescaleDB persistence, Redis
  caching for hot symbols, and a Kafka topic that mirrors intraday updates into the
  runtime event bus.
- Deliver a minimally viable HOW organ and revive the ANOMALY sensor so the default
  runtime exercises all five dimensions with calibrated thresholds and audit logs.
- Replace the stub genome provider with a small but real genome catalogue sourced
  from historical strategies; wire population management into the evolution engine
  behind a feature flag.
- Publish compliance starter kits – MiFID II/Dodd-Frank control matrices, KYC/AML
  workflow outlines, and audit storage requirements mapped to existing interfaces.
- Extend operational telemetry by defining monitoring SLOs, alert channels, and
  backup/restore drills for the new data services.

### 90-day considerations (Later)
- Graduate the data backbone by stress-testing Kafka/Spark batch jobs, documenting
  failover procedures, and proving that ingest tiers can switch without downtime.
- Expand sensory/evolution validation with live-paper trading experiments, anomaly
  drift monitoring, and feedback loops that tune strategy genomes automatically.
- Deliver the first broker/FIX integration pilot complete with supervised async
  lifecycles, expanded risk gates, compliance checkpoints, and observability hooks.
- Publish ROI instrumentation – track fee savings, infrastructure spend, and
  capital efficiency so the €250 → institutional journey is grounded in evidence.
- Update marketing and onboarding assets once the above pilots demonstrate the
  promised capabilities.

## Document-driven high-impact streams

These streams translate the encyclopedia ambitions into execution tracks. Each
stream should keep the concept excerpts, current-state references, and acceptance
criteria together inside its epic template so engineers inherit the same context.

### Stream A – Institutional data backbone

**Mission** – Replace the bootstrap-only ingest helpers with tier-aware, resilient
data services (TimescaleDB, Redis, Kafka, Spark) that can scale alongside clients.

**Key deliverables**

- TimescaleDB schema design (markets, macro, alternative data) with migration
  automation and retention policies documented.
- Redis caching strategy for hot symbols, limits, and session state, including
  eviction policies and observability hooks.
- Kafka topics that replicate intraday updates into the runtime event bus and feed
  downstream Spark jobs for batch analytics.
- Orchestration logic capable of switching between bootstrap and institutional tiers
  with confidence checks, rollback steps, and operator documentation.
- Validation suites (pytest + notebooks) that prove ingest freshness, latency, and
  recovery times meet agreed SLOs.

**Dependencies & context** – Coordinate with existing state-store refactors,
deployment automation, and the ops telemetry initiative to ensure new services are
monitored from day one.

### Stream B – 4D+1 sensory cortex & evolution engine

**Mission** – Deliver all five sensory organs with calibrated data feeds and revive
the evolution engine so strategies adapt continuously and surface anomaly insights.

**Key deliverables**

- HOW organ implementation focused on order-flow/microstructure or execution-cost
  analytics, with dependency injection for new data feeds.
- Anomaly detection service connected to the runtime bus, leveraging statistical or
  ML detectors with explainability hooks.
- Upgraded WHY/WHAT/WHEN heuristics that ingest macro data, news sentiment, and
  technical signals via the new data backbone.
- Evolution engine enhancements: real genome catalogue, population lifecycle,
  fitness evaluation metrics, and experiment logging for auditability.
- Integration tests and evaluation harnesses that compare sensor outputs against
  historical benchmarks and flag drift.

**Dependencies & context** – Needs the data backbone, risk/compliance input on
acceptable automated adaptations, and collaboration with the research team for
feature sourcing.

### Stream C – Execution, risk, compliance, and ops readiness

**Mission** – Graduate from the in-memory simulator to institutional-grade
execution and governance with supervised async lifecycles and documented controls.

**Key deliverables**

- FIX/EMS adapter pilot with retry/backoff, drop-copy ingestion, and reconciling
  ledgers.
- Expanded risk engine enforcing tiered limits, drawdown guards, leverage checks,
  and configurable rule sets stored in version-controlled policy files.
- Compliance starter kits turned into executable workflows: KYC/AML checklist,
  MiFID II transaction reporting drafts, audit-trail persistence, and operator
  review cadences.
- Operational hardening: monitoring SLOs, alert routing, security controls,
  credential rotation playbooks, backup/restore runbooks.
- Evidence log demonstrating ROI metrics, cost savings, and institutional readiness
  improvements for stakeholder communications.

**Dependencies & context** – Builds on Streams A/B, the observability roadmap, and
ongoing documentation updates to keep operators and compliance informed.

## Long-horizon remediation plan

| Timeline | Outcomes | Key workstreams |
| --- | --- | --- |
| **0–3 months (Align & prototype)** | Concept promises decomposed into epics, first production data slice live, and sensory/evolution scaffolds in place. | - Publish alignment briefs and context packs linking encyclopedia excerpts to code gaps.<br>- Deploy the TimescaleDB prototype with ingest smoke tests and Redis/Kafka design notes.<br>- Implement HOW organ skeleton, revive anomaly hooks, and replace stub genomes with historical seeds.<br>- Clarify compliance tone in public docs and capture validation hypotheses for ROI claims. |
| **3–6 months (Build & integrate)** | Data backbone reliable in CI/staging, sensors and evolution engine operating on new feeds, compliance workflows documented. | - Harden TimescaleDB/Redis/Kafka services with monitoring, backups, and orchestration toggles.<br>- Finish 4D+1 sensor uplift, integrate anomaly analytics, and run paper-trading evolution experiments.<br>- Deliver compliance starter kits, risk policy files, and operational telemetry for new services.<br>- Begin FIX/EMS adapter pilot with supervised async patterns and reconciliation tests. |
| **6–12 months (Institutionalise)** | Execution stack, risk controls, and compliance workflows withstand institutional scrutiny with evidence-backed ROI. | - Expand into Spark batch analytics, tiered deployment automation, and failover drills.<br>- Onboard real broker integrations with audit trails, policy versioning, and continuous validation dashboards.<br>- Track ROI metrics (cost savings, capital efficiency) and iterate marketing claims based on data.<br>- Prepare external audits or partner reviews leveraging the evidence log and documentation corpus. |

## Portfolio snapshot

| Initiative | Phase | Outcome we need | Current status | Next checkpoint |
| --- | --- | --- | --- | --- |
| Institutional data backbone | A | Bootstrap ingest upgraded to TimescaleDB + Redis + Kafka with switchable tiers and recovery drills documented. | TimescaleDB prototype underway; Redis/Kafka designs captured in alignment briefs; DuckDB/CSV helpers still power production runs. | Ship the prototype ingest slice with smoke tests and publish orchestration switch design (Week 4). |
| Sensory cortex & evolution uplift | B | All five sensory organs online with calibrated feeds and the evolution engine managing real genomes under feature flags. | WHY/WHAT/WHEN heuristics live; HOW absent; ANOMALY stubbed; evolution Tier‑2 disabled. Alignment brief drafted with data requirements. | Implement HOW organ skeleton, anomaly scaffolding, and seed genomes; integrate into staging runtime (Week 6). |
| Execution, risk, compliance, ops readiness | C | Broker/FIX integration pilot operating with expanded risk controls, compliance workflows, and observability. | In-memory simulator still primary path; compliance/audit shims empty; risk checks minimal. Compliance starter kit outline drafted. | Finalise risk policy file format, document compliance workflows, and begin FIX adapter spike (Week 8). |
| Supporting modernization (formatter, regression, telemetry) | Legacy | Foundational hygiene remains green while high-impact streams ramp up. | Formatter Stage 4 landed, regression suites expanded, telemetry automation live; follow-on cleanups pending. | Keep CI health snapshot current and scope maintenance tickets so core teams stay unblocked. |

## Active modernization streams

Legacy initiatives below remain in flight to keep the repo healthy while the
document-driven streams ramp. Treat them as supporting tracks—ensure they stay
green but do not let them crowd out the higher leverage work above.

### Initiative 1 – Formatter normalization (Phase 6)

**Mission** – Land `ruff format` in reviewable slices until the repository passes
`ruff format --check .` without leaning on `config/formatter/ruff_format_allowlist.txt`.

**Definition of done**

- Stage 0 (`tests/current/`) formatted, pytest green, and allowlist expanded
  accordingly.
- Stage 1 (`src/system/`, `src/core/configuration.py`) formatted with documented
  manual edits and follow-up tickets for non-mechanical cleanups.
- Remaining directories sequenced in [`docs/development/formatter_rollout.md`](development/formatter_rollout.md)
  with owners and merge order captured.
- CI enforces the formatter globally and the allowlist shrinks to empty (or is
  removed entirely) without exceptions.
- Contributor guidance in [`docs/development/setup.md`](development/setup.md) and
  the PR checklist reflects the post-rollout workflow.

**Key context**

- `scripts/check_formatter_allowlist.py`
- `config/formatter/ruff_format_allowlist.txt`
- [`docs/development/formatter_rollout.md`](development/formatter_rollout.md)
- [`docs/development/setup.md`](development/setup.md)
- [`docs/status/ci_health.md`](status/ci_health.md) – formatter progress snapshots

**Recent progress**

- Stage 3 completed for `src/sensory/organs/dimensions/` with pytest remaining green.
- Stage 4 now enforces `src/sensory/` and the `src/data_foundation/config/`,
  `src/data_foundation/ingest/`, and `src/data_foundation/persist/` packages.
- Formatter sequencing notes updated in the rollout guide with owner assignments and
  merge windows.
- `src/data_foundation/replay/` and `src/data_foundation/schemas.py` normalized under
  `ruff format`, and the Stage 4 operational/performance briefing now coordinates
  owners, reviewers, and freeze windows.

**Now**

- [x] Normalize `src/data_foundation/replay/` with targeted replays to verify no data
      integrity drift before widening the allowlist. (Verified via `ruff format` and
      replay smoke checks.)
- [x] Diff `src/data_foundation/schemas.py` against formatter output, capture any
      manual reconciliations, and stage pytest smoke checks for downstream users.
- [x] Publish a Stage 4 briefing that lines up `src/operational/` and
      `src/performance/` slices with nominated reviewers and freeze windows.

**Next**

- [x] Land the data integration, operational, and performance formatting PRs with
      paired allowlist updates and focused pytest runs covering
      `src/operational/metrics.py`, `src/performance/vectorized_indicators.py`, and
      the ingestion slices.
- [ ] Collapse the remaining allowlist entries and wire `ruff format --check .` into
      the main CI workflow once the Stage 4 backlog clears.
- [ ] Update contributor docs (`setup.md`, PR checklist) to describe the new default
      formatter workflow and local tooling expectations.

**Delivery checkpoints**

- Data foundation replay + schemas formatting ready for review (Week 1).
- Operational/performance slices rehearsed and queued (Week 3).
- Allowlist removal RFC circulated with rollout guardrails (Week 5).

**Later**

- [ ] Retire the allowlist guard and remove redundant formatters (for example,
      Black) once Ruff owns enforcement end-to-end.

**Risks & watchpoints**

- High-churn directories causing repeated merge conflicts – mitigate by staging PRs
  early in the week and communicating freeze windows.
- Generated files or vendored assets accidentally formatted – confirm each slice
  honors project-level excludes before landing.
- Formatting changes obscuring behavior tweaks – insist on mechanical-only commits
  paired with focused follow-ups for real fixes.

**Telemetry**

- Track formatted-directory count and allowlist size in the CI health snapshot.
- Flag the rollout timeline in retrospectives so future contributors understand the
  historical sequencing.

### Initiative 2 – Regression depth in trading & risk (Phase 7)

**Mission** – Convert the CI baseline hotspots into deterministic regression suites
that guard trading execution, risk controls, and orchestration wiring.

**Definition of done**

- Coverage for `src/operational/metrics.py`, `src/trading/models/position.py`,
  `src/data_foundation/config/`, and
  `src/sensory/dimensions/why/yield_signal.py` improves measurably and holds steady
  in CI.
- FIX execution flows (order routing, reconciliation, error handling) have
  explicit success and failure scenarios documented and enforced by pytest.
- Risk guardrails (position limits, drawdown gates, Kelly sizing) are exercised
  across happy paths and failure modes with clear assertions.
- Orchestration composition tests verify adapters, event bus wiring, and optional
  module degradation across supported configurations.
- Coverage deltas captured in [`docs/status/ci_health.md`](status/ci_health.md)
  after each regression batch.

**Key context**

- [`docs/ci_baseline_report.md`](ci_baseline_report.md)
- Existing regression suites in `tests/current/`
- `src/trading/models/position.py`
- `src/operational/metrics.py`
- `src/data_foundation/config/`
- `src/sensory/dimensions/why/yield_signal.py`

**Recent progress**

- Baseline hotspots decomposed into regression tickets and logged in
  [`docs/status/regression_backlog.md`](status/regression_backlog.md).
- Deterministic FIX failure-path coverage added in
  `tests/current/test_fix_manager_failures.py`.
- Regression tests landed for position lifecycle accounting, risk guardrails, data
  foundation config loaders, and orchestration runtime smoke checks.

**Now**

- [x] Extend `tests/current/test_execution_engine.py` (or add a new suite) to cover
      partial fills, retries, and reconciliation paths in
      `src/trading/execution/execution_engine.py`. (`tests/current/test_execution_engine.py`)
- [x] Add regression coverage for drawdown recovery and Kelly-sizing adjustments in
      `src/risk/risk_manager_impl.py` with fixtures that mirror production configs.
      (`tests/current/test_risk_manager_impl.py`)
- [x] Introduce property-based tests around order mutation flows in
      `src/trading/models/order.py` to lock in serialization and validation logic.
      (`tests/current/test_order_model_properties.py`)

**Next**

- [ ] Chain orchestration, execution, and risk modules in an end-to-end scenario test
      that verifies event bus wiring and fallback behavior.
- [ ] Record coverage deltas in the CI health snapshot after each regression landing
      and alert on regressions outside agreed thresholds.
- [ ] Expand sensory regression focus to `src/sensory/dimensions/why/yield_signal.py`
      with fixtures derived from historical market data.

**Later**

- [ ] Expand into scenario-based integration tests once formatter noise is gone and
      churn stabilizes.
- [ ] Capture coverage trendlines in the CI dashboard and celebrate modules that
      cross agreed thresholds.

**Risks & watchpoints**

- Mocked subsystems may give false confidence – document limitations and plan for
  real integrations in future roadmaps.
- Flaky tests can erode trust – add telemetry hooks to surface retries and
  investigate immediately.

**Telemetry**

- Update coverage by module in the CI health snapshot after each landing.
- Track flaky-test counts or retry rates to feed back into the observability plan.

### Initiative 3 – Operational telemetry & alerting (Phase 8)

**Mission** – Provide actionable observability for CI so failures surface without
manual log digging or the deprecated Kilocode bridge.

**Definition of done**

- A single alerting channel (GitHub issue automation, Slack webhook, or email
  digest) selected, documented, and validated with a forced failure.
- Lightweight dashboard or README section summarizing recent CI runs, formatter
  status, and coverage trends published and maintained.
- On-call expectations, escalation paths, and response targets codified alongside
  the observability plan.
- CI failure telemetry (e.g., pytest flake metadata) stored in an accessible
  location for trend analysis.

**Key context**

- `.github/workflows/ci.yml`
- `.github/workflows/ci-failure-alerts.yml`
- [`docs/operations/observability_plan.md`](operations/observability_plan.md)
- [`docs/status/ci_health.md`](status/ci_health.md)

**Recent progress**

- GitHub issue automation is live and validated through the `alert_drill` dispatch
  input in CI.
- Flake telemetry captured in `tests/.telemetry/flake_runs.json` with documentation
  in the observability plan.
- CI health dashboard refreshed with formatter expansion notes and alert-drill
  references.

**Now**

- [x] Document the Slack/webhook integration plan (owners, secrets, rollout steps)
      directly in the observability plan and CI health snapshot.
- [x] Automate ingestion of `tests/.telemetry/flake_runs.json` into a lightweight
      dashboard or summary table that highlights retry frequency and failure types.
- [x] Publish the alert-drill calendar (quarterly cadence) and add a checklist for
      pre/post drill verification steps.

**Next**

- [ ] Deliver the Slack/webhook bridge once credentials are provisioned and run a
      forced-failure drill to validate the end-to-end flow.
- [ ] Expand telemetry capture to include coverage trendlines and formatter adoption
      metrics with references in CI artifacts.
- [ ] Evaluate whether runtime (non-CI) observability hooks belong in this initiative
      or a follow-on roadmap.

**Later**

- [ ] Evaluate whether runtime observability (beyond CI) should join the roadmap
      once formatter and regression work settles.
- [ ] Schedule quarterly reviews of alert effectiveness with on-call participants.

**Risks & watchpoints**

- Alert fatigue if the signal is noisy – ensure alerts auto-resolve and include
  actionable context.
- Ownership drift – keep rotation details in the observability plan and revisit at
  each roadmap review.

**Telemetry**

- Capture alert tests, run history, and open incidents in the CI health document.
- Track mean time to acknowledgment (MTTA) and resolution (MTTR) once alerts are
  live.

### Initiative 4 – Dead code remediation & modular cleanup (Phase 9)

**Mission** – Keep the dead-code signal actionable while decomposing high-fanin
modules into maintainable components.

**Definition of done**

- Latest dead-code audit findings triaged as delete, refactor, or intentional
  dynamic hook with rationale captured.
- Confirmed dead paths removed with documentation or tests updated to reflect any
  behavioral changes.
- Remaining monolithic modules in `src/core` and adjacent packages decomposed into
  focused components without breaking public interfaces.
- Recurring audits scheduled after each cleanup batch, with results appended to
  [`docs/reports/dead_code_audit.md`](reports/dead_code_audit.md).

**Key context**

- [`docs/reports/dead_code_audit.md`](reports/dead_code_audit.md)
- `scripts/audit_dead_code.py`
- `.github/workflows/dead-code-audit.yml`
- Architecture references in `docs/architecture/`

**Recent progress**

- September audit triage logged with decisions on retained protocol parameters.
- Unused `_nn` type import and stale parity-checker helper removed alongside the
  latest smoke tests.
- Audit backlog prioritized for upcoming cleanup passes.

**Now**

- [x] Decompose `src/core/state_store.py` by extracting persistence adapters and
      documenting the new interfaces.
- [x] Triage `src/core/performance/` and `src/core/risk/` helpers for deletion or
      consolidation, mapping dependencies before submitting PRs.
- [x] Align relevant docs and examples after each deletion/refactor so onboarding
      material stays current.

**Next**

- [ ] Schedule automated dead-code audits post-merge and file tickets for anything
      that persists across two consecutive scans.
- [ ] Pair with the architecture guild to identify additional high-fanin modules and
      charter decomposition spikes.
- [ ] Integrate module fan-in metrics into the CI health snapshot to track progress.

**Later**

- [ ] Continue scheduled audits after each cleanup batch, noting trends and
      tracking resolved findings in the report.
- [ ] Evaluate whether additional tooling or static analysis would reduce audit
      noise.

**Risks & watchpoints**

- Accidentally deleting dynamic entry points – document intentional keepers and add
  regression tests where reasonable.
- Audit fatigue – keep the report curated so future scans remain trustworthy.

**Telemetry**

- Log each audit pass, decisions made, and resulting PRs in the audit report.
- Track module fan-in or fan-out metrics where available to measure decomposition
  impact.

### Initiative 5 – Runtime orchestration & risk hardening (Phase 10)

**Mission** – Harden the live-trading runtime by carving clear service boundaries,
supervising async workloads, and enforcing the risk policies that already exist in
configuration.

**Definition of done**

- `main.py` replaced by a dependency-injected application builder with discrete
  entrypoints for ingestion and live trading plus documented shutdown hooks.
- Background services supervised via structured task groups with deterministic
  cancellation and shutdown tests.
- `RiskManager` enforces `RiskConfig` inputs (position sizing, leverage, drawdown,
  exposure) with regression coverage and documentation updates.
- Operator playbooks, architecture notes, and the top-level `README.md` reflect the
  new runtime builder, task supervision story, and rollback procedures.
- Safety and risk configuration changes flow through a documented review workflow
  with compliance/operations sign-off and audit breadcrumbs.
- All system validation checks in `system_validation_report.json` passing in CI,
  with dashboards highlighting drift.
- Public API exports (for example `src/core/__init__.py`) cleaned up so advertised
  symbols exist, are documented, or are intentionally deprecated.

**Key context**

- `main.py`
- `src/risk/risk_manager_impl.py`
- `src/core/__init__.py`
- `src/brokers/fix/` adapters and sensory orchestrators
- `system_validation_report.json`
- `README.md`, `docs/ARCHITECTURE_REALITY.md`, and future runtime/runbook drafts

**Recent progress**

- Technical debt assessment documented the monolithic runtime, unsupervised async
  tasks, and ineffective risk enforcement.
- Preliminary regression suites now touch execution-engine partial fills and risk
  drawdown recovery, providing scaffolding for broader runtime coverage.
- Formatter and modular cleanup work reduced noise around `src/core/` so runtime
  refactors can proceed with less churn.

**Now**

- [ ] Draft a runtime builder and shutdown sequence design that separates ingestion
      and trading workloads, including testing strategy and rollout plan.
- [ ] Inventory `asyncio.create_task` usage across brokers, sensory organs, and
      orchestrators, documenting supervision gaps and proposed `TaskGroup`
      migrations.
- [ ] Map `RiskConfig` parameters to required enforcement logic, outlining tests,
      documentation updates, and telemetry hooks.
- [ ] Define the compliance/operations review workflow for safety configuration
      changes and catalogue the observability gaps in FIX/orchestrator adapters.

**Next**

- [ ] Implement the application builder with dedicated CLIs, integrate structured
      shutdown hooks, and add smoke tests for restart flows.
- [ ] Replace ad-hoc event loops in FIX adapters with supervised async bridges or
      executor shims, validating graceful shutdown in regression suites.
- [ ] Rebuild `RiskManager` to honor leverage, exposure, and drawdown limits with
      deterministic pytest coverage and updated operator guides.
- [ ] Deliver structured logging, metrics, and health checks for the runtime
      builder, FIX bridges, and orchestrators; record steady-state expectations in
      the runbook.

**Later**

- [ ] Evaluate carving FIX and orchestration adapters into separately deployable
      services with health checks and metrics once the builder lands.
- [ ] Extend system validation into continuous monitoring (dashboards, alerts) so
      drift is caught immediately.
- [ ] Introduce configuration-as-code or policy-versioning workflows once risk
      enforcement stabilizes.
- [ ] Capture audit trails for runtime configuration changes and integrate them
      with SBOM/policy reporting as part of the compliance toolkit.

**Risks & watchpoints**

- Runtime refactors can destabilize trading loops – insist on feature flags and
  rollback plans for each landing.
- Async supervision changes may expose latent race conditions; schedule paired
  regression runs and soak tests.
- Risk guardrail reimplementation could block orders if misconfigured – document
  defaults and provide sandbox rehearsals before rollout.

**Telemetry**

- Track runtime validation status and shutdown test coverage alongside existing CI
  health metrics.
- Record risk enforcement outcomes (violations caught, config versions) to prove
  guardrails are active.
- Capture namespace cleanup deltas (removed/added exports) in changelogs for
  downstream consumers.

## Completed phases (0–5)

| Phase | Focus | Status | Highlights |
| --- | --- | --- | --- |
| 0 | Immediate hygiene | ✅ Complete | Retired the Kilocode bridge, resolved `.github/workflows/ci.yml` merge markers, and captured the CI baseline. |
| 1 | Policy consolidation | ✅ Complete | Centralized forbidden-integration checks and aligned `config.yaml` with the FIX-only posture. |
| 2 | Dependency & environment hygiene | ✅ Complete | Promoted `requirements/base.txt`, pinned the development toolchain, and added the runtime requirements check CLI. |
| 3 | Repository cleanup | ✅ Complete | Pruned stray artifacts, added guardrails to keep them out, and seeded the dead-code audit backlog. |
| 4 | Test coverage & observability | ✅ Complete | Expanded regression nets across configuration, FIX parity, risk management, and orchestration while improving CI log surfacing. |
| 5 | Strategic refactors | ✅ Complete | Decomposed high-coupling modules (e.g., `src/core/interfaces`, trading performance tracker) and refreshed architecture docs to match the layering contract. |

Consult the linked documentation in each phase for implementation details when
debugging regressions or planning follow-up work.

## Review cadence & reporting

- Keep a shared checklist linked to this roadmap so contributors can claim items
  and capture findings.
- Review progress in weekly debt triage meetings and update the Now / Next / Later
  board accordingly.
- Revisit the roadmap quarterly (or after completing any initiative above) to
  adjust priorities based on emerging risks or product requirements.
- Note material updates in commit messages so change history explains why items
  moved or definitions shifted.
