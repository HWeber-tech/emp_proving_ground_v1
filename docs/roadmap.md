# Modernization Roadmap

This roadmap orchestrates the remaining modernization work for the EMP Professional
Predator codebase. Pair it with the living
[Technical Debt Assessment](technical_debt_assessment.md) when planning discovery
spikes, execution tickets, or milestone reviews.

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
- Expand Stage 4 formatter enforcement through the remaining data foundation
  packages (`replay/`, `schemas.py`) and socialise the diffs ahead of the rollout PRs.
- Draft targeted regression scenarios for execution/risk modules so upcoming tests
  cover partial fills, reconciliation, and drawdown recoveries.
- Capture the alert-drill cadence, webhook design, and flake-ingestion workflow in the
  observability plan while keeping GitHub issue automation healthy.
- Break apart the highest-fanin `src/core/` modules to shrink the backlog surfaced in
  the latest dead-code audit.

### 60-day outcomes (Next)
- Stage 4 formatting now covers `src/data_integration/`, `src/operational/`, and
  `src/performance/`; retire the allowlist once the slices stay green in CI.
- Land cross-cutting regression suites that chain the orchestration runtime through
  FIX execution and risk managers with deterministic fixtures.
- Stand up a lightweight telemetry dashboard that visualises drill history, flake
  frequency, and formatter coverage trends from CI artifacts.
- Publish the first modularisation PRs for `src/core/` and related helpers, pairing
  them with updated docs and examples.

### 90-day considerations (Later)
- Replace the formatter allowlist guard with repo-wide `ruff format` enforcement in CI
  and pre-commit once the backlog is cleared.
- Graduate the regression program into scenario and load testing after the core
  coverage gaps close.
- Evaluate runtime (non-CI) observability requirements and budget for any additional
  infrastructure in the next roadmap revision.
- Keep recurring dead-code audits on a predictable cadence and automate ticket
  creation for anything that survives more than two passes.

## Portfolio snapshot

| Initiative | Phase | Outcome we need | Current status | Next checkpoint |
| --- | --- | --- | --- | --- |
| Formatter normalization | 6 | Repository passes `ruff format --check .` without relying on an allowlist. | Stage 4 enforces `src/sensory/`, all data foundation packages, and the `src/data_integration/`, `src/operational/`, and `src/performance/` directories; only tooling helpers remain outside the formatter guard. | Format `scripts/check_formatter_allowlist.py` helpers and plan the allowlist retirement (Week 4). |
| Regression depth in trading & risk | 7 | Coverage hotspots wrapped in deterministic regression suites. | Regression suites now cover execution-engine partial fills/retries, risk drawdown recovery, and property-based order mutations alongside the existing FIX, config, and orchestration smoke tests. | Capture coverage deltas in `docs/status/ci_health.md` and plan the orchestration + risk end-to-end scenario. |
| Operational telemetry & alerting | 8 | CI failures surface automatically with actionable context. | GitHub issue automation is live, alert drills run via the `alert_drill` dispatch, and flake telemetry is stored in git. Slack/webhook mirroring and dashboards remain open. | Document the webhook rollout plan, schedule quarterly drills, and surface telemetry trends in the CI health dashboard. |
| Dead code remediation & modular cleanup | 9 | Dead-code audit remains actionable and high-fanin modules are decomposed. | Latest audit triage logged; unused imports removed. Structural decomposition for `src/core/` families and supporting docs are still outstanding. | Deliver the first decomposition PR (targeting `src/core/state_store.py` dependents) with updated documentation and audit sign-off. |

## Active modernization streams

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

- [ ] Document the Slack/webhook integration plan (owners, secrets, rollout steps)
      directly in the observability plan and CI health snapshot.
- [ ] Automate ingestion of `tests/.telemetry/flake_runs.json` into a lightweight
      dashboard or summary table that highlights retry frequency and failure types.
- [ ] Publish the alert-drill calendar (quarterly cadence) and add a checklist for
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

- [ ] Decompose `src/core/state_store.py` by extracting persistence adapters and
      documenting the new interfaces.
- [ ] Triage `src/core/performance/` and `src/core/risk/` helpers for deletion or
      consolidation, mapping dependencies before submitting PRs.
- [ ] Align relevant docs and examples after each deletion/refactor so onboarding
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
