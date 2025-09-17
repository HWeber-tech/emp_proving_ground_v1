# Modernization Roadmap

This roadmap orchestrates the remaining modernization work for the EMP Professional
Predator codebase. Pair it with the living
[Technical Debt Assessment](technical_debt_assessment.md) when planning discovery
spikes, execution tickets, or milestone reviews.

## Execution rhythm

- **Stage work into reviewable tickets** – Translate bullets into 1–2 day tasks with a
  Definition of Done, clear owner, and explicit validation steps.
- **Run a Now / Next / Later board** – Groom items as soon as they move to 'Later' so
  the next engineer can start without another planning meeting.
- **Time-box discovery** – Use 4–8 hour spike tickets when investigation is required
  and close them with a written summary so execution inherits the findings.
- **Weekly sync** – Spend 15 minutes each Friday to capture status, blockers, and any
  resequencing. Update this document and the tracking board immediately afterwards.
- **Keep telemetry fresh** – Refresh formatter progress, coverage deltas, and alerting
  status in [`docs/status/ci_health.md`](status/ci_health.md) so dashboards mirror the
  roadmap.

## Portfolio snapshot

| Initiative | Phase | Outcome we need | Current status | Next checkpoint |
| --- | --- | --- | --- | --- |
| Formatter normalization | 6 | Repository passes `ruff format --check .` without relying on an allowlist. | Stages 0–2 landed (`tests/current/`, `src/system/`, `src/core/configuration.py`, `src/trading/execution/`, `src/trading/models/`), Stage 3 covers the entire `src/sensory/organs/dimensions/` package (including `__init__.py`, `utils.py`, `what_organ.py`, `when_organ.py`, `why_organ.py`, and the previously normalized organs), and Stage 4 now enforces the entire `src/sensory/` tree (organs, services, tests, vendor shims) via the allowlist after collapsing the module-level entries. | Sequence Stage 4 follow-ups by lining up `src/data_foundation/config/` and neighboring packages while keeping pytest green. |
| Regression depth in trading & risk | 7 | Coverage hotspots wrapped in deterministic regression suites. | Baseline coverage at 76%; first regression PR hardened the data foundation config loaders while remaining hotspots await coverage. | Prioritize the FIX execution suite and orchestration smoke tests next. |
| Operational telemetry & alerting | 8 | CI failures surface automatically with actionable context. | Logs and tails captured; GitHub issue alerts now open/close automatically while Slack/webhook mirroring remains on the to-do list; flake telemetry artifacts write to disk. | Force a failure to validate the channel and fold learnings into the response playbook. |
| Dead code remediation & modular cleanup | 9 | Dead-code audit remains actionable and high-fanin modules are decomposed. | Weekly audit artifact generated; backlog not triaged. | Triage the latest audit, delete safe targets, and schedule the next cleanup batch. |

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

**Now**

- [x] Ship Stage 0 PR formatting `tests/current/`, expand the allowlist, and rerun
      pytest to verify no behavioral drift.
- [x] Publish the Stage 0 change log (manual edits, conflicts to expect) so
      reviewers have context for subsequent slices.
- [x] Normalize Stage 1 (`src/system/`, `src/core/configuration.py`), capture the
      rollout notes, and keep `config/formatter/ruff_format_allowlist.txt`
      alphabetized.
- [x] Dry-run Stage 2 (`src/trading/execution/`, `src/trading/models/`) with a
      paired regression test plan before opening the formatting PR. Completed –
      directories are normalized and the allowlist now enforces them.
- [x] Coordinate Stage 2 merge windows with trading/risk owners to minimize
      conflicts while coverage work lands.
- [x] Assign Stage 3 (`src/sensory/organs/dimensions/`) ownership to the Sensory
      guild rotation and reserve merge windows before each organ lands.
- [x] Dry-run the initial Stage 3 organ (`anomaly_detection.py`), capture manual
      adjustments (none), and confirm pytest stays green prior to formatting.
- [x] Normalize the next Stage 3 organ (`base_organ.py`), document any manual
      edits, and enroll it in the allowlist without regressing coverage.
- [x] Normalize `src/sensory/organs/dimensions/chaos_adaptation.py`, capture any
      manual cleanups, and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/chaos_dimension.py`, capture any
      manual cleanups, and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/anomaly_dimension.py`, capture any
      manual cleanups, and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/data_integration.py`, document
      manual edits (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/integration_orchestrator.py`,
      capture manual cleanups (timing call reflow), and extend the allowlist while
      keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/institutional_tracker.py`,
      document manual edits (none), and extend the allowlist while keeping pytest
      green.
- [x] Normalize `src/sensory/organs/dimensions/order_flow.py`, capture manual
      cleanups (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/pattern_engine.py`, capture
      manual cleanups (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/patterns.py`, capture manual
      cleanups (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/regime_detection.py`, capture
      manual cleanups (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/sensory_signal.py`, capture manual
      cleanups (none), and extend the allowlist while keeping pytest green.
- [x] Normalize `src/sensory/organs/dimensions/economic_analysis.py`,
      `src/sensory/organs/dimensions/how_organ.py`,
      `src/sensory/organs/dimensions/indicators.py`,
      `src/sensory/organs/dimensions/macro_intelligence.py`, and
      `src/sensory/organs/dimensions/temporal_system.py`, expand the allowlist,
      and rerun pytest to confirm the slice stays green.
- [x] Normalize `src/sensory/organs/dimensions/__init__.py`,
      `src/sensory/organs/dimensions/utils.py`,
      `src/sensory/organs/dimensions/what_organ.py`,
      `src/sensory/organs/dimensions/when_organ.py`, and
      `src/sensory/organs/dimensions/why_organ.py`; Stage 3 is now fully enforced
      via the allowlist with pytest remaining green.

**Next**

- [x] Update the rollout plan with Stage 4 sequencing notes as each package
      lands, keeping owners and calendar slots current.
- [x] Continue sequencing follow-up directories so the allowlist shrinks after
      each slice lands. Completed by enrolling `src/data_foundation/config/` and
      staging the data foundation ingest/persist slices next.
- [ ] Dry-run `src/data_foundation/ingest/` and `src/data_foundation/persist/`
      to confirm they are formatter-clean before expanding the allowlist again.

**Immediate actions**

- Broadcast the Stage 4 expansion that now enforces the entire `src/sensory/`
  tree and the newly enrolled `src/data_foundation/config/` slice while
  confirming `scripts/check_formatter_allowlist.py` stays green after each
  update.
- Prepare the `src/data_foundation/ingest/` and `src/data_foundation/persist/`
  slices for formatting, documenting any manual adjustments and lining up
  reviewers before running `ruff format`.
- Land each Stage 4 package with a paired allowlist update, pytest run, and
  roadmap/CI snapshot refresh so the guardrail covers newly formatted code
  without masking regressions.

**Later**

- [ ] Sequence remaining directories by churn and coupling, noting any directories
      that require pairing or joint review.
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

**Now**

- [x] Break the baseline hotspots into discrete regression tickets with scoped
      inputs, expected outputs, and ownership. Captured in
      [`docs/status/regression_backlog.md`](status/regression_backlog.md) with
      ticket stubs for metrics, FIX, trading, and sensory coverage.
- [x] Prioritize the FIX execution suite so mocked integrations remain predictable
      while real endpoints are still pending. Added deterministic failure-path
      coverage in `tests/current/test_fix_manager_failures.py` and linked follow-up
      tickets in the regression backlog.

**Next**

- [x] Land tests that exercise position lifecycle accounting and reconciliation in
      `src/trading/models/position.py`.
- [x] Cover risk guardrails with both threshold-hitting and recovery scenarios.
- [x] Harden data foundation config loaders with deterministic regression tests
      so YAML overrides remain trustworthy.
- [ ] Introduce orchestration smoke tests that wire the event bus, adapters, and
      optional modules to catch configuration drift.

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

**Now**

- [x] Decide on the alert delivery channel and document the rationale and owners in
      the observability plan (GitHub issue automation is live; Slack/webhook mirroring
      still pending).
- [ ] Prepare a test failure to validate end-to-end notifications before relying
      on the channel.

**Next**

- [ ] Publish or update the CI health dashboard or README section with formatter
      and coverage telemetry.
- [x] Add pytest flake metadata collection or similar artifacts to the CI run.
- [ ] Backfill historical flakes (where possible) and link the JSON artifact in the
      observability plan for quick access.

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

**Now**

- [ ] Triage each outstanding audit entry with delete, refactor, or retain
      decisions and capture follow-up tickets.
- [ ] Remove high-confidence dead paths and run targeted tests to confirm behavior.

**Next**

- [ ] Decompose remaining high-fanin modules (for example, dense helpers inside
      `src/core`) into smaller components with clear interfaces.
- [ ] Align documentation and examples with the new module layout.

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
