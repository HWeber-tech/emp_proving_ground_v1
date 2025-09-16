# Modernization Roadmap

This roadmap captures the next steps for reducing technical debt and restoring confidence in the EMP Professional Predator codebase and delivery pipelines. Items are grouped by phase so the team can focus on one layer of stability at a time while keeping downstream work unblocked.

For a detailed audit of the current state, pair this plan with the living [Technical Debt Assessment](technical_debt_assessment.md).

## How to work the roadmap
- **Triage → ticket → deliver**: Start every phase by translating the bullet points below into discrete tickets (1–2 day efforts). Each ticket should have a crisp Definition of Done and a designated owner so work moves in parallel without surprise dependencies.
- **Keep a rolling kanban**: Maintain a three-column board (Now / Next / Later) that reflects the current slice of roadmap work. As soon as a ticket lands in "Later" it can be groomed and sized so the next engineer can pick it up without needing a fresh planning meeting.
- **Time-box discovery spikes**: When a task needs investigation (for example, understanding why CI fails after the conflict is resolved) reserve explicit spike tickets with 4–8 hour caps and a required summary comment in the ticket before it is closed.
- **Close the feedback loop**: End each week with a 15-minute review to capture status, blockers, and any adjustments to sequencing. Update this document with lessons learned so it stays authoritative.

### Initial kickoff backlog (completed)
- [x] **Confirm deprecated integrations are inert** – verify secrets/labels for Kilocode are removed and document findings.
- [x] **Resolve `ci.yml` merge markers** – reproduce the conflict locally, apply the correct sections, and push a validating branch build.
- [x] **Re-run CI end-to-end** – execute the fixed workflow locally or via a draft PR and log which jobs fail plus top errors.
- [x] **Spin up the tracking board** – set up the shared project board (GitHub Projects, Linear, or Jira) populated with the initial tickets.

### Next-iteration kickoff backlog
Spin up these tickets to begin the next modernization wave:
 codex/assess-technical-debt-in-codebase
- [ ] **Stage 0 formatter rollout** – Format every file under `tests/current/`, expand `config/formatter/ruff_format_allowlist.txt`, and confirm pytest still passes so CI begins enforcing the first slice.
- [ ] **Prep Stage 1 formatting** – Dry-run formatting for `src/system/` and `src/core/configuration.py`, document any manual edits required, and line up the allowlist expansion once Stage 0 lands.
- [ ] **Regression gap tickets** – Break down the coverage hotspots from [CI Baseline – 2025-09-16](ci_baseline_report.md)—notably `src/operational/metrics.py`, `src/trading/models/position.py`, the `src/data_foundation/config/` modules, and `src/sensory/dimensions/why/yield_signal.py`—into reviewable test additions.
- [ ] **Dead-code audit follow-up** – Work through the remaining findings in [Dead code audit – 2025-09-16](reports/dead_code_audit.md), filing delete/refactor tickets and annotating intentional dynamic hooks so future scans stay actionable.
- [ ] **CI flake telemetry** – Implement the pytest failure metadata artifact described in `docs/operations/observability_plan.md` and publish a lightweight dashboard or report for recurring flakes.

- [ ] **Publish the formatter rollout plan** – agree on the directory-by-directory schedule for running `ruff format` so changesets stay reviewable.
- [ ] **Prioritize coverage hotspots** – translate the red zones from `docs/ci_baseline_report.md` into scoped regression tickets for trading, risk, and orchestration modules.
- [ ] **Triage dead-code findings** – classify each entry in [Dead code audit – 2025-09-16](reports/dead_code_audit.md) as delete, refactor, or keep (with justification) and turn outcomes into follow-up issues.
- [ ] **Select the alerting channel** – choose between GitHub notifications, Slack, or email digests for CI failures and capture the decision in `docs/operations/observability_plan.md`.
 main

## Phase 0 – Immediate hygiene (Week 1)
- [x] **Retire redundant automation**: Remove the deprecated Kilocode CI Bridge workflow and confirm no remaining secrets or labels reference the integration.
- [x] **Unblock CI parsing**: Resolve the merge-conflict markers in `.github/workflows/ci.yml` so GitHub can execute the workflow again.
- [x] **Baseline pipeline health**: Run the fixed CI workflow end-to-end (policy, lint, mypy, pytest) and capture current failure modes to inform later phases. Results are logged in [CI Baseline – 2025-09-16](ci_baseline_report.md).

## Phase 1 – Policy enforcement consolidation (Week 2)
- [x] **Single source of truth for OpenAPI/cTrader ban**: Collapse the duplicated policy checks (`policy` job inside `ci.yml` and `policy-openapi-block.yml`) behind a reusable workflow so the guardrail is defined once.
- [x] **Configuration alignment**: Audit `config.yaml` and documentation to remove or clearly flag cTrader/OpenAPI placeholders so runtime defaults match the enforced FIX-only posture.
  - [x] Default `config.yaml` to FIX simulator settings with IC Markets credentials placeholders.
  - [x] Introduced an integration policy and legacy disclaimers so documentation mirrors the FIX-only stance.

## Phase 2 – Dependency and environment hygiene (Weeks 3–4)
- [x] **Rationalize requirements**: Promote `requirements/base.txt` as the canonical runtime manifest and layer development tooling through `requirements/dev.txt`, updating automation and docs accordingly.
- [x] **Pin critical tooling**: Align versions for linting, typing, and testing tools across local dev and CI to eliminate "works on my machine" discrepancies.
  - [x] Lock mypy, Ruff, Black, pytest, and supporting stubs in `requirements/dev.txt` so automation and local shells resolve the same toolchain.
- [x] **Scientific stack verification**: Document the minimum supported numpy/pandas/scipy versions, wire them into the setup guide, and expose a `python -m src.system.requirements_check` CLI so deploys can fail fast when the stack drifts.

## Phase 3 – Repository cleanup (Weeks 4–5)
- [x] **Prune stray artifacts**: Delete `.orig` backups, `changed_files_*.txt`, and other scratch files from version control.
- [x] **Establish guardrails**: Add `.gitignore` updates or pre-commit hooks to prevent regenerated artifacts from re-entering the repo.
- [x] **Audit for dead code**: Captured the first pass in [Dead code audit – 2025-09-16](reports/dead_code_audit.md) generated via `scripts/audit_dead_code.py` to seed follow-up cleanup tickets.

## Phase 4 – Test coverage and observability (Weeks 6–7)
- [x] **Strengthen regression nets**: Expand pytest coverage around the event bus, trading pipelines, and configuration loaders to reduce reliance on manual smoke tests.
  - [x] Added regression coverage for `SystemConfig.with_updated` conversions and the scientific stack guard.
  - [x] Added FIX parity checker tests covering order/position mismatch accounting and emitted telemetry gauges.
  - [x] Added `core.configuration` regression tests that exercise environment overrides, nested accessors, YAML round-tripping, and module-level global state updates.
- [x] **Type safety focus areas**: Use the nightly mypy reports to target high-churn modules and drive them toward clean type annotations.
  - [x] Refactored the legacy `Configuration` helper to use typed mapping access with safe nested mutation semantics backed by new tests.
- [x] **Operational insights**: Evaluate lightweight telemetry or structured logging so failures surface without external services like Kilocode.
  - [x] CI now appends pytest tails to the GitHub Step Summary and uploads the full log as an artifact, with the follow-up alerting plan captured in `docs/operations/observability_plan.md`.

## Phase 5 – Strategic refactors (Weeks 8+)
- [x] **Subsystem decomposition**: Prioritize modules with the highest coupling (e.g., `core` ↔ `trading`) for refactor spikes once the pipeline is green.
  - [x] Updated `docs/architecture/refactor_roadmap.md` and the refreshed architecture overview to anchor upcoming decomposition work on the validated layering contract.
- [x] **Documentation refresh**: Produce system architecture and runbook documentation reflecting the stabilized workflows and policies.
  - [x] Replaced the stubbed architecture overview with a layered system guide linked from policy and roadmap docs.
- [x] **Plan for future automation**: Revisit alerting and failure triage needs after CI is reliable, considering native GitHub features or self-hosted tooling if necessary.
  - [x] Authored `docs/operations/observability_plan.md` describing the CI summary uploads, hygiene baselines, and incremental alerting enhancements.

## Phase 6 – Formatter normalization (Weeks 8–9)
 codex/assess-technical-debt-in-codebase
- [x] **Document the staged rollout**: Captured in [`docs/development/formatter_rollout.md`](development/formatter_rollout.md) with owners, gating tasks, and sequencing for each formatting slice.
- [x] **Automate guardrails**: Added `scripts/check_formatter_allowlist.py` + `config/formatter/ruff_format_allowlist.txt` and updated CI to enforce formatter checks only on opt-in paths.
- [x] **Update contributor guidance**: Extended `docs/development/setup.md` with formatter expectations and the workflow for expanding the allowlist as directories are normalized.

## Phase 7 – Regression depth in trading & risk (Weeks 10–11)
- [x] **Backfill trading execution tests**: Add pytest coverage for FIX execution flows (order routing, failure handling, and reconciliation) prioritized by the CI baseline's uncovered lines.
  - [x] Exercised quantity validation, missing-order cancellations, and defensive copies of active orders in `tests/current/test_fix_executor.py` to harden legacy FIX stubs.
- [x] **Exercise risk management edges**: Target `src/risk` hot spots—such as position limits and drawdown guards—with unit tests that cover both success and failure paths.
  - [x] Added `tests/current/test_risk_manager_impl.py` with async validation, Kelly sizing, limit updates, and summary aggregation scenarios exercising success paths and failure fallbacks.
- [x] **Stabilize orchestration flows**: Create integration-style tests for the event bus and orchestrator wiring so configuration errors surface before live runs.
  - [x] Added `tests/current/test_orchestration_compose.py` to verify gateway adapters invoke injected organs, adaptation services normalize payloads, and the validation bundle degrades gracefully when optional modules are missing.

## Phase 8 – Operational telemetry & alerting (Weeks 11–12)
- [x] **Implement CI alert delivery**: Added `.github/workflows/ci-failure-alerts.yml` to open (or update) a `CI failure alerts` issue when the pipeline fails, with automatic closure after the next green run.
- [x] **Surface health dashboards**: Published [`docs/status/ci_health.md`](status/ci_health.md) summarizing latest runs, coverage, formatter rollout progress, and triage entry points.
- [x] **Codify on-call expectations**: Expanded `docs/operations/observability_plan.md` with rotation duties, response-time targets, and escalation guidance tied to the automated alerts.

## Phase 9 – Dead code remediation & modular cleanup (Weeks 12+)
- [x] **Retire confirmed dead paths**: Deleted the high-confidence candidates called out in the latest `scripts/audit_dead_code.py` report—removing legacy event imports, tightening the portfolio monitor helper, and shoring up optional dependency shims—while documenting the rationale in the debt assessment.
- [x] **Decompose monolith modules**: Use the updated architecture guide to split overly coupled packages (for example, shared utilities inside `src/core`) into focused components.
  - [x] Extracted `performance_metrics.py` from the trading performance tracker so analytical helpers live outside the monitoring entry point.
  - [x] Broke the interface hub into `src/core/interfaces/` submodules (`base.py`, `ecosystem.py`, `metrics.py`, `analysis.py`) with a re-exporting `__init__` so the high-fanin Protocols no longer live in a single 350-line file.
- [x] **Schedule follow-up audits**: Re-run the dead-code audit after each cleanup batch and append findings to `docs/reports/dead_code_audit.md` to track progress over time.
  - [x] Added a scheduled GitHub Action (`dead-code-audit.yml`) that runs the vulture report weekly, on demand, or when the script changes, uploading the Markdown snapshot as an artifact for review.

- [ ] **Document the staged rollout**: Capture the agreed sequencing (by package or subsystem) for applying `ruff format` so reviewers can focus on mechanical diffs.
- [ ] **Automate guardrails**: Introduce opt-in directory allow-lists (for example, via `ruff.toml`'s `extend-exclude`) and tighten CI to block regressions once each slice is formatted.
- [ ] **Update contributor guidance**: Extend `docs/development/setup.md` with formatting expectations, including when to run `ruff format` locally and how to handle conflicts.

## Phase 7 – Regression depth in trading & risk (Weeks 10–11)
- [ ] **Backfill trading execution tests**: Add pytest coverage for FIX execution flows (order routing, failure handling, and reconciliation) prioritized by the CI baseline's uncovered lines.
- [ ] **Exercise risk management edges**: Target `src/risk` hot spots—such as position limits and drawdown guards—with unit tests that cover both success and failure paths.
- [ ] **Stabilize orchestration flows**: Create integration-style tests for the event bus and orchestrator wiring so configuration errors surface before live runs.

## Phase 8 – Operational telemetry & alerting (Weeks 11–12)
- [ ] **Implement CI alert delivery**: Stand up the chosen notification channel (GitHub issue automation, Slack webhook, or email summaries) and validate it with a test failure.
- [ ] **Surface health dashboards**: Publish a lightweight status page or README section summarizing latest CI runs, coverage, and formatter completion percentages.
- [ ] **Codify on-call expectations**: Document who triages alerts, response-time goals, and escalation steps alongside the observability plan.

## Phase 9 – Dead code remediation & modular cleanup (Weeks 12+)
- [ ] **Retire confirmed dead paths**: Delete the high-confidence candidates identified in the latest `scripts/audit_dead_code.py` report and document any intentionally retained hooks.
- [ ] **Decompose monolith modules**: Use the updated architecture guide to split overly coupled packages (for example, shared utilities inside `src/core`) into focused components.
- [ ] **Schedule follow-up audits**: Re-run the dead-code audit after each cleanup batch and append findings to `docs/reports/dead_code_audit.md` to track progress over time.
 main

### Tracking and review
- Review progress in weekly debt triage meetings.
- Keep a shared checklist linked to this roadmap so contributors can claim items and record findings.
- Revisit phases quarterly (or after each Phase 6–9 milestone) to update priorities based on new discoveries or product needs.
