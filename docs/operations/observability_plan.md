# Observability and Alerting Plan

With the Kilocode bridge retired, the modernization roadmap called for a
lightweight alternative that still surfaces failures quickly.  The plan below
outlines the guardrails that are now in place and the follow-up automation we
can layer on without introducing third-party dependencies.

## Current capabilities

 codex/assess-technical-debt-in-codebase
* **CI failure alerts** – `.github/workflows/ci-failure-alerts.yml` opens (or
  appends to) a `CI failure alerts` issue whenever the main pipeline finishes in
  a failing, cancelled, or timed-out state. The workflow also auto-closes the
  issue after the next successful run so the backlog reflects only outstanding
  problems.
* **CI step summaries** – The `tests` job in `.github/workflows/ci.yml` tees
  pytest output into `pytest.log`, appends the last chunk to the GitHub Step
  Summary, and uploads the full log as an artifact even on failure. This
  preserves visibility without the fragile Kilocode relay.
* **Health snapshot** – [`docs/status/ci_health.md`](../status/ci_health.md)
  tracks the latest pipeline status, coverage baseline, formatter rollout
  progress, and where to look first when jobs fail.

* **CI step summaries** – The `tests` job in `.github/workflows/ci.yml` now
  tees pytest output into `pytest.log`, appends the last chunk to the GitHub
  Step Summary, and uploads the full log as an artifact even on failure.  This
  preserves visibility without the fragile Kilocode relay.
 main
* **Baseline hygiene reports** – `docs/reports/ci_baseline_report.md` and
  `docs/reports/dead_code_audit.md` capture periodic snapshots of pipeline
  health so regressions are easy to spot.
* **Runtime logging** – The platform leans on Python's structured logging
 codex/assess-technical-debt-in-codebase
  (module loggers + UTC timestamps). Configuration and policy failures abort
  startup with actionable errors.

## On-call expectations

* **Primary rotation** – The trading-platform team maintains a weekly rotation.
  The current order is documented in the shared on-call calendar; each handoff
  occurs during Monday stand-up.
* **Alert intake** – When the CI failure issue opens, the on-call engineer must
  acknowledge it within one business hour and either drive the fix or pair the
  change with the triggering contributor.
* **Resolution** – Once the blocking run passes, close the alert issue. Add a
  comment summarizing the root cause and any follow-up tickets that were filed.
* **Escalation path** – If the failure blocks production hotfixes or persists
  past the business day, page the engineering manager and post in the
  `#ci-alerts` Slack channel (mirrors the GitHub issue feed).

## Immediate next steps

1. **Test flake dashboard** – Extend the pytest log artifact uploader to retain
   failure metadata (module, test name, error message) in a small JSON file.
   Periodically aggregate this into a flake leaderboard.
2. **Runtime healthchecks** – For long-running deployments, expose a `/health`
   endpoint that checks FIX connectivity, market-data freshness, and telemetry
   exporter status.
3. **Optional webhook** – If the Slack mirror proves insufficient, explore
   GitHub's native notification integrations for paging during off-hours.

  (module loggers + UTC timestamps).  Configuration and policy failures abort
  startup with actionable errors.

## Immediate next steps

1. **Slack or email webhook (optional)** – If stakeholders need push
   notifications, wire GitHub's native workflow notification integration to a
   shared channel.  This avoids custom code while restoring proactive alerts.
2. **Test flake dashboard** – Extend the pytest log artifact uploader to retain
   failure metadata (module, test name, error message) in a small JSON file.
   Periodically aggregate this into a flake leaderboard.
3. **Runtime healthchecks** – For long-running deployments, expose a `/health`
   endpoint that checks FIX connectivity, market-data freshness, and telemetry
   exporter status.
 main

## Long-term instrumentation ideas

* Promote Prometheus metrics beyond the existing counters so we can alert on
  order execution latency, event bus backlog, and ingestion freshness.
* Introduce OpenTelemetry tracing when strategic refactors land; the event bus
  provides a natural place to propagate trace context.
* Adopt GitHub's dependency review and code scanning alerts once the formatting
  backlog is addressed to avoid noisy signal during active cleanup.

Owners should revisit this plan quarterly and adjust the roadmap as new
observability gaps surface.

