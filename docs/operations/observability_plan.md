# Observability and Alerting Plan

With the Kilocode bridge retired, the modernization roadmap called for a
lightweight alternative that still surfaces failures quickly.  The plan below
outlines the guardrails that are now in place and the follow-up automation we
can layer on without introducing third-party dependencies.

## Current capabilities

* **CI failure alerts** – `.github/workflows/ci-failure-alerts.yml` opens (or
  appends to) a `CI failure alerts` issue whenever the main pipeline finishes in
  a failing, cancelled, or timed-out state. The workflow also auto-closes the
  issue after the next successful run. A Slack/webhook mirror is planned but not
  yet wired up.
* **CI step summaries** – The `tests` job in `.github/workflows/ci.yml` tees
  pytest output into `pytest.log`, appends the last chunk to the GitHub Step
  Summary, and uploads the full log as an artifact even on failure. This
  preserves visibility without the fragile Kilocode relay.
* **Pytest flake telemetry** – Each run writes `tests/.telemetry/flake_runs.json`
  (configurable via `PYTEST_FLAKE_LOG`/`--flake-log-file`) with failure metadata
  so flakes can be trended over time.
* **Health snapshot** – [`docs/status/ci_health.md`](../status/ci_health.md)
  tracks the latest pipeline status, coverage baseline, formatter rollout
  progress, and where to look first when jobs fail.
* **Baseline hygiene reports** – `docs/reports/ci_baseline_report.md` and
  `docs/reports/dead_code_audit.md` capture periodic snapshots of pipeline
  health so regressions are easy to spot.
* **Runtime logging** – The platform leans on Python's structured logging
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
  past the business day, page the engineering manager and post a manual update
  in the `#ci-alerts` Slack channel (the automation does not mirror there yet).

## Immediate next steps

1. **Run a forced-failure drill** – Trigger a controlled CI failure to confirm
   the GitHub issue alert and on-call acknowledgment path behave as expected.
2. **Wire Slack/webhook mirroring** – Land the notification relay that will copy
   issue updates into `#ci-alerts` (or the chosen channel).
3. **Publish flake-reading guidance** – Document how to consume
   `flake_runs.json`, aggregate events, and feed insights into retrospectives.
4. **Runtime healthchecks** – For long-running deployments, expose a `/health`
   endpoint that checks FIX connectivity, market-data freshness, and telemetry
   exporter status.

## Long-term instrumentation ideas

* Promote Prometheus metrics beyond the existing counters so we can alert on
  order execution latency, event bus backlog, and ingestion freshness.
* Introduce OpenTelemetry tracing when strategic refactors land; the event bus
  provides a natural place to propagate trace context.
* Adopt GitHub's dependency review and code scanning alerts once the formatting
  backlog is addressed to avoid noisy signal during active cleanup.

Owners should revisit this plan quarterly and adjust the roadmap as new
observability gaps surface.

