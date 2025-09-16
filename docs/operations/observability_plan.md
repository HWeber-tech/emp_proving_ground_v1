# Observability and Alerting Plan

With the Kilocode bridge retired, the modernization roadmap called for a
lightweight alternative that still surfaces failures quickly.  The plan below
outlines the guardrails that are now in place and the follow-up automation we
can layer on without introducing third-party dependencies.

## Current capabilities

* **CI step summaries** – The `tests` job in `.github/workflows/ci.yml` now
  tees pytest output into `pytest.log`, appends the last chunk to the GitHub
  Step Summary, and uploads the full log as an artifact even on failure.  This
  preserves visibility without the fragile Kilocode relay.
* **Baseline hygiene reports** – `docs/reports/ci_baseline_report.md` and
  `docs/reports/dead_code_audit.md` capture periodic snapshots of pipeline
  health so regressions are easy to spot.
* **Runtime logging** – The platform leans on Python's structured logging
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

## Long-term instrumentation ideas

* Promote Prometheus metrics beyond the existing counters so we can alert on
  order execution latency, event bus backlog, and ingestion freshness.
* Introduce OpenTelemetry tracing when strategic refactors land; the event bus
  provides a natural place to propagate trace context.
* Adopt GitHub's dependency review and code scanning alerts once the formatting
  backlog is addressed to avoid noisy signal during active cleanup.

Owners should revisit this plan quarterly and adjust the roadmap as new
observability gaps surface.

