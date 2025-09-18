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
* **Telemetry summary CLI** – `tools/telemetry/summarize_flakes.py` converts the
  flake log into a human-readable digest for status updates and retros.
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

- [x] **Run a forced-failure drill** – Dispatch the `CI` workflow with
      `alert_drill=true` (via `workflow_dispatch`) to intentionally fail the
      tests job. The drill on 2025-09-22 confirmed that the
      `CI failure alerts` issue opened automatically and auto-closed after the
      rerun succeeded.
- [x] **Document Slack/webhook mirroring** – See the rollout plan below for
      owners, secrets, and the validation checklist.
- [x] **Publish flake-reading guidance** – See the new "Reading flake telemetry"
      section below for how to interpret `tests/.telemetry/flake_runs.json` and
      correlate entries with GitHub runs.
- [ ] **Runtime healthchecks** – For long-running deployments, expose a
      `/health` endpoint that checks FIX connectivity, market-data freshness, and
      telemetry exporter status.

## Slack/webhook rollout plan

| Step | Owner | Details |
| --- | --- | --- |
| Provision webhook | Platform (M. Rivera) | Create an incoming webhook in the shared `#ci-alerts` Slack channel and store the URL in the existing GitHub Actions secret `SLACK_CI_WEBHOOK`. |
| Deploy relay | Reliability (C. Gomez) | Extend `.github/workflows/ci-failure-alerts.yml` with a job that posts issue updates to the webhook. Re-use the existing JSON payload so the Slack message links back to the issue and run. |
| Validation drill | Trading (J. McKay) | Trigger `workflow_dispatch` with `alert_drill=true` after the relay ships. Confirm the Slack message arrives with run context, then re-run CI to close the loop. |
| Documentation refresh | Observability (L. Chen) | Update this plan and `docs/status/ci_health.md` with the relay go-live date and add screenshots to the team wiki. |

The relay job should run only on failure/cancelled outcomes to avoid noise. The
message body should include the run URL, branch name, triggering actor, and a
link back to the `CI failure alerts` issue for traceability.

## Alert drill cadence

| Quarter | Target week | Notes |
| --- | --- | --- |
| Q4 2025 | Week of 21 Oct | Pair with formatter rollout freeze to avoid overlapping recoveries. |
| Q1 2026 | Week of 20 Jan | Rotate the drill owner to Reliability to keep knowledge fresh. |
| Q2 2026 | Week of 21 Apr | Combine with coverage snapshot refresh and capture lessons in the retro. |

## Alert drills

Use the manual trigger on the `CI` workflow to exercise the alerting pipeline:

1. Open the **CI** workflow in GitHub Actions and select **Run workflow**.
2. Set the **alert_drill** input to `true` and supply a short reason in the run
   summary.
3. Let the workflow fail; the `.github/workflows/ci-failure-alerts.yml` run will
   append context to the `CI failure alerts` issue.
4. Re-run the workflow with **alert_drill** left at the default `false` value to
   close the issue and verify the recovery path.

Document the drill outcome in the on-call handoff notes so the next engineer
knows the cadence and last validation date.

## Reading flake telemetry

The pytest plugin writes a JSON payload to `tests/.telemetry/flake_runs.json`
after each session. The backfilled sample now in the repository mirrors two
recent CI failures and successful recovery runs.

* `meta` contains timestamps, Python/runtime information, exit status, and the
  CI run identifiers that produced the telemetry.
* Each entry in `events` records a failing nodeid, duration, outcome, and a
  clipped failure trace for quick triage.
* The optional `history` array links telemetry entries to GitHub Actions URLs so
  engineers can jump directly to the failing workflow.

When a new failure occurs, download the `pytest-log-*` artifact for full
context, compare against the corresponding telemetry entry, and capture any new
flakes or fixes in the team retrospective notes.

## Long-term instrumentation ideas

* Promote Prometheus metrics beyond the existing counters so we can alert on
  order execution latency, event bus backlog, and ingestion freshness.
* Introduce OpenTelemetry tracing when strategic refactors land; the event bus
  provides a natural place to propagate trace context.
* Adopt GitHub's dependency review and code scanning alerts once the formatting
  backlog is addressed to avoid noisy signal during active cleanup.

Owners should revisit this plan quarterly and adjust the roadmap as new
observability gaps surface.

