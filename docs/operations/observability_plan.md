# Observability and Alerting Plan

With the Kilocode bridge retired, the modernization roadmap called for a
lightweight alternative that still surfaces failures quickly.  The plan below
outlines the guardrails that are now in place and the follow-up automation we
can layer on without introducing third-party dependencies.

## Current capabilities

* **CI failure alerts** – `.github/workflows/ci-failure-alerts.yml` opens (or
  appends to) a `CI failure alerts` issue whenever the main pipeline finishes in
  a failing, cancelled, or timed-out state. The workflow also auto-closes the
  issue after the next successful run and now mirrors the alert to Slack when a
  `SLACK_CI_WEBHOOK` secret is configured, keeping the issue and chat surfaces
  in sync.
* **Slack mirroring** – The same workflow's `notify-slack` job posts a concise
  message with run metadata and links back to the alert issue so responders can
  jump between GitHub and the `#ci-alerts` channel without losing context. The
  step no-ops when the webhook secret is absent so forks do not fail builds.
* **CI step summaries** – The `tests` job in `.github/workflows/ci.yml` tees
  pytest output into `pytest.log`, appends the last chunk to the GitHub Step
  Summary, and uploads the full log as an artifact even on failure. This
  preserves visibility without the fragile Kilocode relay.
* **Pytest flake telemetry** – Each run writes `tests/.telemetry/flake_runs.json`
  (configurable via `PYTEST_FLAKE_LOG`/`--flake-log-file`) with failure metadata
  so flakes can be trended over time.
* **Telemetry summary CLI** – `tools/telemetry/summarize_flakes.py` converts the
  flake log into a human-readable digest for status updates and retros.
* **Coverage & formatter telemetry** – `tools/telemetry/update_ci_metrics.py`
  ingests coverage XML and records the active formatter mode to append trend entries to
  `tests/.telemetry/ci_metrics.json`, giving CI dashboards a machine-readable
  feed for coverage percentages and formatter enforcement with pytest validating the
  JSON contract.【F:tools/telemetry/update_ci_metrics.py†L1-L120】【F:tools/telemetry/ci_metrics.py†L1-L164】【F:tests/tools/test_ci_metrics.py†L1-L160】【F:tests/.telemetry/ci_metrics.json†L1-L4】
* **Operational snapshot export** – `tools/telemetry/export_operational_snapshots.py`
  builds the professional runtime, extracts high-value operational blocks (professional
  readiness, security, incident response, and system validation), and emits a JSON
  payload suitable for Grafana/DataDog ingestion so dashboards can poll one file instead
  of scraping Markdown summaries. pytest covers success, missing-section failures, and the
  `--allow-missing` escape hatch.【F:tools/telemetry/export_operational_snapshots.py†L1-L143】【F:tests/tools/test_operational_export.py†L1-L86】
* **Data backbone snapshot export** – `tools/telemetry/export_data_backbone_snapshots.py`
  mirrors the operational exporter but focuses on data backbone posture (readiness,
  validation, retention, ingest trends/scheduler state, and Kafka readiness). Dashboards
  can poll a single JSON payload instead of scraping Markdown, and pytest exercises the
  success path plus the missing-section guardrails so the CLI stays regression-safe.【F:tools/telemetry/export_data_backbone_snapshots.py†L1-L147】【F:tests/tools/test_data_backbone_export.py†L1-L74】
* **Risk & compliance snapshot export** – `tools/telemetry/export_risk_compliance_snapshots.py`
  bundles risk policy, execution readiness, compliance readiness/workflow snapshots, and Timescale
  journal statistics into a governance-ready JSON feed, letting reviewers pull evidence without manual
  SQL. pytest covers the CLI flow while journal tests exercise the aggregation helpers that power the export.【F:tools/telemetry/export_risk_compliance_snapshots.py†L1-L308】【F:tests/tools/test_risk_compliance_export.py†L1-L113】【F:src/data_foundation/persist/timescale.py†L1211-L2163】【F:tests/data_foundation/test_timescale_compliance_journal.py†L57-L206】【F:tests/data_foundation/test_timescale_execution_journal.py†L86-L123】
* **Health snapshot** – [`docs/status/ci_health.md`](../status/ci_health.md)
  tracks the latest pipeline status, coverage baseline, formatter rollout
  progress, and where to look first when jobs fail.
* **Ingest trends telemetry** – `evaluate_ingest_trends` converts Timescale ingest
  journal history into `telemetry.ingest.trends` snapshots and the runtime summary
  now stores the latest payload so operators can spot row-count drops or
  freshness regressions before health checks fail.【F:src/operations/ingest_trends.py†L1-L240】【F:src/runtime/runtime_builder.py†L900-L1090】【F:src/runtime/predator_app.py†L680-L735】【F:tests/operations/test_ingest_trends.py†L1-L90】【F:tests/runtime/test_professional_app_timescale.py†L240-L330】
* **Data retention telemetry** – `evaluate_data_retention` inspects Timescale
  daily, intraday, and macro tables, publishes `telemetry.data_backbone.retention`,
  and records the markdown snapshot inside the professional runtime so operators
  can confirm historical coverage alongside other backbone feeds.【F:src/operations/retention.py†L1-L192】【F:src/runtime/runtime_builder.py†L1080-L1210】【F:src/runtime/predator_app.py†L150-L360】【F:tests/operations/test_data_retention.py†L1-L118】【F:tests/runtime/test_runtime_builder.py†L200-L360】【F:tests/runtime/test_professional_app_timescale.py†L600-L720】
* **Baseline hygiene reports** – `docs/reports/ci_baseline_report.md` and
  `docs/reports/dead_code_audit.md` capture periodic snapshots of pipeline
  health so regressions are easy to spot.
* **Runtime logging** – `configure_structured_logging` enables JSON logs with
  runtime/tier metadata when `RUNTIME_LOG_STRUCTURED` extras are provided,
  honouring optional `RUNTIME_LOG_LEVEL` and `RUNTIME_LOG_CONTEXT` overrides so
  operators can tag deployments without code changes. Tests cover the formatter
  contract and ensure the runtime builder wires the extras into the handler
  configuration.【F:src/observability/logging.py†L1-L122】【F:src/runtime/runtime_builder.py†L1838-L1877】【F:tests/observability/test_logging.py†L1-L74】【F:tests/runtime/test_runtime_builder.py†L600-L676】
* **Execution readiness telemetry** – `evaluate_execution_readiness` publishes
  `telemetry.operational.execution` with fill rates, rejection trends, latency,
  and drop-copy metrics, and the professional runtime records the markdown
  snapshot so desks can audit execution posture alongside ingest, risk, and
  compliance feeds.【F:src/operations/execution.py†L1-L430】【F:src/runtime/runtime_builder.py†L1700-L1840】【F:src/runtime/predator_app.py†L200-L360】【F:tests/operations/test_execution.py†L1-L110】【F:tests/runtime/test_professional_app_timescale.py†L360-L420】
* **Incident response telemetry** – `evaluate_incident_response` rolls runbook
  coverage, responder rosters, training cadence, and postmortem backlog data
  into `telemetry.operational.incident_response`, publishes the snapshot on the
  event bus, and records the Markdown summary in professional runtime reports so
  operators can confirm drill readiness alongside security and cache feeds.【F:src/operations/incident_response.py†L1-L233】【F:src/runtime/runtime_builder.py†L2160-L2215】【F:src/runtime/predator_app.py†L260-L360】【F:tests/operations/test_incident_response.py†L1-L108】【F:tests/runtime/test_runtime_builder.py†L360-L520】【F:tests/runtime/test_professional_app_timescale.py†L520-L600】
* **FIX pilot telemetry** – `FixIntegrationPilot` supervises FIX sessions and
  background workers while `FixDropcopyReconciler` feeds drop-copy executions
  into pilot snapshots; `evaluate_fix_pilot` emits
  `telemetry.execution.fix_pilot` with queue metrics, order posture, drop-copy
  reconciliation, and compliance coverage so FIX deployments remain observable
  from the runtime summary.【F:src/runtime/fix_pilot.py†L1-L210】【F:src/runtime/fix_dropcopy.py†L1-L228】【F:src/operations/fix_pilot.py†L1-L240】【F:src/runtime/runtime_builder.py†L2040-L2130】【F:tests/runtime/test_fix_pilot.py†L1-L190】【F:tests/runtime/test_fix_dropcopy.py†L1-L60】
* **Execution readiness journal** – `TimescaleExecutionJournal` mirrors each
  execution snapshot to Timescale (`telemetry.execution_snapshots`) and the
  professional runtime summary exposes recent and latest entries via the
  `execution_journal` block so operators inherit an auditable execution history
  without leaving the runtime interface.【F:src/data_foundation/persist/timescale.py†L900-L1290】【F:src/runtime/predator_app.py†L200-L760】【F:tests/data_foundation/test_timescale_execution_journal.py†L1-L91】【F:tests/runtime/test_professional_app_timescale.py†L400-L460】
* **Sensory drift telemetry** – `evaluate_sensory_drift` analyses the fused
  sensory audit trail, publishes `telemetry.sensory.drift`, and records the
  latest snapshot in the professional runtime summary so operators can monitor
  WHY/HOW deltas alongside ingest telemetry.【F:src/operations/sensory_drift.py†L1-L215】【F:src/runtime/runtime_builder.py†L960-L1062】【F:src/runtime/predator_app.py†L320-L360】【F:tests/operations/test_sensory_drift.py†L1-L64】【F:tests/runtime/test_professional_app_timescale.py†L720-L780】
* **Evolution experiment telemetry** – `evaluate_evolution_experiments`
  aggregates paper-trading experiment events and ROI posture into
  `telemetry.evolution.experiments`; the trading manager records experiment
  events, the runtime builder publishes the snapshot, and professional summaries
  expose the markdown block for operators.【F:src/operations/evolution_experiments.py†L1-L248】【F:src/trading/trading_manager.py†L1-L360】【F:src/runtime/runtime_builder.py†L2236-L2268】【F:src/runtime/predator_app.py†L968-L986】【F:tests/operations/test_evolution_experiments.py†L1-L114】【F:tests/trading/test_trading_manager_execution.py†L1-L140】【F:tests/runtime/test_professional_app_timescale.py†L200-L271】
* **Evolution tuning telemetry** – `evaluate_evolution_tuning` combines experiment
  and strategy snapshots into `telemetry.evolution.tuning`, logs Markdown summaries,
  publishes the feed on the runtime bus, and records the latest block in the
  professional runtime so operators can review automated tuning guidance alongside
  experiment metrics.【F:src/operations/evolution_tuning.py†L1-L443】【F:src/runtime/runtime_builder.py†L2566-L2649】【F:src/runtime/predator_app.py†L229-L515】【F:src/runtime/predator_app.py†L1098-L1104】【F:tests/operations/test_evolution_tuning.py†L1-L172】【F:tests/runtime/test_professional_app_timescale.py†L1298-L1338】
* **Strategy performance telemetry** – `evaluate_strategy_performance` groups
  trading-manager experiment events and ROI snapshots into
  `telemetry.strategy.performance`, logs Markdown summaries, and the professional
  runtime records the latest block so desks can track execution/rejection mix per
  strategy alongside ROI posture.【F:src/operations/strategy_performance.py†L1-L537】【F:src/runtime/runtime_builder.py†L2236-L2294】【F:src/runtime/predator_app.py†L200-L986】【F:tests/runtime/test_runtime_builder.py†L272-L523】【F:tests/runtime/test_professional_app_timescale.py†L200-L270】
* **Event bus health telemetry** – `evaluate_event_bus_health` converts event
  bus statistics into `telemetry.event_bus.health` snapshots, publishes the
  payload after ingest runs, and stores the markdown summary in the
  professional runtime so operators can audit queue depth, drops, and handler
  errors alongside other readiness feeds.【F:src/operations/event_bus_health.py†L1-L220】【F:src/runtime/runtime_builder.py†L1872-L1916】【F:src/runtime/predator_app.py†L200-L380】【F:tests/operations/test_event_bus_health.py†L1-L82】【F:tests/runtime/test_professional_app_timescale.py†L380-L452】
* **Kafka readiness telemetry** – `evaluate_kafka_readiness` merges connection
  settings, topic provisioning summaries, publisher availability, and lag
  snapshots into `telemetry.kafka.readiness`, with the runtime builder
  publishing the snapshot and the professional app storing the markdown so
  streaming posture appears beside ingest, cache, and failover feeds.【F:src/operations/kafka_readiness.py†L1-L213】【F:src/runtime/runtime_builder.py†L600-L930】【F:src/runtime/predator_app.py†L200-L400】【F:tests/operations/test_kafka_readiness.py†L1-L97】【F:tests/runtime/test_runtime_builder.py†L665-L782】【F:tests/runtime/test_professional_app_timescale.py†L1116-L1160】
* **System validation telemetry** – `evaluate_system_validation` parses the
  roadmap-aligned validation report, publishes `telemetry.operational.system_validation`,
  and records the markdown snapshot in the professional runtime so operators can
  confirm architecture checks alongside ingest, security, and readiness feeds.【F:src/operations/system_validation.py†L1-L230】【F:src/runtime/runtime_builder.py†L1905-L1950】【F:src/runtime/predator_app.py†L120-L360】【F:tests/operations/test_system_validation.py†L1-L120】【F:tests/runtime/test_professional_app_timescale.py†L400-L455】【F:tests/runtime/test_runtime_builder.py†L200-L360】
* **Data backbone runbooks** – The [Redis cache outage](runbooks/redis_cache_outage.md)
  and [Kafka ingest offset recovery](runbooks/kafka_ingest_offset_recovery.md)
  playbooks translate the cache health and Kafka lag telemetry into on-call
  procedures so responders can stabilise institutional ingest before escalating.【F:docs/operations/runbooks/redis_cache_outage.md†L1-L60】【F:docs/operations/runbooks/kafka_ingest_offset_recovery.md†L1-L66】【F:src/operations/cache_health.py†L60-L199】【F:src/data_foundation/streaming/kafka_stream.py†L1682-L1919】

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
  past the business day, page the engineering manager and pin the automated
  update from the `notify-slack` job in the `#ci-alerts` Slack channel with
  remediation notes.

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
- [x] **Deploy Slack webhook relay** – Extend `.github/workflows/ci-failure-
      alerts.yml` with the `notify-slack` job gated by `SLACK_CI_WEBHOOK`,
      mirroring the alert issue into `#ci-alerts` with actor, branch, and run
      links.
- [x] **Runtime healthchecks** – The runtime now exposes an aiohttp-backed
      `/health` endpoint that grades FIX connectivity, market-data freshness, and
      telemetry exporters using `RuntimeHealthServer`; the builder starts the
      server automatically and stores the latest snapshot in the professional
      runtime summary for operators.【F:src/runtime/healthcheck.py†L1-L258】【F:src/runtime/runtime_builder.py†L1816-L1863】

## Slack/webhook rollout plan

| Step | Owner | Details |
| --- | --- | --- |
| Provision webhook | Platform (M. Rivera) | Create an incoming webhook in the shared `#ci-alerts` Slack channel and store the URL in the existing GitHub Actions secret `SLACK_CI_WEBHOOK`. |
| Deploy relay | Reliability (C. Gomez) | Extend `.github/workflows/ci-failure-alerts.yml` with a job that posts issue updates to the webhook. Re-use the existing JSON payload so the Slack message links back to the issue and run. **Status:** Completed via the `notify-slack` job; verified during the September 2025 alert drill. |
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
* OpenTelemetry tracing now instruments event bus publishes and handler fan-out
  when `OTEL_ENABLED` extras are set. Configure exporters with the
  `OTEL_EXPORTER_OTLP_*` settings; the runtime wires the tracer into
  `AsyncEventBus` so spans carry queue depth and dispatch lag metadata for
  every event, and reuses the same tracer to wrap runtime startup/shutdown
  hooks, workload execution, and Timescale ingest orchestration so traces show
  plan evaluation, fallback drills, and ingest success. 【F:src/observability/tracing.py†L1-L244】【F:src/runtime/predator_app.py†L1560-L1610】【F:src/runtime/runtime_builder.py†L250-L408】【F:src/runtime/runtime_builder.py†L1996-L2320】【F:tests/core/test_event_bus_tracing.py†L1-L118】【F:tests/runtime/test_runtime_tracing.py†L1-L134】
* Adopt GitHub's dependency review and code scanning alerts once the formatting
  backlog is addressed to avoid noisy signal during active cleanup.

Owners should revisit this plan quarterly and adjust the roadmap as new
observability gaps surface.

