# Alignment brief – Operational readiness & resilience telemetry

## Concept promise

- The encyclopedia outlines operational security, observability, and backup
  requirements that accompany institutional deployments, including multi-layer
  monitoring and incident response.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L360-L395】【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L8841-L8918】
- The architecture overview frames orchestration/operational layers as first-class
  domains alongside trading and risk.【F:docs/architecture/overview.md†L9-L37】

## Reality snapshot

- CI baseline captures lint/type/test success but exposes weak coverage in
  operational metrics and missing alert channels; observability is brittle.【F:docs/ci_baseline_report.md†L8-L27】
- Technical debt assessments cite unsupervised async tasks, missing documentation,
  and open alerts workflow items.【F:docs/technical_debt_assessment.md†L33-L112】
- Legacy guides (OpenAPI/cTrader) persist in `docs/legacy`, signalling incomplete
  cleanup and risk of policy drift.【F:docs/legacy/README.md†L1-L12】

## Gap themes

1. **Supervised operations** – Adopt task supervision, document shutdown/restart
   drills, and validate failover behaviour.
2. **Observability** – Expand metrics, logs, and alert routing beyond CI summaries;
   integrate telemetry from ingest, sensory, risk, and compliance streams.
3. **Documentation hygiene** – Remove deprecated runbooks, align public docs with
   the FIX-only posture, and keep status pages in lockstep with reality.

## Delivery plan

### Now (0–30 days)

- Finish the task supervision rollout across runtime and operational helpers.【F:docs/technical_debt_assessment.md†L33-L56】
  - ✅ Runtime CLI orchestration and the bootstrap sensory loop now execute under
    `TaskSupervisor`, eliminating direct `asyncio.create_task` usage for
    entrypoint workflows and providing deterministic signal/timeout shutdowns.【F:src/runtime/cli.py†L206-L249】【F:src/runtime/bootstrap_runtime.py†L227-L268】
  - Progress: A dedicated runtime runner now wraps professional workloads in a
    shared `TaskSupervisor`, wiring signal handlers, optional timeouts, and
    shutdown callbacks so production launches share the same supervised lifecycle
    contract as the builder, with pytest covering normal completion and timeout
    cancellation flows.【F:src/runtime/runtime_runner.py†L1-L120】【F:main.py†L71-L125】【F:tests/runtime/test_runtime_runner.py†L1-L58】
  - Progress: Phase 3 orchestrator registers its continuous analysis and
    performance monitors with the shared supervisor, drains background tasks on
    shutdown, and ships a smoke test validating the supervised lifecycle so
    thinking pipelines inherit the same operational guardrails as runtime
    entrypoints, while new persistence fallbacks log and retry state-store writes
    so analysis snapshots degrade gracefully without silent drops.【F:src/thinking/phase3_orchestrator.py†L103-L276】【F:src/thinking/phase3_orchestrator.py†L596-L626】【F:tests/current/test_orchestration_runtime_smoke.py†L19-L102】
  - Progress: Timescale ingest scheduler now surfaces its supervised task handle, exposes `wait_until_stopped`, and the runtime builder shields shutdown cleanup while institutional ingest services track the task, so ingest loops drain deterministically under guardrail coverage alongside the TaskSupervisor fallbacks for liquidity probes and FIX loops.【F:src/data_foundation/ingest/scheduler.py†L221-L282】【F:src/data_foundation/ingest/institutional_vertical.py†L394-L441】【F:src/runtime/runtime_builder.py†L3934-L3998】【F:src/trading/execution/liquidity_prober.py†L83-L128】【F:src/trading/integration/fix_broker_interface.py†L126-L186】【F:tests/data_foundation/test_ingest_scheduler.py†L177-L236】
  - Progress: Runtime builder now resolves ingest/trading restart policy overrides from `config.extras`, threads operator-provided limits into bootstrap ingest and trading workloads, and regression coverage confirms overrides stick while invalid inputs fall back with warnings.【src/runtime/runtime_builder.py:2687】【src/runtime/runtime_builder.py:2791】【src/runtime/runtime_builder.py:3340】【tests/runtime/test_runtime_builder.py:657】【tests/runtime/test_runtime_builder.py:698】
  - Progress: RuntimeApplication.summary now buckets TaskSupervisor snapshots by workload and attaches the live task entries to each ingestion/trading block, giving operators per-workload visibility without scraping raw supervisor dumps; guardrail coverage locks the new snapshot contract.【F:src/runtime/runtime_builder.py†L1219-L1273】【F:tests/runtime/test_runtime_builder.py†L2364-L2397】
  - Progress: Runtime shutdown now directs the shared `TaskSupervisor` to cancel active workloads before running teardown callbacks, logs cancelled workloads instead of surfacing spurious exceptions, and leaves no dangling task snapshots; regression coverage asserts long-running ingest/trade coroutines are cancelled and summarised cleanly.【F:src/runtime/runtime_builder.py†L946-L1083】【F:tests/runtime/test_runtime_builder.py†L167-L224】
  - Progress: Bootstrap runtime now exposes `create_background_task`/`register_background_task` helpers that bind work to the shared supervisor, clone metadata into describeable snapshots, and scrub entries upon completion; guardrail tests confirm metadata visibility/cleanup and force the loop to crash once to prove restart bookkeeping and error logging meet the roadmap’s supervised-run torture test.【F:src/runtime/bootstrap_runtime.py†L680-L765】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L246-L418】
  - Progress: Ingest execution now threads the shared `TaskSupervisor` into the operational backbone pipeline, merges pipeline task snapshots back into runtime telemetry, and ships regression coverage so readiness packets list delegated ingest workers instead of dropping them when the pipeline succeeds.【F:src/runtime/runtime_builder.py†L329-L386】【F:src/runtime/runtime_builder.py†L1800-L1868】【F:tests/runtime/test_runtime_builder.py†L345-L362】
  - Progress: ProfessionalPredatorApp now detects supervisor-aware components, injects the shared TaskSupervisor via setter or keyword, registers component-spawned tasks, and merges metadata when workloads re-register; new runtime coverage confirms supervised components expose their background tasks and supervisor state for operators.【F:src/runtime/predator_app.py†L360-L928】【F:src/runtime/task_supervisor.py†L170-L200】【F:tests/current/test_runtime_professional_app.py†L99-L199】
  - Progress: Event bus diagnostics expose `AsyncEventBus.task_snapshots()` (with `TopicBus` delegation) so supervised worker metadata is queryable, and `ProfessionalPredatorApp.summary()` captures the snapshots under `event_bus_tasks`, giving operations a live roster of supervised workloads; regression coverage exercises the metadata contract and summary payload wiring while keeping diagnostics best-effort on failure.【F:src/core/_event_bus_impl.py†L291-L708】【F:src/runtime/predator_app.py†L1491-L1512】【F:tests/current/test_event_bus_task_supervision.py†L103-L156】【F:tests/current/test_runtime_professional_app.py†L433-L461】
  - Progress: Professional runtime bootstrap now resolves trade throttle config from `SystemConfig.extras`, normalises JSON/file overrides and individual toggles, and passes the result to the trading manager so operators can dial rate limits from deployment settings with regression coverage validating the extras path end-to-end.【F:src/runtime/predator_app.py†L1723-L2470】【F:src/runtime/bootstrap_runtime.py†L123-L201】【F:tests/runtime/test_trade_throttle_configuration.py†L1-L55】
  - Progress: Live-shadow ingest CLI now orchestrates the Timescale→Redis→Kafka pipeline, captures scheduler/cache/connectivity evidence, supports JSON/Markdown/text summaries, and honours connector requirement flags so ops can archive supervised drills without bespoke notebooks, with CLI regressions covering summary content and flag routing.【F:tools/data_ingest/run_live_shadow.py†L1-L341】【F:tools/data_ingest/run_live_shadow.py†L241-L341】【F:tests/tools/test_run_live_shadow.py†L176-L210】
  - Progress: Final dry run harness now delegates pumps, timeout, and wait tasks to a shared `TaskSupervisor`, cancelling owned workloads on exit with regression coverage proving supervised task registration and cleanup so UAT rehearsals inherit the same guardrails as runtime entrypoints; the wrapper also ships as an EMP CLI subcommand so ops teams can launch `emp final-dry-run` with documented switches backed by CLI pass/fail tests, and fresh CLI knobs (`--diary-stale-*`, `--performance-stale-*`, `--evidence-*`) now drive evidence freshness monitors that surface WARN/FAIL incidents, land in progress snapshots, and are documented in the runbook with regression coverage over diary stall detection.【F:src/operations/final_dry_run.py†L203-L823】【F:emp/cli/final_dry_run.py†L1-L420】【F:emp/__main__.py†L1-L33】【F:docs/runbooks/final_dry_run.md†L36-L42】【F:tools/operations/final_dry_run.py†L194-L452】【F:tests/operations/test_final_dry_run_cli.py†L49-L134】【F:tests/operations/test_final_dry_run.py†L138-L411】 New `--objective` flags record acceptance outcomes directly into review briefs/JSON so readiness checkpoints are logged alongside incidents, while stdout/stderr stack traces now escalate to fail-severity harness incidents and persist in progress snapshots under guardrail coverage.【F:emp/cli/final_dry_run.py†L246-L360】【F:tools/operations/final_dry_run.py†L290-L392】【F:src/operations/final_dry_run_review.py†L220-L307】【F:docs/runbooks/final_dry_run.md†L80-L90】【F:src/operations/final_dry_run.py†L1206-L1321】【F:tests/operations/test_final_dry_run.py†L383-L401】 The smoke harness now includes a simulated runtime and CLI shim so operators can rehearse evidence collection, packet assembly, and review briefs in minutes before multi-day drills.【F:src/operations/final_dry_run_simulated_runtime.py†L1-L214】【F:src/operations/final_dry_run_smoke.py†L1-L298】【F:tools/operations/final_dry_run_smoke.py†L1-L361】【F:tests/operations/test_final_dry_run_smoke.py†L1-L43】【F:docs/runbooks/final_dry_run.md†L54-L73】 New `--compress-logs` support spans EMP/tooling entrypoints, writing `.jsonl.gz`/`.log.gz` artefacts and exercising gzip decode guards so long rehearsals stay storage-friendly without breaking audit ingestion.【F:emp/cli/final_dry_run.py†L268-L424】【F:src/operations/final_dry_run.py†L67-L724】【F:tools/operations/final_dry_run.py†L257-L454】【F:tests/operations/test_final_dry_run.py†L339-L411】【F:docs/runbooks/final_dry_run.md†L19-L77】
  - Progress: A turnkey `final_dry_run_orchestrator.py` wrapper now provisions timestamped run directories, injects diary/performance/log paths via `FINAL_DRY_RUN_*` environment variables, and streams objectives, attendees, and notes into generated review/evidence packets; regression coverage and the refreshed runbook capture the end-to-end workflow so operators can launch rehearsals without bespoke scripting.【F:tools/operations/final_dry_run_orchestrator.py†L1-L575】【F:tests/operations/test_final_dry_run_orchestrator.py†L1-L180】【F:docs/runbooks/final_dry_run.md†L1-L118】
  - Progress: Progress reporter now escalates the highest-severity incident into the live snapshot, writes WARN/FAIL telemetry immediately when log monitors trip, and guardrail coverage tails the JSON progress file to confirm incidents surface before the harness terminates so governance can watch rehearsals in flight.【F:src/operations/final_dry_run.py†L209-L338】【F:tests/operations/test_final_dry_run.py†L277-L313】
  - Progress: Dry-run log ingestion now streams gz/bz2/xz archives and trims trailing `Z` suffixes from ISO timestamps, eliminating manual decompression while regression coverage exercises gzip/bz2 fixtures and diary timestamp parsing.【F:src/operations/dry_run_audit.py†L31-L671】【F:src/understanding/decision_diary.py†L41-L54】【F:tests/operations/test_dry_run_audit.py†L1-L47】
  - Progress: TaskSupervisor restart policies now relaunch failed workloads with bounded backoff, enforce per-task hang timeouts, and surface those limits through `describe()` snapshots so hung loops trigger predictable restarts under guardrail coverage.【F:src/runtime/task_supervisor.py†L167-L274】【F:tests/runtime/test_task_supervisor.py†L132-L220】 Runtime builder/summary wiring now threads configurable `hang_timeout` values from workload extras onto ingestion/trading workloads, exposing the limits in runtime telemetry so operators see exact safeguards in effect.【F:src/runtime/runtime_builder.py†L1129-L1210】【F:src/runtime/runtime_builder.py†L2869-L3095】【F:tests/runtime/test_runtime_builder.py†L572-L603】
  - Progress: Task supervisor snapshots now surface each task’s age in seconds, and pytest coverage asserts the metric grows monotonically so ops consoles can flag hung workloads before cancellation deadlines.【F:src/runtime/task_supervisor.py†L15-L33】【F:src/runtime/task_supervisor.py†L203-L224】【F:tests/runtime/test_task_supervisor.py†L25-L47】
  - Progress: Runtime builder metadata now tags ingestion and trading workloads—including skip paths—with `workload_kind` and `supervised_components`, and regression coverage proves summaries surface the labels while supervised retries keep trading live after an ingest failure, giving ops dashboards deterministic context for recovery drills.【F:src/runtime/runtime_builder.py†L2589-L2609】【F:src/runtime/runtime_builder.py†L4155-L4160】【F:tests/runtime/test_runtime_builder.py†L470-L561】
  - Progress: Runtime application supervision now tracks per-workload states, logs failures without collapsing sibling loops, and surfaces supervisor namespace, active task counts, restart policies, and workload states via `summary()` so operators can verify trading continues while ingest restarts under regression coverage.【F:src/runtime/runtime_builder.py†L1007-L1070】【F:tests/runtime/test_runtime_builder.py†L1683-L1767】
  - Progress: Cancel-timeout diagnostics now log the hung task name, captured metadata, and asyncio state whenever shutdown exceeds the configured timeout so operators can chase specific ingestion or trading loops instead of generic timeouts, with regression coverage locking the enriched error payload.【F:src/runtime/task_supervisor.py†L205-L248】【F:tests/runtime/test_task_supervisor.py†L141-L167】
  - Progress: Async event bus now provisions a scoped TaskSupervisor by default, cancels supervised handlers when factories swap, and exposes a fallback supervisor helper for ad-hoc jobs so lifecycle transitions no longer leak background tasks under regression coverage.【F:src/core/_event_bus_impl.py†L44-L520】【F:tests/runtime/test_runtime_builder.py†L60-L129】
- Harden operational telemetry publishers so security, system validation, and
  professional readiness feeds warn on runtime bus failures, fall back
  deterministically, and raise on unexpected errors with pytest coverage
  guarding the behaviour. The system validation track now derives reliability
  summaries, evaluates gate decisions, emits gate alerts, and publishes via the
  shared failover helper so responders inherit blocking reasons, stale-hour
  thresholds, and failover guarantees alongside validation results.【F:src/operations/security.py†L536-L579】【F:tests/operations/test_security.py†L101-L211】【F:src/operations/system_validation.py†L470-L746】【F:tests/operations/test_system_validation.py†L1-L432】【F:src/operations/professional_readiness.py†L268-L305】【F:tests/operations/test_professional_readiness.py†L164-L239】
- Harden incident response readiness by parsing policy/state mappings into a
  severity snapshot, deriving targeted alert events, publishing telemetry via
  the guarded runtime→global failover path, and tracking major incident review
  cadence with structured issue catalogs so overdue postmortems escalate under
  regression coverage documenting publish failures and gate metadata.【F:src/operations/incident_response.py†L242-L558】【F:tests/operations/test_incident_response.py†L1-L276】【F:src/operations/event_bus_failover.py†L1-L174】
- Document Timescale failover drill requirements via the institutional ingest
  provisioner, which now exposes drill metadata from configuration and captures
  the workflow in updated runbooks so operators can rehearse recoveries using a
  consistent source of truth.【F:src/data_foundation/ingest/institutional_vertical.py†L160-L239】【F:docs/operations/timescale_failover_drills.md†L1-L27】
- Progress: Operational backbone caches the latest successful market and macro frames, reusing them when Timescale fetches raise so supervised ingest runs keep sensory and belief feeds populated while upstream services recover; warnings differentiate fallback reuse from empty data to aid incident triage.【F:src/data_foundation/pipelines/operational_backbone.py†L187-L377】
- Aggregate operational readiness into a single severity snapshot that merges
  system validation, incident response, drift, and ingest SLO posture, emits
  Markdown summaries, evaluates gate decisions with blocking/warn thresholds,
  and exposes status breakdowns plus per-component issue catalogs so dashboards
  and alerts share deterministic remediation context under regression coverage
  and updated status docs.【F:src/operations/operational_readiness.py†L113-L744】【F:tests/operations/test_operational_readiness.py†L86-L389】【F:docs/status/operational_readiness.md†L1-L140】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
  - Progress: Strategy performance tracker now computes per-strategy KPIs,
    loop metrics, ROI posture, and Markdown summaries so readiness dashboards
    can surface trading-loop health from one aggregation surface under pytest
    coverage.【F:src/operations/strategy_performance_tracker.py†L1-L596】【F:tests/operations/test_strategy_performance_tracker.py†L1-L122】
  - Progress: Final dry-run audit bundle now consolidates logs, diary flags,
    readiness posture, and KPI telemetry into a single Markdown report with
    severity roll-ups, while detecting log continuity gaps, uptime degradation,
    and configurable warn/fail thresholds via the CLI so UAT rehearsals and
    sign-off reviews surface evidence drop-outs alongside curated summaries under
    pytest coverage.【F:src/operations/dry_run_audit.py†L18-L789】【F:tests/operations/test_dry_run_audit.py†L1-L220】【F:tools/operations/final_dry_run_audit.py†L1-L118】
  - Progress: Sign-off evaluation now layers minimum-duration and uptime gates on top of the audit summary, requires diary/performance evidence by default, and exposes CLI toggles so review boards can approve, warn, or fail runs with consistent criteria under regression coverage.【F:src/operations/dry_run_audit.py†L792-L835】【F:tools/operations/final_dry_run_audit.py†L21-L150】【F:tests/operations/test_dry_run_audit.py†L189-L259】
  - Progress: Sign-off guardrails now enforce configurable Sharpe-ratio floors and add a `--sign-off-min-sharpe` CLI switch so governance can demand risk-adjusted performance proof, with tests covering success, failure, and missing-metric cases.【F:src/operations/dry_run_audit.py†L323-L1072】【F:tools/operations/final_dry_run_audit.py†L100-L178】【F:tests/operations/test_dry_run_audit.py†L226-L360】
  - Progress: Diary audits now enforce per-day entry minimums with configurable severities and CLI toggles, raising explicit `coverage/daily/*` issues when journaling gaps appear so sign-off packets surface documentation lapses under regression coverage.【F:src/operations/dry_run_audit.py†L738-L947】【F:tools/operations/final_dry_run_audit.py†L21-L206】【F:tests/operations/test_dry_run_audit.py†L188-L304】
- Progress: Default alert policy now delivers email, SMS, webhook, Slack, and
  GitHub issue transports out of the box, with regression coverage asserting
  channel fan-out for readiness, incident response, and drift sentry alerts and
  runbooks/status pages documenting the new escalation paths.【F:src/operations/alerts.py†L407-L823】【F:tests/operations/test_alerts.py†L1-L338】【F:docs/operations/runbooks/drift_sentry_response.md†L28-L33】【F:docs/status/operational_readiness.md†L68-L82】
- Progress: Drift sentry detectors now publish understanding-loop telemetry via the
  failover helper, feed the new `drift_sentry` readiness component, and link the
  shared runbook so incident response inherits Page–Hinkley/variance issue
  catalogs alongside sensory drift, with regression coverage across the snapshot,
  alert derivation, and documentation updates.【F:src/operations/drift_sentry.py†L1-L399】【F:tests/operations/test_drift_sentry.py†L43-L135】【F:tests/operations/test_operational_readiness.py†L200-L283】【F:docs/operations/runbooks/drift_sentry_response.md†L1-L69】
- Progress: Sensory drift regression now ships a deterministic Page–Hinkley replay
  fixture and metadata assertions so alert payloads reproduce the detector catalog,
  runbook link, and severity stats that readiness dashboards expect, under pytest
  coverage.【F:tests/operations/fixtures/page_hinkley_replay.json†L1-L128】【F:tests/operations/test_sensory_drift.py†L157-L218】
- Wire the observability dashboard to consume the readiness snapshot directly,
  rendering a dedicated panel with component summaries and remediation roll-ups
  under pytest coverage so operators see readiness posture alongside risk,
  latency, and backbone panels without bespoke integrations.【F:src/operations/observability_dashboard.py†L754-L815】【F:tests/operations/test_observability_dashboard.py†L220-L293】
- Progress: Understanding-loop diagnostics now populate an observability panel summarising regime confidence, drift exceedances, gating decisions, and ledger approvals so AlphaTrade reviewers see loop posture alongside readiness metrics with regression coverage guarding the snapshot contract.【F:src/operations/observability_dashboard.py†L822-L875】【F:tests/operations/test_observability_dashboard.py†L582-L624】
  - Progress: Compliance panels now ingest the AlphaTrade loop’s enriched governance events, surfacing stage labels, stage-gate reasons, and breach counts directly on the dashboard so operators can reconcile forced-paper decisions without spelunking through logs.【F:src/operations/observability_dashboard.py†L825-L963】【F:tests/operations/test_observability_dashboard.py†L340-L368】
- Update incident response docs with current limitations and TODOs; remove or
  archive obsolete OpenAPI references where possible.【F:docs/legacy/README.md†L1-L12】
- Extend CI step summaries to include risk, ingest, and sensory telemetry status so
  failures surface promptly.

### Next (30–90 days)

- Rehearse forced-failure drills for the new Slack/GitHub transports, measure
  MTTA/MTTR, and capture evidence in the context packs as called out in the
  technical debt plan.【F:docs/technical_debt_assessment.md†L156-L174】
- Build operator dashboards for ingest health, task supervision status, and risk
  policy compliance.
- Document cross-region failover, cache outages, and Kafka lag recovery once the
  data backbone stabilises.

### Later (90+ days)

- Establish continuous system validation with automated gating on readiness
  metrics.
- Integrate security reviews, secrets management, and compliance sign-offs into
  the operational cadence.【F:docs/technical_debt_assessment.md†L45-L112】
- Remove dead-code operational scripts after new runbooks are in place to reduce
  maintenance burden.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Depends on data backbone telemetry and risk enforcement delivering actionable
  signals.
- Needs collaboration with compliance initiatives to ensure incident response
  covers regulatory obligations.
