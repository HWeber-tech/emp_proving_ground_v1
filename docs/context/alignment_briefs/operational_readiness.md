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
  - Progress: Timescale ingest scheduler, liquidity probes, and the FIX broker interface now launch background loops via TaskSupervisor fallbacks and tear them down cleanly, with regression proving ingest failures no longer cancel sibling monitors.【F:src/data_foundation/ingest/scheduler.py†L99-L167】【F:src/trading/execution/liquidity_prober.py†L83-L128】【F:src/trading/integration/fix_broker_interface.py†L126-L186】【F:tests/runtime/test_task_supervisor.py†L63-L103】
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
- Progress: Default alert policy now delivers email, SMS, webhook, Slack, and
  GitHub issue transports out of the box, with regression coverage asserting
  channel fan-out for readiness, incident response, and drift sentry alerts and
  runbooks/status pages documenting the new escalation paths.【F:src/operations/alerts.py†L407-L823】【F:tests/operations/test_alerts.py†L1-L338】【F:docs/operations/runbooks/drift_sentry_response.md†L28-L33】【F:docs/status/operational_readiness.md†L68-L82】
- Progress: Drift sentry detectors now publish understanding-loop telemetry via the
  failover helper, feed the new `drift_sentry` readiness component, and link the
  shared runbook so incident response inherits Page–Hinkley/variance issue
  catalogs alongside sensory drift, with regression coverage across the snapshot,
  alert derivation, and documentation updates.【F:src/operations/drift_sentry.py†L1-L399】【F:tests/intelligence/test_drift_sentry.py†L43-L135】【F:tests/operations/test_operational_readiness.py†L200-L283】【F:docs/operations/runbooks/drift_sentry_response.md†L1-L69】
- Progress: Sensory drift regression now ships a deterministic Page–Hinkley replay
  fixture and metadata assertions so alert payloads reproduce the detector catalog,
  runbook link, and severity stats that readiness dashboards expect, under pytest
  coverage.【F:tests/operations/fixtures/page_hinkley_replay.json†L1-L128】【F:tests/operations/test_sensory_drift.py†L157-L218】
- Wire the observability dashboard to consume the readiness snapshot directly,
  rendering a dedicated panel with component summaries and remediation roll-ups
  under pytest coverage so operators see readiness posture alongside risk,
  latency, and backbone panels without bespoke integrations.【F:src/operations/observability_dashboard.py†L754-L815】【F:tests/operations/test_observability_dashboard.py†L220-L293】
- Progress: Understanding-loop diagnostics now populate an observability panel summarising regime confidence, drift exceedances, gating decisions, and ledger approvals so AlphaTrade reviewers see loop posture alongside readiness metrics with regression coverage guarding the snapshot contract.【F:src/operations/observability_dashboard.py†L822-L875】【F:tests/operations/test_observability_dashboard.py†L582-L624】
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
