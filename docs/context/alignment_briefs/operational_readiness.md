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
- Harden operational telemetry publishers so security, system validation, and
  professional readiness feeds warn on runtime bus failures, fall back
  deterministically, and raise on unexpected errors with pytest coverage
  guarding the behaviour. The system validation track now evaluates structured
  reports into readiness snapshots, derives alert events, publishes via the
  shared failover helper, and exposes gating helpers that return blocking
  reasons and success metrics so responders see the exact degradation and
  deployment posture.【F:src/operations/security.py†L536-L579】【F:tests/operations/test_security.py†L101-L211】【F:src/operations/system_validation.py†L470-L889】【F:tests/operations/test_system_validation.py†L1-L279】【F:src/operations/professional_readiness.py†L268-L305】【F:tests/operations/test_professional_readiness.py†L164-L239】
- Harden incident response readiness by parsing policy/state mappings into a
  severity snapshot, deriving targeted alert events, and publishing telemetry
  via the guarded runtime→global failover path so outage evidence, roster gaps,
  postmortem backlog context, and structured issue catalogs (counts, highest
  severity, category tags) stay visible under pytest coverage documenting
  escalation and publish failures.【F:src/operations/incident_response.py†L242-L715】【F:tests/operations/test_incident_response.py†L1-L200】【F:src/operations/event_bus_failover.py†L1-L174】
- Document Timescale failover drill requirements via the institutional ingest
  provisioner, which now exposes drill metadata from configuration and captures
  the workflow in updated runbooks so operators can rehearse recoveries using a
  consistent source of truth.【F:src/data_foundation/ingest/institutional_vertical.py†L160-L239】【F:docs/operations/timescale_failover_drills.md†L1-L27】
- Aggregate operational readiness into a single severity snapshot that merges
  system validation, incident response, and ingest SLO posture, emits Markdown
  summaries, derives alert events, and exposes status breakdowns, component
  status maps, issue counts, and per-component issue catalogs so dashboards can
  render severity chips and remediation context without recomputing logic, with
  pytest guarding alert derivation, routing, and the failover publish path while
  docs capture the enriched payload contract.【F:src/operations/operational_readiness.py†L113-L373】【F:tests/operations/test_operational_readiness.py†L86-L221】【F:docs/status/operational_readiness.md†L1-L73】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
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
  latency, and backbone panels without bespoke integrations.【F:src/operations/observability_dashboard.py†L443-L493】【F:tests/operations/test_observability_dashboard.py†L135-L236】
- Progress: Understanding-loop diagnostics now populate an observability panel summarising regime confidence, drift exceedances, gating decisions, and ledger approvals so AlphaTrade reviewers see loop posture alongside readiness metrics with regression coverage guarding the snapshot contract.【F:src/operations/observability_dashboard.py†L513-L548】【F:tests/operations/test_observability_dashboard.py†L371-L384】
- Update incident response docs with current limitations and TODOs; remove or
  archive obsolete OpenAPI references where possible.【F:docs/legacy/README.md†L1-L12】
- Extend CI step summaries to include risk, ingest, and sensory telemetry status so
  failures surface promptly.

### Next (30–90 days)

- Implement alert routing (GitHub + Slack/webhook) and rehearse forced-failure
  drills, measuring MTTA/MTTR as called out in the technical debt plan.【F:docs/technical_debt_assessment.md†L156-L174】
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
