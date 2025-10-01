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
- Harden operational telemetry publishers so security and system validation
  feeds warn on runtime bus failures, fall back deterministically, and raise on
  unexpected errors with pytest coverage guarding the behaviour, including
  regressions that capture global-bus fallbacks, runtime-not-running paths, and
  unexpected error escalation for security publishing.【F:src/operations/security.py†L536-L579】【F:tests/operations/test_security.py†L101-L211】【F:src/operations/system_validation.py†L269-L312】【F:tests/operations/test_system_validation.py†L85-L137】
- Harden incident response telemetry with the shared failover helper so
  snapshots reuse the guarded runtime→global publish path, surface warning/error
  logs for degraded transports, and raise typed errors under pytest coverage
  instead of silently skipping outage evidence.【F:src/operations/incident_response.py†L350-L375】【F:tests/operations/test_incident_response.py†L123-L167】【F:src/operations/event_bus_failover.py†L1-L174】
- Aggregate operational readiness into a single severity snapshot that merges
  system validation, incident response, and ingest SLO posture, emits Markdown
  summaries, derives alert events, and now exposes status breakdown/component
  metadata so dashboards can render severity chips without recomputing logic,
  with pytest guarding the contract and docs capturing the payload update.【F:src/operations/operational_readiness.py†L1-L256】【F:tests/operations/test_operational_readiness.py†L1-L86】【F:docs/status/operational_readiness.md†L1-L34】【F:tests/runtime/test_professional_app_timescale.py†L722-L799】
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
