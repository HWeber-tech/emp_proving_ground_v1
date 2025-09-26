# Alignment brief – Operational readiness & resilience telemetry

**Why this brief exists:** Layer 5 of the concept blueprint emphasises monitoring, security, incident response, and disaster
recovery. The modernization roadmap now ships backup drills, incident response grading, security posture snapshots, and
professional readiness aggregates, but the context pack corpus lacked a dedicated operational brief tying those artefacts
back to the concept promises. This document keeps ongoing work anchored to the roadmap milestones and telemetry evidence.
【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L1611-L1638】【F:docs/roadmap.md†L30-L47】

## Concept promise

- Operations and monitoring must deliver real-time health, alerting, security controls, and disaster recovery aligned with the
  institutional tier claims.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L1611-L1638】
- The roadmap’s gap table commits to designing the monitoring stack, access controls, and backup routines that satisfy the
  concept doc’s operational maturity targets.【F:docs/roadmap.md†L30-L47】
- Later roadmap checkpoints track Spark failover drills, cross-region rehearsals, and automated scheduler cutover, establishing
  long-horizon expectations for operational resilience.【F:docs/roadmap.md†L83-L132】【F:docs/roadmap.md†L260-L280】

## Reality snapshot (September 2025)

- `evaluate_professional_readiness` merges data backbone readiness, backup posture, SLOs, failover decisions, and recovery
  recommendations into a single snapshot the runtime publishes and persists for operators.【F:src/operations/professional_readiness.py†L200-L258】【F:src/runtime/runtime_builder.py†L2190-L2320】【F:tests/runtime/test_professional_app_timescale.py†L1-L320】
- Backup readiness policies grade frequency, retention, and restore drills, with runtime builder wiring that records snapshots
  and publishes `telemetry.operational.backups` after each ingest run.【F:src/operations/backup.py†L134-L220】【F:src/runtime/runtime_builder.py†L2190-L2270】
- Operational SLO, security posture, incident response, system validation, and event bus health modules each emit reusable
  snapshots with Markdown renderers, runtime publications, and regression coverage documented in CI health decks.【F:src/operations/slo.py†L200-L290】【F:src/operations/security.py†L460-L552】【F:src/operations/incident_response.py†L240-L339】【F:src/operations/system_validation.py†L203-L297】【F:src/operations/event_bus_health.py†L1-L200】【F:docs/status/ci_health.md†L83-L210】
- Spark stress drills, data backbone validation, cache health, and failover exercises integrate into the runtime lifecycle,
  logging markdown summaries and storing Timescale history for audit trails.【F:src/operations/spark_stress.py†L1-L166】【F:src/operations/data_backbone.py†L15-L360】【F:src/operations/cache_health.py†L1-L200】【F:src/runtime/runtime_builder.py†L657-L1256】
- OpenTelemetry tracing can be toggled with `OTEL_*` extras; the runtime calls `configure_event_bus_tracer` so `AsyncEventBus`
  attaches queue depth and dispatch lag attributes to publish and handler spans, and reuses the same tracer for runtime
  workloads so startup/shutdown hooks, workload execution, and Timescale ingest orchestration add plan, fallback, and success
  metadata into distributed traces.【F:src/observability/tracing.py†L1-L244】【F:src/runtime/predator_app.py†L1560-L1610】【F:src/runtime/runtime_builder.py†L250-L408】【F:src/runtime/runtime_builder.py†L1996-L2320】【F:tests/core/test_event_bus_tracing.py†L1-L118】【F:tests/runtime/test_runtime_tracing.py†L1-L134】
- Structured logging can now be enabled via `RUNTIME_LOG_STRUCTURED`, `RUNTIME_LOG_LEVEL`, and `RUNTIME_LOG_CONTEXT`
  extras, which the runtime builder feeds into `configure_structured_logging` to emit JSON lines with runtime tier,
  environment, and deployment metadata for drill playback and centralised ingestion.【F:src/observability/logging.py†L1-L122】【F:src/runtime/runtime_builder.py†L1838-L1877】【F:tests/observability/test_logging.py†L1-L74】【F:tests/runtime/test_runtime_builder.py†L600-L676】
- `tools/telemetry/export_operational_snapshots.py` now exports professional readiness, security, incident response, and system
  validation blocks as JSON for Grafana/DataDog ingestion, closing the dashboard gap highlighted in this brief. pytest covers
  success, missing-section warnings, and the `--allow-missing` override.【F:tools/telemetry/export_operational_snapshots.py†L1-L143】【F:tests/tools/test_operational_export.py†L1-L86】

## Gap map

| Concept excerpt | Observable gap | Impact |
| --- | --- | --- |
| Operations layer requires security, alerting, and recovery drills.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L1611-L1638】 | Incident response and backup telemetry exist, but no single runbook summarises how to react to combined degradations (e.g., security failure + ingest outage). | On-call engineers must cross-reference multiple docs under pressure, increasing incident response time. |
| Roadmap emphasises cross-region failover rehearsals and scheduler cutover.【F:docs/roadmap.md†L260-L280】 | Cross-region telemetry ships, yet we lack a documented schedule for executing the drills or logging evidence in the CI health deck. | Without cadence, resilience claims risk drifting from reality and failing audits. |
| Monitoring stack should provide dashboard-ready feeds.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L1619-L1638】 | Telemetry publishes to event bus/Kafka, but there is no automated export that feeds operations dashboards or portfolio snapshot metrics. | Stakeholders cannot easily compare operational posture across runs without manual bus subscriptions. |

## Delivery plan

### Now (30-day outlook)

1. **Composite incident playbook** – Author an operations runbook that chains backup, security, and incident response telemetry
   into decision trees, referencing the new markdown payloads so responders can act on runtime summaries.【F:docs/operations/runbooks/redis_cache_outage.md†L1-L60】【F:src/operations/incident_response.py†L240-L339】
2. **Drill cadence logging** – Extend the CI health snapshot with timestamps for failover, Spark stress, and incident drills to
   prove cadence compliance and guide future rehearsals.【F:docs/status/ci_health.md†L83-L210】【F:src/operations/failover_drill.py†L1-L213】
3. **Dashboard export CLI** – ✅ Completed via `tools/telemetry/export_operational_snapshots.py`, which dumps the four
   high-value operational blocks to JSON (with pytest coverage). Next step: wire the JSON output into the shared observability
   dashboard job so the data lands alongside CI metrics.【F:tools/telemetry/export_operational_snapshots.py†L1-L143】【F:tests/tools/test_operational_export.py†L1-L86】

4. **Roadmap guardrail** – Update `tools.roadmap.snapshot` so the modernization portfolio check imports backup, event bus,
   SLO, system validation, Kafka readiness, and backbone validation helpers, causing the dashboard to fail fast if operational
   telemetry disappears.【F:tools/roadmap/snapshot.py†L188-L205】

### Next (60-day outlook)

1. **Cross-region rehearsal schedule** – Encode drill cadence into the runtime extras and add a scheduler task that records
   rehearsal metadata so professional summaries and CI logs show when replica checks last ran.【F:src/operations/cross_region_failover.py†L1-L238】【F:src/runtime/runtime_builder.py†L2440-L2520】
2. **Alert routing verification** – Expand operational SLO tests to assert custom alert routes reach configured channels (Slack,
   PagerDuty) using test doubles, documenting expectations in the observability plan.【F:src/operations/slo.py†L200-L290】【F:docs/operations/observability_plan.md†L1-L120】
3. **Data backbone validation dashboard** – Publish Timescale queries that trend backbone validation, cache health, and event bus
   status so resilience metrics appear in the portfolio snapshot automation.【F:src/data_foundation/persist/timescale.py†L1-L720】【F:docs/roadmap.md†L381-L400】

### Later (90-day+ considerations)

- Automate evidence packages for SOC2-style reviews by bundling system validation reports, incident telemetry, and backup logs
  into versioned artefacts per release.【F:src/operations/system_validation.py†L203-L297】【F:docs/status/ci_health.md†L180-L210】
- Integrate operational telemetry into deployment workflows so production releases block on degraded readiness snapshots and
  provide rollback recommendations automatically.【F:src/runtime/runtime_builder.py†L2190-L2520】

## Validation hooks

- **Operational pytest suites** – Keep backup, security, incident response, and professional readiness tests green so regressions
  surface immediately during CI runs.【F:tests/operations/test_backup.py†L1-L120】【F:tests/operations/test_security.py†L1-L140】【F:tests/operations/test_incident_response.py†L1-L110】【F:tests/operations/test_professional_readiness.py†L1-L140】
- **Runtime integration tests** – `tests/runtime/test_professional_app_timescale.py` already asserts summary exposure for backup,
  security, incident, and system validation blocks; extend it when new telemetry feeds land.【F:tests/runtime/test_professional_app_timescale.py†L320-L620】
- **Event bus publishing** – Use the runtime builder tests to verify each telemetry helper publishes onto the event bus so
  downstream dashboards continue to receive operational signals.【F:tests/runtime/test_runtime_builder.py†L200-L520】

## Open questions

1. Which operational telemetry should gate production deployments versus merely warn operators, and how do we encode those
   policies in the runtime builder?【F:src/runtime/runtime_builder.py†L2190-L2520】
2. How frequently should cross-region failover and Spark stress drills run to satisfy institutional SLAs, and how do we track
   completion evidence?【F:docs/roadmap.md†L260-L280】
3. What automation should package incident/backup telemetry for external audits (SOC2, ISO) without manual intervention?【F:docs/status/ci_health.md†L83-L210】
