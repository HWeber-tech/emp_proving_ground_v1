# Alignment brief – Sensory cortex & anomaly telemetry

## Concept promise

- The encyclopedia positions the 4D+1 sensory cortex (WHY, WHAT, WHEN, HOW,
  ANOMALY) as the layer that perceives markets in real time and feeds downstream
  intelligence.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L360-L436】
- Weekly milestones highlight enhanced sensory cortex delivery with calibrated
  telemetry and integration across the runtime stack.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L534-L704】

## Reality snapshot

- Evolution, intelligence, execution, and strategy subsystems remain mock
  frameworks; sensory organs still ship as placeholders with limited heuristics
  and no production data feeds.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Dead-code and dependency audits list sensory modules among unused paths,
  underscoring the lack of executable coverage.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Technical debt priorities call out async hazards and namespace drift that block
  reliable runtime wiring for sensory subscriptions.【F:docs/technical_debt_assessment.md†L33-L80】
- HOW and ANOMALY organs now emit sanitised lineage records with shared
  threshold posture assessments, surfacing state/breach metadata alongside the
  telemetry so downstream consumers can trace signal provenance and escalation
  context under pytest coverage, though inputs remain synthetic until the ingest
  backbone is live.【F:src/sensory/how/how_sensor.py†L67-L194】【F:src/sensory/anomaly/anomaly_sensor.py†L121-L220】【F:src/sensory/thresholds.py†L1-L76】【F:tests/sensory/test_how_anomaly_sensors.py†L87-L175】【F:tests/sensory/test_thresholds.py†L1-L57】
- Real sensory organ fuses WHY/WHAT/WHEN/HOW/ANOMALY outputs, publishes
  telemetry snapshots with lineage metadata, and exposes audit/status helpers
  while still consuming synthetic data until institutional ingest arrives.【F:src/sensory/real_sensory_organ.py†L20-L208】【F:src/sensory/real_sensory_organ.py†L210-L336】【F:tests/sensory/test_real_sensory_organ.py†L1-L107】
- Sensory metrics layer now converts organ status snapshots into
  dimension-strength/confidence metrics, captures drift alerts, and publishes via
  the failover helper so dashboards can surface cortex posture even when the
  runtime bus degrades.【F:src/operations/sensory_metrics.py†L1-L159】【F:tests/operations/test_sensory_metrics.py†L1-L92】
- Core package now logs and documents the sensory organ import fallback, keeping
  stub exports visible under pytest coverage so bootstrap environments surface
  degraded wiring instead of silently masking missing dependencies.【F:src/core/__init__.py†L11-L45】【F:tests/core/test_core_init_fallback.py†L1-L43】

## Gap themes

1. **Executable organs** – Implement deterministic HOW and ANOMALY organs, align
   existing WHY/WHAT/WHEN organs with canonical signals, and integrate them into
   runtime summaries.
2. **Telemetry discipline** – Produce drift, confidence, and lineage telemetry
   with coverage that exercises event bus publications and storage.
3. **Data fidelity** – Connect organs to real ingest feeds once the institutional
   data backbone is online, with fallbacks documented for bootstrap mode.

## Delivery plan

### Now (0–30 days)

- Inventory current sensory imports, remove deprecated aliases, and align module
  exports with canonical names to prevent namespace drift.【F:docs/technical_debt_assessment.md†L73-L80】
- Establish placeholder telemetry contracts (schemas, topics) and add pytest
  scaffolding so new organs can land incrementally.
- Update documentation to reflect the mock status, preserving the truth-first
  narrative for reviewers.【F:docs/DEVELOPMENT_STATUS.md†L7-L35】
- Progress: Sensory summary telemetry now constructs ranked Markdown/JSON
  rollups from integrated sensor payloads, preserves drift metadata, and
  publishes via the event-bus failover helper so dashboards receive resilient
  sensory status updates backed by regression coverage of runtime and fallback
  paths.【F:src/operations/sensory_summary.py†L1-L215】【F:tests/operations/test_sensory_summary.py†L1-L155】
- Progress: Integrated organ now exposes a metrics view with integrated
  strength/confidence snapshots, dimension posture, and drift metadata so the
  new metrics publisher can build dashboard payloads without rehydrating raw
  snapshots.【F:src/sensory/real_sensory_organ.py†L201-L233】【F:tests/sensory/test_real_sensory_organ.py†L130-L162】
- Progress: Bootstrap runtime now instantiates the real sensory organ with a
  drift-configured history buffer, streams observations into cortex metrics, and
  surfaces samples/audits via `status()` so supervisors inherit sensory posture
  during live-shadow runs under regression coverage.【F:src/runtime/bootstrap_runtime.py†L210-L334】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L107-L132】

### Next (30–90 days)

- Implement HOW organ metrics (participation, liquidity, volatility) and ANOMALY
  detection tied to ingest feeds; record audit trails in runtime summaries.
- Extend WHY/WHAT/WHEN organs with calibrated signals sourced from Timescale once
  the institutional backbone is live.
- Publish drift telemetry (`telemetry.sensory.drift`) and ensure event bus
  consumers plus storage layers capture the payloads with regression coverage.

### Later (90+ days)

- Introduce adaptive thresholds linked to the evolution engine; capture lineage
  metadata for governance.
- Build operator runbooks for sensor outages and recalibration, including alert
  routing.
- Close dead-code findings by deleting redundant sensory templates once the new
  organs stabilise.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Requires ingest telemetry and risk enforcement to mature in parallel so sensors
  operate on trustworthy data and feed policy checks.
- Evolution engine uplift must expose catalogue snapshots and mutation logs to
  contextualise sensory-driven decisions.【F:docs/technical_debt_assessment.md†L95-L112】
