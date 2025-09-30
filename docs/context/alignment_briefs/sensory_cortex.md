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
