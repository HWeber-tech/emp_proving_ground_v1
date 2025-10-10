# Alignment brief – Sensory cortex & anomaly telemetry

## Concept promise

- The encyclopedia positions the 4D+1 sensory cortex (WHY, WHAT, WHEN, HOW,
  ANOMALY) as the layer that perceives markets in real time and feeds downstream
  intelligence.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L360-L436】
- Weekly milestones highlight enhanced sensory cortex delivery with calibrated
  telemetry and integration across the runtime stack.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L534-L704】

## Reality snapshot

- Evolution, intelligence, execution, and strategy subsystems remain mock
  frameworks; sensory organs only recently gained the ability to ingest recorded
  market slices via the managed data backbone and still lack continuous
  production feeds.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】【F:src/data_integration/real_data_slice.py†L95-L198】
- Dead-code and dependency audits list sensory modules among unused paths,
  underscoring the lack of executable coverage.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Technical debt priorities call out async hazards and namespace drift that block
  reliable runtime wiring for sensory subscriptions.【F:docs/technical_debt_assessment.md†L33-L80】
- HOW and ANOMALY organs now clamp minimum confidence, sanitise sequence input,
  surface dropped-sample counts, enrich telemetry with order-book analytics, and
  emit sanitised lineage records plus shared threshold posture metadata; the
  anomaly stack now wraps a shared `BasicAnomalyDetector` that enforces rolling
  z-score windows with deterministic sampling constraints and guardrail
  coverage, though inputs remain synthetic until the ingest backbone is
  live.【F:src/sensory/how/how_sensor.py†L21-L210】【F:src/sensory/anomaly/anomaly_sensor.py†L21-L302】【F:src/sensory/anomaly/basic_detector.py†L1-L140】【F:tests/sensory/test_how_anomaly_sensors.py†L170-L294】【F:tests/sensory/test_basic_anomaly_detector.py†L1-L39】
- WHAT/WHEN/WHY sensors now attach structured `quality` metadata (source,
  timestamp, confidence) alongside lineage dictionaries, and regression tests
  assert the enriched payloads to keep downstream consumers aligned while real
  data feeds remain forthcoming.【F:src/sensory/what/what_sensor.py†L131-L210】【F:src/sensory/when/when_sensor.py†L147-L230】【F:src/sensory/why/why_sensor.py†L126-L238】【F:tests/sensory/test_primary_dimension_sensors.py†L1-L98】
- Real sensory organ fuses WHY/WHAT/WHEN/HOW/ANOMALY outputs, publishes
  telemetry snapshots with lineage metadata and bundled metrics payloads, and
  exposes audit/status helpers
  while still consuming synthetic data until institutional ingest arrives. It
  now wires an optional sensory lineage publisher that normalises HOW/ANOMALY
  payloads, keeps a bounded inspection history, emits lineage telemetry via
  runtime or fallback buses, and serialises per-dimension metadata plus numeric
  telemetry so responders and downstream metrics inherit audit-ready payloads
  under pytest coverage.【F:src/sensory/real_sensory_organ.py†L41-L489】【F:src/sensory/real_sensory_organ.py†L198-L205】【F:src/sensory/lineage_publisher.py†L1-L193】【F:tests/sensory/test_real_sensory_organ.py†L96-L183】【F:tests/sensory/test_real_sensory_organ.py†L132-L144】【F:tests/sensory/test_lineage.py†L85-L145】
- Executable HOW/ANOMALY sensory organs now wrap the canonical sensors, normalise
  market frames or sequences, maintain calibrated windows, and emit structured
  lineage/telemetry payloads under guardrail regression coverage, though they
  still consume synthetic payloads until ingest is live.【F:src/sensory/organs/dimensions/executable_organs.py†L1-L226】【F:tests/sensory/test_dimension_organs.py†L1-L93】
- Sensory metrics layer now converts organ status snapshots into
  dimension-strength/confidence metrics, harvests numeric telemetry from audit
  and order-book metadata, captures drift alerts, and publishes via the failover
  helper so dashboards surface cortex posture and raw telemetry even when the
  runtime bus degrades.【F:src/operations/sensory_metrics.py†L1-L200】【F:tests/operations/test_sensory_metrics.py†L1-L130】
- Core package now re-exports the canonical sensory organ, exposes the drift
  configuration dataclass, and coerces legacy drift-config payloads while
  dropping the defensive stub fallback so runtime consumers always resolve the
  real implementation under pytest coverage.【F:src/core/__init__.py†L14-L33】【F:src/core/sensory_organ.py†L1-L36】【F:tests/core/test_core_sensory_exports.py†L1-L22】

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
- Progress: Sensory package docs now spell out that the legacy `src.sensory.core`
  tree is retired, keeping only executable organ shims for backward
  compatibility, while the market-intelligence shim redirects callers to the
  enhanced dimensions so new integrations land on the supported surfaces.【F:src/sensory/__init__.py†L1-L11】【F:src/market_intelligence/dimensions/__init__.py†L1-L24】
- Progress: Integration regression now imports the enhanced sensory organs through their canonical namespaces, eliminating ad-hoc path hacks and exercising the fusion engine against the official adapters under pytest coverage.【F:src/sensory/tests/test_integration.py†L1-L550】
- Progress: Fusion engine and demos now expose an `analyze_market_understanding` coroutine (with a deprecated alias for legacy callers) so orchestration helpers, scripts, and docs share the understanding-first API while maintaining compatibility under regression coverage.【F:src/orchestration/enhanced_intelligence_engine.py†L209-L307】【F:scripts/minimal_sensory_demo.py†L117-L188】【F:src/sensory/README.md†L152-L480】
- Progress: Market intelligence namespace now re-exports canonical core and orchestration primitives via lazy shims, with regression coverage confirming legacy imports resolve to the sensory engines without noisy side effects.【src/market_intelligence/core/base.py:1】【src/market_intelligence/orchestration/enhanced_intelligence_engine.py:1】【tests/current/test_mi_to_sensory_forwarding_phase2.py:1】
- Progress: Namespace integrity guardrail now scans `src/`, `tools/`, and `scripts/` for banned legacy prefixes so regressions cannot reintroduce deprecated sensory or intelligence modules, and the demo/replay scripts import the canonical sensory namespaces under regression coverage.【F:tests/current/test_namespace_integrity.py†L1-L84】【F:scripts/sensory_demo.py†L1-L160】【F:scripts/replay_into_sensory.py†L1-L93】【F:scripts/minimal_sensory_demo.py†L1-L160】
- Progress: Coverage sweep tooling now targets `src/sensory/enhanced` and asserts coverage payloads list the canonical `src.understanding` files, keeping analysis triage and coverage reports aligned with the supported sensory/intelligence namespaces.【F:scripts/analysis/triage_batch6.py†L60-L76】【F:tests/tools/test_coverage_matrix.py†L1-L210】
- Progress: Sensory summary telemetry now constructs ranked Markdown/JSON
  rollups from integrated sensor payloads, preserves drift metadata, and
  publishes via the event-bus failover helper so dashboards receive resilient
  sensory status updates backed by regression coverage of runtime and fallback
  paths.【F:src/operations/sensory_summary.py†L1-L215】【F:tests/operations/test_sensory_summary.py†L1-L155】
- Progress: Integrated organ now exposes a metrics view with integrated
  strength/confidence snapshots, dimension posture, and drift metadata so the
  new metrics publisher can build dashboard payloads without rehydrating raw
  snapshots.【F:src/sensory/real_sensory_organ.py†L201-L233】【F:tests/sensory/test_real_sensory_organ.py†L130-L162】
- Progress: Executable dimension organs now publish HOW/ANOMALY readings and
  wrap WHAT/WHEN/WHY sensors, normalising frames, merging macro/narrative
  context, and forwarding calibrated lineage/quality metadata so runtime
  summaries and downstream telemetry inherit deterministic dimension payloads
  under guardrail coverage.【F:src/sensory/organs/dimensions/executable_organs.py†L1-L279】【F:tests/sensory/test_dimension_organs.py†L20-L279】
- Progress: Bootstrap runtime now instantiates the real sensory organ with a
  drift-configured history buffer, streams observations into cortex metrics,
  publishes summary/metrics/drift telemetry via the event-bus failover helper,
  and surfaces samples/audits via `status()` so supervisors inherit sensory
  posture and live telemetry during live-shadow runs under regression
  coverage.【F:src/runtime/bootstrap_runtime.py†L214-L492】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L120-L196】
- Progress: Real data slice pipeline persists CSV fixtures into Timescale through the managed backbone, hydrates the real sensory organ, and emits belief states so cortex drills can rehearse on live-formatted evidence with CLI and integration coverage documenting the workflow.【F:src/data_integration/real_data_slice.py†L95-L198】【F:tests/integration/test_real_data_slice_ingest.py†L11-L39】【F:tools/data_ingest/run_real_data_slice.py†L1-L125】

### Next (30–90 days)

- Implement HOW organ metrics (participation, liquidity, volatility) and ANOMALY
  detection tied to ingest feeds; record audit trails in runtime summaries.
- Extend WHY/WHAT/WHEN organs with calibrated signals sourced from Timescale once
  the institutional backbone is live.
- Progress: Bootstrap runtime publishes `telemetry.sensory.summary`,
  `.metrics`, and `.drift` payloads through the failover helper, caching the
  latest metrics for status reports while guardrail tests lock the telemetry
  path; downstream consumers/storage remain to be wired.【F:src/runtime/bootstrap_runtime.py†L214-L492】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L120-L196】

### Later (90+ days)

- Introduce adaptive thresholds linked to the evolution engine; capture lineage
  metadata for governance.
- Build operator runbooks for sensor outages and recalibration, including alert
  routing.
- Close dead-code findings by deleting redundant sensory templates once the new
  organs stabilise.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Definition of Done templates

**Executable Sensory Organs (WHAT/WHEN/HOW/WHY/ANOMALY).** The feature is complete when each dimension organ emits lineage, quality, and confidence metadata consumed by `src/sensory/real_sensory_organ.py`, regression suites `tests/sensory/test_dimension_organs.py` and `tests/sensory/test_primary_dimension_sensors.py` assert deterministic trend and anomaly fixtures, the fused payload runs end-to-end through `tests/runtime/test_bootstrap_runtime_sensory.py` without relying on placeholder stubs, and observability topics (`telemetry.sensory.summary`, `.metrics`, `.drift`) surface live updates in the dashboard snapshot captured for operations reviews.

**Belief & Regime Integration with Real Data.** This deliverable is done once Timescale/Kafka ingest populates the belief buffer via the real sensory organ, covariance safeguards in `src/understanding/belief.py` remain PSD during live replays, guardrail tests in `tests/understanding/test_belief_updates.py` exercise constant versus erratic feeds to assert calm→storm transitions, orchestration smoke tests log regime events with timestamps and volatility tags, and the decision diary records the resulting regime state for each iteration.

## Dependencies & coordination

- Requires ingest telemetry and risk enforcement to mature in parallel so sensors
  operate on trustworthy data and feed policy checks.
- Evolution engine uplift must expose catalogue snapshots and mutation logs to
  contextualise sensory-driven decisions.【F:docs/technical_debt_assessment.md†L95-L112】
