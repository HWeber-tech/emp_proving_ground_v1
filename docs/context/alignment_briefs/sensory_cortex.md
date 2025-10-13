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
- Institutional understanding engine now injects a bounded random source for
  jitter, normalises NaN/∞ samples, and reads bid/ask spreads so HOW signals
  respond to order-book shifts; regression fixtures assert bullish versus
  bearish frames produce divergent strengths with deterministic jitter for
  governance evidence.【F:src/sensory/enhanced/how_dimension.py†L19-L109】【F:src/sensory/how/how_sensor.py†L182-L200】【F:tests/sensory/test_primary_dimension_sensors.py†L146-L197】
- WHAT/WHEN/WHY sensors now attach structured `quality` metadata (source,
  timestamp, confidence) alongside lineage dictionaries, and regression tests
  assert the enriched payloads to keep downstream consumers aligned while real
  data feeds remain forthcoming. The WHAT sensor also offloads pattern synthesis
  into a worker thread when invoked inside an active asyncio loop so pattern
  payloads stay populated during runtime drills, with coverage capturing the
  async-path metadata.【F:src/sensory/what/what_sensor.py†L114-L224】【F:src/sensory/when/when_sensor.py†L147-L230】【F:src/sensory/why/why_sensor.py†L126-L238】【F:tests/sensory/test_primary_dimension_sensors.py†L1-L116】
- Real sensory organ fuses WHY/WHAT/WHEN/HOW/ANOMALY outputs, publishes
  telemetry snapshots with lineage metadata and bundled metrics payloads, and
  exposes audit/status helpers
  while still consuming synthetic data until institutional ingest arrives. It
  now wires an optional sensory lineage publisher that normalises HOW/ANOMALY
  payloads, keeps a bounded inspection history, emits lineage telemetry via
  runtime or fallback buses, and serialises per-dimension metadata plus numeric
  telemetry so responders and downstream metrics inherit audit-ready payloads
  under pytest coverage.【F:src/sensory/real_sensory_organ.py†L41-L520】【F:src/sensory/real_sensory_organ.py†L429-L495】【F:src/sensory/lineage_publisher.py†L1-L193】【F:tests/sensory/test_real_sensory_organ.py†L96-L183】【F:tests/sensory/test_real_sensory_organ.py†L113-L166】【F:tests/sensory/test_lineage.py†L85-L145】 Fallback signals delivered by the fused organ now clone per-dimension `quality` and `lineage` metadata into snapshots/metrics so diaries retain audit context and lineage publishing still fires when upstream sensors return empty frames.【src/sensory/real_sensory_organ.py:137】【src/sensory/real_sensory_organ.py:493】【tests/sensory/test_real_sensory_organ.py:113】
- Live diagnostics helper replays live frames through the fused organ, exports
  anomaly posture, drift telemetry, and WHY quality explanations for governance
  evidence packs, and ships with regression coverage to keep the reports
  deterministic.【F:src/sensory/monitoring/live_diagnostics.py†L1-L223】【F:tests/sensory/test_live_diagnostics.py†L1-L126】
- Executable HOW/ANOMALY sensory organs now wrap the canonical sensors, normalise
  market frames or sequences, maintain calibrated windows, and emit structured
  lineage/telemetry payloads under guardrail regression coverage, though they
  still consume synthetic payloads until ingest is live. Latest hardening
  synthesises missing `quality` envelopes (source, timestamp, confidence,
  strength) when upstream payloads omit them and exports the WHAT/WHEN/WHY organ
  entry points from the package so downstream imports stay canonical, with
  regression coverage locking the auto-populated metadata.【F:src/sensory/organs/dimensions/executable_organs.py†L151-L214】【F:src/sensory/organs/__init__.py†L5-L18】【F:tests/sensory/test_dimension_organs.py†L365-L377】
- Sensory dimension modules now lazily import executable organs to avoid
  bootstrap-era cyclical imports, keeping paper simulation and runtime helpers
  import-safe even when sensory packages are initialised independently under
  regression coverage.【F:src/sensory/organs/dimensions/__init__.py†L1-L41】【F:src/sensory/dimensions/__init__.py†L13-L41】
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
  tree is retired, and the `market_intelligence` shim has been removed so new
  integrations land directly on the supported sensory surfaces.【F:src/sensory/__init__.py†L1-L11】【F:scripts/cleanup/duplicate_map.py†L60-L91】
- Progress: Integration regression now imports the enhanced sensory organs through their canonical namespaces, eliminating ad-hoc path hacks and exercising the fusion engine against the official adapters under pytest coverage.【F:src/sensory/tests/test_integration.py†L1-L550】
- Progress: Fusion engine and demos now expose an `analyze_market_understanding` coroutine, and the retired `enhanced_intelligence_engine` shim now raises a targeted `ModuleNotFoundError`, forcing callers onto the understanding namespace while regression suites and docs exercise the canonical API.【F:src/orchestration/enhanced_understanding_engine.py†L216-L318】【F:src/orchestration/enhanced_intelligence_engine.py†L1-L17】【F:src/sensory/tests/test_integration.py†L122-L150】【F:src/sensory/README.md†L349-L420】
- Progress: The retired `src.sensory.organs.dimensions.macro_intelligence` shim now raises a guided `ModuleNotFoundError` that points callers at the enhanced WHY engine, with the shim-removal guardrail ensuring macro integrations adopt the canonical sensory namespace instead of reviving legacy surfaces.【F:src/sensory/organs/dimensions/macro_intelligence.py†L1-L14】【F:tests/thinking/test_shim_import_failures.py†L8-L25】
- Progress: The legacy `market_intelligence` namespace has been fully retired; regression tests now exercise the canonical sensory modules directly to prevent drift back to removed shims.【F:tests/current/test_mi_to_sensory_forwarding.py†L1-L60】【F:tests/current/test_mi_to_sensory_forwarding_phase2.py†L1-L90】
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
- Progress: Real data slice pipeline persists CSV fixtures into Timescale through the managed backbone, hydrates the real sensory organ, and emits belief states so cortex drills can rehearse on live-formatted evidence while the operational-backbone CLI surfaces belief/regime snapshots, understanding decisions, and ingest-failure telemetry under integration coverage.【F:src/data_integration/real_data_slice.py†L95-L198】【F:tests/integration/test_real_data_slice_ingest.py†L11-L39】【F:src/data_foundation/pipelines/operational_backbone.py†L82-L366】【F:tools/data_ingest/run_operational_backbone.py†L1-L378】【F:tests/integration/test_operational_backbone_pipeline.py†L198-L295】【F:tests/tools/test_run_operational_backbone.py†L17-L105】
- Progress: Guardrail coverage now replays a volatility spike through `RealSensoryOrgan`, the HOW engine, and the belief emitter, asserting drift summaries, anomaly payloads, and belief posterior features all materialise so the sensory-to-understanding rehearsal path stays intact on canonical surfaces.【F:tests/understanding/test_sensory_belief_pipeline.py†L84-L139】【F:src/sensory/real_sensory_organ.py†L190-L235】【F:src/understanding/belief.py†L649-L706】
- Progress: Belief buffer now maps dimension-specific sensory extras—`WHAT_last_close`, HOW liquidity participation metrics, and WHEN session/news/gamma hints—into the posterior feature vector while the refreshed golden snapshot and pipeline guardrail lock the expanded feature order so downstream analytics inherit the richer audit trail.【F:src/understanding/belief.py†L172-L255】【F:tests/understanding/golden/belief_snapshot.json†L1-L120】【F:tests/understanding/test_sensory_belief_pipeline.py†L130-L144】
- Progress: Live belief manager now applies hysteresis before expanding regime thresholds, publishes belief/regime telemetry for cached ingest snapshots, and guardrail suites replay volatility spikes to prove monotonic scaling with stable routing outputs.【F:src/understanding/live_belief_manager.py†L18-L168】【F:tests/understanding/test_live_belief_manager.py†L210-L296】【F:tests/integration/test_operational_backbone_pipeline.py†L198-L295】
- Progress: Belief/regime calibration helpers now derive Hebbian learning rates, variance caps, and calm/storm thresholds from historical EURUSD series, returning diagnostics and constructing calibrated belief/regime components that pass calm-versus-storm regression drills.【F:src/understanding/belief_regime_calibrator.py†L1-L175】【F:tests/operations/test_belief_regime_calibrator.py†L34-L136】
- Progress: Belief buffer now records posterior covariance trace, condition ratios, and eigenvalue extrema, with guardrail-marked EURUSD slice suites asserting PSD eigenvalues, bounded maxima, non-negative minima, and telemetry emission so the real-data stability evidence remains current.【F:src/understanding/belief.py†L316-L400】【F:tests/understanding/test_belief_real_data_integration.py†L91-L163】【F:tests/data_integration/test_real_data_slice_belief.py†L48-L140】
- Progress: Belief assimilation now rejects NaN/∞ sensory strengths before Hebbian updates and clamps regime scaling to finite values, with a guardrail replay that injects pathological telemetry to prove emitted observations, covariance eigenvalues, and regime diagnostics remain bounded.【F:src/understanding/belief.py†L167-L333】【F:src/understanding/belief_real_data_utils.py†L72-L93】【F:tests/understanding/test_belief_real_data_integration.py†L179-L224】
- Progress: Live belief manager now bridges real sensory snapshots into calibrated belief/regime components, auto-scales threshold posture when volatility spikes, and ships guardrail tests that replay calm versus storm EURUSD frames to prove PSD covariance and regime transitions hold on live-formatted data.【F:src/understanding/live_belief_manager.py†L1-L170】【F:src/understanding/belief_real_data_utils.py†L18-L89】【F:src/data_integration/real_data_slice.py†L213-L299】【F:tests/understanding/test_live_belief_manager.py†L130-L199】

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

**Belief & Regime Integration with Real Data.** This deliverable is done once Timescale/Kafka ingest populates the belief buffer via the real sensory organ, covariance safeguards in `src/understanding/belief.py` remain PSD during live replays, guardrail tests in `tests/understanding/test_belief_real_data_integration.py` and `tests/data_integration/test_real_data_slice_belief.py` exercise calm versus storm feeds to assert eigenvalue bounds, telemetry metadata, and regime transitions, orchestration smoke tests log regime events with timestamps and volatility tags, and the decision diary records the resulting regime state for each iteration.

## Dependencies & coordination

- Requires ingest telemetry and risk enforcement to mature in parallel so sensors
  operate on trustworthy data and feed policy checks.
- Evolution engine uplift must expose catalogue snapshots and mutation logs to
  contextualise sensory-driven decisions.【F:docs/technical_debt_assessment.md†L95-L112】
