# Alignment brief – Sensory cortex & anomaly telemetry

**Why this brief exists:** The concept blueprint calls for a 4D+1 sensory cortex that fuses market microstructure, macro
signals, anomaly detection, and confidence scoring, yet the modernization backlog lacked a dedicated context pack that
links those promises to the shipped HOW/WHEN/WHY/ANOMALY organs, drift telemetry, and runtime summaries. This brief keeps
execution, validation, and documentation tethered to the same narrative so follow-up work expands the cortex without
slipping away from the concept intent.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L318-L344】【F:docs/roadmap.md†L27-L47】

## Concept promise

- The encyclopedia defines the sensory cortex as a multi-dimensional perception layer that blends behavioural, macro, and
  anomaly insights with confidence reporting, establishing the north star for the organ interfaces.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L318-L344】
- The roadmap’s gap table reiterates that all five organs must ship together with calibrated feeds, pushing teams to close
  the historical HOW/ANOMALY deficiencies and surface observability for operators.【F:docs/roadmap.md†L27-L47】
- Current 60-day outcomes expect drift telemetry, gamma-aware timing, and yield-curve analytics to land with pytest coverage
  so regression evidence accompanies every sensory upgrade.【F:docs/roadmap.md†L126-L145】

## Reality snapshot (September 2025)

- `HowSensor` bridges the institutional intelligence engine into the legacy pipeline, emitting liquidity, participation, and
  volatility telemetry with markdown-ready metadata, while regression tests assert the audit payloads.【F:src/sensory/how/how_sensor.py†L1-L114】【F:tests/sensory/test_how_anomaly_sensors.py†L1-L80】
- `AnomalySensor` now supports sequence and market-data modes, records dispersion/baseline context, and publishes calibrated
  thresholds validated under pytest so spikes surface in telemetry.【F:src/sensory/anomaly/anomaly_sensor.py†L1-L120】【F:tests/sensory/test_how_anomaly_sensors.py†L55-L80】
- `WhenSensor` blends session overlap, macro-event proximity, and `GammaExposureAnalyzer` summaries into a weighted
  readiness score, exposing gamma metadata to operators and tests alike.【F:src/sensory/when/when_sensor.py†L1-L200】【F:src/sensory/when/gamma_exposure.py†L1-L200】【F:tests/sensory/test_when_gamma.py†L1-L70】
- `WhySensor` fuses macro bias with `YieldSlopeTracker` snapshots so inversion risk and slope dynamics appear in the
  resulting signals, backed by focused regression coverage.【F:src/sensory/why/why_sensor.py†L1-L112】【F:tests/sensory/test_why_yield.py†L19-L85】
- Sensory audit trails stream through `evaluate_sensory_drift`, producing reusable snapshots and markdown that the runtime
  builder publishes, the professional app stores, and the summary exposes under dedicated tests.【F:src/operations/sensory_drift.py†L1-L200】【F:src/runtime/runtime_builder.py†L900-L1064】【F:src/runtime/predator_app.py†L524-L1064】【F:tests/runtime/test_professional_app_timescale.py†L1188-L1224】

## Gap map

| Concept excerpt | Observable gap | Impact |
| --- | --- | --- |
| Multi-dimensional sensory cortex integrates behavioural, macro, and anomaly feeds with confidence metrics.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L318-L344】 | Default runtime records individual organ outputs but lacks a consolidated “sensory posture” summary across runs. | Operators still piece together multiple blocks to understand overall sensory health, complicating drift analysis during incidents. |
| Roadmap commits to calibrated HOW/ANOMALY organs plus drift telemetry and gamma-aware timing.【F:docs/roadmap.md†L27-L145】 | Drift telemetry is present but portfolio snapshots and CI health decks have not yet embedded the cortex evidence alongside ingest readiness. | Documentation trails fall behind runtime reality, reducing discoverability for reviewers and stakeholders. |
| Later roadmap steps call for live-paper experiments and automated tuning loops driven by sensory telemetry.【F:docs/roadmap.md†L268-L335】 | Evolution experiments consume sensory data but there is no automation wiring that feeds sensory drift back into tuning decisions. | Opportunity to close the feedback loop remains untapped, delaying adaptive behaviour promised for Tier‑2. |

## Delivery plan

### Now (30-day outlook)

1. **Publish cortex evidence in status artefacts** – Update the CI health snapshot and roadmap portfolio table to surface
   sensory drift, gamma posture, and yield telemetry so reviewers see the shipped organs without running code.【F:src/runtime/predator_app.py†L524-L1064】【F:docs/status/ci_health.md†L97-L170】
2. **Consolidate sensory posture** – Add a runtime summary helper (and pytest coverage) that aggregates HOW/WHEN/WHY/ANOMALY
   strengths plus drift severity into a single block for operators rehearsing incidents.【F:src/runtime/predator_app.py†L524-L1064】【F:tests/runtime/test_professional_app_timescale.py†L1188-L1224】
3. **Capture cortex metrics in roadmap CLI** – Extend `tools.roadmap.snapshot` with a requirement that imports
   `operations.sensory_drift` so the automation fails fast if the telemetry surface regresses, with regression coverage locking
   the full sensory/evolution guardrail set so portfolio snapshots stay honest. The guardrail now also requires the evolution
   experiment and tuning evaluators so feedback-loop telemetry remains visible in the portfolio snapshot.【F:tools/roadmap/snapshot.py†L188-L200】【F:tests/tools/test_roadmap_snapshot.py†L57-L104】

### Next (60-day outlook)

1. **Feedback loop experiments** – Feed sensory drift deltas into the evolution experiment evaluator so ROI telemetry records
   how sensory degradation influences strategy trials.【F:src/operations/evolution_experiments.py†L1-L240】【F:src/runtime/runtime_builder.py†L2059-L2116】
2. **Alternative data hooks** – Prototype Timescale readers for news/sentiment sources referenced in the concept doc and map
   them into HOW/WHY metadata to close the “behavioural feed” promise.【F:src/data_foundation/persist/timescale_reader.py†L1-L200】【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L318-L344】
3. **Operator playbooks** – Extend the sensory runbooks with incident response triggers (e.g., gamma flip risk) so the new
   telemetry blocks come with actionable responder guidance.【F:docs/operations/runbooks/README.md†L1-L19】

### Later (90-day+ considerations)

- Integrate live exchange data (depth, volatility surfaces) and calibrate the sensors against intraday ground truth using
  Spark stress datasets once the batch drills move to production.【F:src/operations/spark_stress.py†L1-L166】
- Automate cortex regression dashboards that trend drift severity over time, linking to Timescale-backed audit tables for
  compliance reviews.【F:src/data_foundation/persist/timescale.py†L1-L720】【F:docs/status/ci_health.md†L120-L210】

## Validation hooks

- **Pytest suite** – Maintain the HOW/ANOMALY/WHEN/WHY tests and ensure new data sources land with regression coverage so
  CI continues to validate signal composition.【F:tests/sensory/test_how_anomaly_sensors.py†L1-L80】【F:tests/sensory/test_when_gamma.py†L1-L70】【F:tests/sensory/test_why_yield.py†L19-L85】
- **Runtime summary drill** – Extend `tests/runtime/test_professional_app_timescale.py` with fixtures that record sensory drift
  snapshots and assert the consolidated posture block renders as expected.【F:tests/runtime/test_professional_app_timescale.py†L1188-L1224】
- **Telemetry subscription** – Rehearse `telemetry.sensory.drift` subscriptions through the event bus and Kafka bridge so
  operators can confirm ingestion of cortex telemetry without manual inspection.【F:src/runtime/runtime_builder.py†L900-L1064】【F:src/data_foundation/streaming/kafka_stream.py†L1700-L1919】

## Open questions

1. How should we normalise alternative data feeds (news, sentiment, flow analytics) so they blend cleanly with the existing
   cortex without diluting signal quality?【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L318-L344】
2. What governance checkpoints are required before sensory drift metrics influence automated evolution tuning loops?【F:docs/roadmap.md†L268-L335】
3. Which dashboards (CI, operations, portfolio) should surface aggregate sensory posture, and how do we keep them in lockstep
   with runtime telemetry exports?【F:docs/status/ci_health.md†L97-L170】【F:tools/roadmap/snapshot.py†L59-L147】
