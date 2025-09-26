# Modernization Roadmap

This roadmap orchestrates the remaining modernization work for the EMP Professional
Predator codebase and reframes the backlog through the lens of the
`EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` concept blueprint. Pair it with the living
[Technical Debt Assessment](technical_debt_assessment.md) when planning discovery
spikes, execution tickets, or milestone reviews.

## Concept alignment context pack

Use these artefacts as the substrate for context-aware planning, ticket grooming,
and code reviews:

- **Concept blueprint** – `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` (truth-first draft of
  the five-layer architecture, compliance posture, and commercialization journeys).
- **Reality snapshot** – [`system_validation_report.json`](../system_validation_report.json)
  and [`docs/technical_debt_assessment.md`](technical_debt_assessment.md) for the
  current engineering baseline.
- **Delta log** – This roadmap plus the quarterly architecture notes capture how the
  implementation is evolving toward the concept claims.

### Promises vs. implementation gaps

| Concept promise | Current state | Roadmap response |
| --- | --- | --- |
| Tiered data backbone with TimescaleDB, Redis, Kafka, and Spark orchestration once we outgrow the 1 GB bootstrap footprint. | Tier‑0 relies on DuckDB/CSV helpers; Yahoo ingest returns placeholders and there is no orchestration for professional data tiers. | Stand up a production-ready ingest slice with TimescaleDB persistence, Redis caching, and Kafka streaming, then wire orchestration that can flip between bootstrap and institutional tiers. |
| 4D+1 sensory cortex delivering superhuman perception with WHY/HOW/WHAT/WHEN/ANOMALY working together. | Only WHY/WHAT/WHEN have lightweight heuristics, ANOMALY is a stub, and HOW is absent from the default runtime. | Build the HOW organ, connect anomaly detection, and upgrade existing sensors with calibrated macro/order-flow data feeds. |
| Evolutionary intelligence that adapts live strategies through genetic evolution and meta-learning. | Tier‑2 evolution mode throws `NotImplementedError`; genomes are stubs and trading decisions remain static thresholds. | Implement population management, realistic genomes, and fitness evaluation; make Tier‑2 executable behind feature flags. |
| Institutional execution, risk, and compliance with OMS/EMS connectivity plus regulatory workflows (MiFID II, Dodd‑Frank, KYC/AML). | Execution is an in-memory simulator, risk checks are minimal, and compliance layers are no-ops. | Deliver FIX/EMS bridges, expand risk controls, and prototype compliance workflows with audit persistence. |
| Operational readiness with monitoring, alerting, security, and disaster recovery suitable for higher tiers. | Bootstrap control center aggregates telemetry but broader ops/security tooling is missing. | Design the monitoring stack, access controls, and backup routines aligned with the concept doc’s operational maturity targets. |
| Commercial roadmap that validates the €250 → institutional journey and 95 % cost savings. | Bootstrap monitor assumes a $100 000 paper account; there is no empirical ROI model or cost tracking. | Instrument cost/fee models, track ROI experiments, and align marketing claims with evidence. |

> **Context engineering tip:** When drafting epics or writing code, pull the relevant
> concept excerpt into the issue description and link the gap table row. This keeps
> discussions anchored to the documented intent.

## Execution rhythm

- **Stage work into reviewable tickets** – Translate bullets into 1–2 day tasks with a
  Definition of Done, clear owner, and explicit validation steps.
- **Run a Now / Next / Later board** – Groom items as soon as they move to "Later" so
  the next engineer can start without another planning meeting.
- **Time-box discovery** – Use 4–8 hour spike tickets when investigation is required
  and close them with a written summary so execution inherits the findings.
- **Weekly sync** – Spend 15 minutes each Friday to capture status, blockers, and any
  resequencing. Update this document and the tracking board immediately afterwards.
- **Keep telemetry fresh** – Refresh formatter progress, coverage deltas, and alerting
  status in [`docs/status/ci_health.md`](status/ci_health.md) so dashboards mirror the
  roadmap.

## Way forward

### 30-day outcomes (Now)
- Draft alignment briefs for the five major gap areas (data backbone, sensory
  cortex, evolution engine, institutional risk/compliance, operational readiness)
  that translate encyclopedia promises into actionable epics and acceptance
  criteria.
  - Data backbone brief: [`docs/context/alignment_briefs/institutional_data_backbone.md`](context/alignment_briefs/institutional_data_backbone.md).
  - Sensory cortex brief: [`docs/context/alignment_briefs/sensory_cortex.md`](context/alignment_briefs/sensory_cortex.md).
  - Evolution brief: [`docs/context/alignment_briefs/evolution_engine.md`](context/alignment_briefs/evolution_engine.md).
  - Risk & compliance brief: [`docs/context/alignment_briefs/institutional_risk_compliance.md`](context/alignment_briefs/institutional_risk_compliance.md).
  - Operational readiness brief: [`docs/context/alignment_briefs/operational_readiness.md`](context/alignment_briefs/operational_readiness.md).
- Ship the “concept context pack” – issue templates, pull request checklists, and
  architecture notes that quote `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` sections next
  to current code references so contributors inherit the same narrative.
- Prototype a TimescaleDB-backed ingest slice fed by the existing Yahoo bootstrap
  downloader to exercise migrations, connection pooling, and state-store wiring.
- Add scaffolding tests or notebooks that demonstrate how the new ingest slice will
  be queried by sensors and risk modules once Redis/Kafka are introduced.
  - `TimescaleReader` smoke tests (`tests/data_foundation/test_timescale_ingest.py`) now document the daily, intraday, and macro
    query paths against the institutional tables, keeping the read-side contract visible while Redis/Kafka land.
- Stream ingest health telemetry into the runtime bus so CI and operators inherit freshness and completeness checks as soon as the Timescale pipelines execute.
  - `evaluate_ingest_health` transforms orchestrator results into health reports, publishes them on the `telemetry.ingest.health` channel, and mirrors the payloads to Kafka via `KafkaIngestHealthPublisher` so streaming consumers receive the same telemetry snapshot.【F:src/data_foundation/ingest/health.py†L1-L197】【F:src/runtime/runtime_builder.py†L445-L653】【F:src/data_foundation/streaming/kafka_stream.py†L338-L470】【F:tests/data_foundation/test_kafka_stream.py†L74-L190】
- `summarise_ingest_metrics` now distils ingest outcomes into a `telemetry.ingest.metrics` payload so dashboards and CI health reports can track total rows, freshness, and symbol coverage per dimension alongside the health grades.【F:src/data_foundation/ingest/metrics.py†L1-L88】【F:src/runtime/runtime_builder.py†L577-L593】【F:tests/data_foundation/test_ingest_metrics.py†L1-L45】
  - `KafkaIngestMetricsPublisher` mirrors the same metrics snapshots to Kafka when institutional brokers/topics are configured, keeping event-bus subscribers and external streaming consumers aligned on ingest totals and freshness.【F:src/data_foundation/streaming/kafka_stream.py†L1022-L1135】【F:src/runtime/runtime_builder.py†L752-L843】【F:tests/data_foundation/test_kafka_stream.py†L360-L520】
  - `evaluate_ingest_quality` grades coverage and freshness into a reusable score published on `telemetry.ingest.quality`, giving the roadmap’s CI health report visibility into density gaps, while `KafkaIngestQualityPublisher` mirrors the payloads for downstream telemetry stacks.【F:src/data_foundation/ingest/quality.py†L1-L208】【F:src/runtime/runtime_builder.py†L649-L706】【F:src/data_foundation/streaming/kafka_stream.py†L1126-L1287】【F:tests/data_foundation/test_ingest_quality.py†L1-L60】【F:tests/data_foundation/test_kafka_stream.py†L200-L310】
  - `build_ingest_observability_snapshot` fuses metrics, health findings, recovery recommendations, and failover decisions into a single `telemetry.ingest.observability` payload logged by the runtime so CI dashboards inherit one authoritative ingest summary per run.【F:src/data_foundation/ingest/observability.py†L1-L211】【F:src/runtime/runtime_builder.py†L624-L635】【F:tests/data_foundation/test_ingest_observability.py†L1-L110】
  - `execute_spark_export_plan` exports Timescale slices into Spark-friendly CSV/JSONL datasets with optional partitioning, writes per-job manifests, and the runtime publishes `telemetry.ingest.spark_exports` while recording the snapshot in professional summaries for operators.【F:src/data_foundation/batch/spark_export.py†L1-L233】【F:src/runtime/runtime_builder.py†L657-L866】【F:src/runtime/predator_app.py†L1-L520】【F:tests/data_foundation/test_spark_export.py†L1-L129】【F:tests/data_foundation/test_ingest_journal.py†L360-L438】【F:tests/runtime/test_professional_app_timescale.py†L1-L620】
  - `evaluate_ingest_trends` analyses Timescale ingest journal history into `telemetry.ingest.trends` snapshots so dashboards and runtime summaries surface momentum, row-count drops, and freshness regressions alongside health, metrics, and quality feeds.【F:src/operations/ingest_trends.py†L1-L240】【F:src/runtime/runtime_builder.py†L900-L1090】【F:src/runtime/predator_app.py†L680-L735】【F:tests/operations/test_ingest_trends.py†L1-L90】【F:tests/runtime/test_professional_app_timescale.py†L240-L330】
  - `evaluate_data_backbone_readiness` now records Spark export snapshots as a readiness component so operators see batch export coverage alongside ingest health, quality, and failover telemetry.【F:src/operations/data_backbone.py†L15-L360】【F:src/runtime/runtime_builder.py†L657-L1160】【F:tests/operations/test_data_backbone.py†L1-L150】
  - `evaluate_data_retention` inspects Timescale daily, intraday, and macro tables to confirm institutional retention windows, publishes `telemetry.data_backbone.retention`, and records the markdown snapshot inside professional runtime summaries so operators can audit historical coverage alongside other backbone feeds.【F:src/operations/retention.py†L1-L192】【F:src/runtime/runtime_builder.py†L1060-L1210】【F:src/runtime/predator_app.py†L150-L360】【F:tests/operations/test_data_retention.py†L1-L118】【F:tests/runtime/test_professional_app_timescale.py†L640-L700】
  - `evaluate_cache_health` grades Redis cache configuration and hit/miss telemetry into a `telemetry.cache.health` snapshot so operators can confirm namespaces, hit rates, and eviction pressure directly from runtime logs and summaries.【F:src/operations/cache_health.py†L1-L191】【F:src/runtime/runtime_builder.py†L608-L741】【F:tests/runtime/test_professional_app_timescale.py†L300-L346】
  - `evaluate_data_backbone_readiness` combines ingest health, quality, recovery, backup posture, Redis/Kafka wiring, and scheduler state into a `telemetry.data_backbone.readiness` snapshot recorded by the runtime and published on the event bus so operators inherit a single institutional readiness surface.【F:src/operations/data_backbone.py†L1-L320】【F:src/runtime/runtime_builder.py†L600-L900】【F:src/runtime/predator_app.py†L200-L430】【F:tests/runtime/test_runtime_builder.py†L160-L280】【F:tests/runtime/test_professional_app_timescale.py†L395-L460】
  - `evaluate_kafka_readiness` folds connection summaries, topic provisioning reports, publisher availability, and consumer lag telemetry into `telemetry.kafka.readiness`, and the runtime builder now publishes plus records the snapshot so professional summaries display streaming posture next to ingest and cross-region readiness feeds.【F:src/operations/kafka_readiness.py†L1-L213】【F:src/runtime/runtime_builder.py†L600-L930】【F:src/runtime/predator_app.py†L200-L400】【F:tests/operations/test_kafka_readiness.py†L1-L97】【F:tests/runtime/test_runtime_builder.py†L665-L782】【F:tests/runtime/test_professional_app_timescale.py†L1116-L1160】
- `evaluate_data_backbone_validation` cross-checks Timescale connection settings, Redis/Kafka expectations, and scheduler telemetry to publish `telemetry.data_backbone.validation`, log markdown summaries, and surface the latest validation snapshot in the professional runtime so operators can confirm institutional toggles before ingest executes.【F:src/operations/data_backbone.py†L120-L320】【F:src/runtime/runtime_builder.py†L650-L760】【F:src/runtime/predator_app.py†L150-L370】【F:tests/operations/test_data_backbone.py†L120-L220】【F:tests/runtime/test_runtime_builder.py†L180-L280】【F:tests/runtime/test_professional_app_timescale.py†L395-L460】
- `tools/telemetry/export_data_backbone_snapshots.py` packages readiness, validation, retention, ingest trend/scheduler, and Kafka posture blocks from the professional runtime summary into a JSON feed so Grafana/DataDog dashboards and runbooks ingest the backbone state without scraping Markdown. pytest covers the exporter success path plus missing-section guardrails.【F:tools/telemetry/export_data_backbone_snapshots.py†L1-L147】【F:tests/tools/test_data_backbone_export.py†L1-L74】
  - Timescale ingest now short-circuits when validation fails, emitting failure snapshots, publishing the degraded readiness to the event bus, and invoking the DuckDB fallback so institutional runs never proceed with a misconfigured backbone.【F:src/runtime/runtime_builder.py†L700-L780】【F:tests/data_foundation/test_ingest_journal.py†L300-L374】
  - `evaluate_professional_readiness` fuses the data backbone snapshot, backup posture, ingest SLOs, failover decisions, and recovery recommendations into `telemetry.operational.readiness`, recording the latest snapshot on the runtime so dashboards inherit a single professional-tier readiness surface alongside event-bus telemetry.【F:src/operations/professional_readiness.py†L1-L190】【F:src/runtime/runtime_builder.py†L360-L873】【F:src/runtime/predator_app.py†L1-L470】【F:tests/operations/test_professional_readiness.py†L1-L120】【F:tests/runtime/test_professional_app_timescale.py†L1-L320】
  - `TimescaleIngestScheduler` and the runtime wiring now replay the ingest plan on configurable intervals, using extras to turn on background runs, jitter, and failure guardrails so institutional tiers keep Timescale data fresh without manual triggers.【F:src/data_foundation/ingest/scheduler.py†L1-L137】【F:src/data_foundation/ingest/configuration.py†L1-L210】【F:src/runtime/runtime_builder.py†L845-L869】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L78】【F:tests/data_foundation/test_timescale_config.py†L120-L170】
  - `TimescaleIngestJournal` persists every orchestrated run (status, rows, freshness, metadata) into the shared Timescale cluster so operators and dashboards can audit ingest history directly from the institutional datastore, and `_record_ingest_journal` wires the runtime to log each run after health evaluation.【F:src/data_foundation/persist/timescale.py†L1-L720】【F:src/runtime/runtime_builder.py†L347-L437】【F:tests/data_foundation/test_ingest_journal.py†L1-L200】
  - `plan_ingest_recovery` now evaluates degraded ingest runs, extends lookbacks for missing symbols, and replays targeted slices before failover kicks in. Recovery attempts merge into the aggregated ingest metrics, health payloads, and Timescale journal metadata so operators can track automatic retries directly from telemetry.【F:src/data_foundation/ingest/recovery.py†L1-L152】【F:src/runtime/runtime_builder.py†L484-L570】【F:tests/data_foundation/test_ingest_recovery.py†L1-L95】【F:tests/data_foundation/test_ingest_journal.py†L120-L197】
  - `ProfessionalPredatorApp.summary()` now surfaces the most recent ingest journal entries and per-dimension statuses so operators can review institutional ingest health directly from the runtime without shell access.【F:src/runtime/predator_app.py†L220-L317】【F:tests/runtime/test_professional_app_timescale.py†L120-L196】
- Adjust public-facing docs (README, roadmap preface, launch deck) to clarify that
  v2.3 is a concept draft with hypotheses awaiting validation.

### 60-day outcomes (Next)
- Land the first production-grade ingest vertical: TimescaleDB persistence, Redis
  caching for hot symbols, and a Kafka topic that mirrors intraday updates into the
  runtime event bus.
  - Redis cache plumbing now powers `TimescaleQueryCache`, serialising reader results into ManagedRedis namespaces and wiring the professional runtime’s Timescale connectors through the shared cache; pytest covers cache hits, TTL expirations, and runtime integration for institutional runs. The [Redis cache outage runbook](operations/runbooks/redis_cache_outage.md) gives on-call engineers the recovery playbook that matches the telemetry blocks exposed in the runtime summary.【F:src/data_foundation/cache/timescale_query_cache.py†L1-L190】【F:src/data_foundation/fabric/timescale_connector.py†L1-L160】【F:src/runtime/predator_app.py†L320-L520】【F:tests/data_foundation/test_timescale_cache.py†L1-L160】【F:docs/operations/runbooks/redis_cache_outage.md†L1-L60】
  - Timescale market-data connectors now enrich each fetch with macro bias and confidence derived from stored macro events via `TimescaleMacroEventService`, giving sensors institutional macro telemetry alongside price data.【F:src/data_foundation/services/macro_events.py†L1-L226】【F:src/data_foundation/fabric/timescale_connector.py†L1-L164】【F:src/runtime/predator_app.py†L420-L470】【F:tests/data_foundation/test_timescale_connectors.py†L1-L150】
- Kafka streaming scaffolding now publishes ingest metadata via
    `KafkaIngestEventPublisher`, while `EventBusIngestPublisher` mirrors the same
    payloads onto `telemetry.ingest` so runtimes can subscribe before Kafka
    consumers exist.【F:src/data_foundation/streaming/kafka_stream.py†L1-L225】【F:src/data_foundation/ingest/telemetry.py†L1-L168】【F:tests/data_foundation/test_timescale_ingest.py†L180-L256】【F:tests/data_foundation/test_ingest_publishers.py†L1-L132】
  - `KafkaIngestEventConsumer` now replays configured ingest topics onto the runtime
    event bus when institutional mode is active, giving Tier‑1 deployments Kafka
    telemetry without bespoke bridges and exercising the background task wiring in
    the professional runtime. Offset management is baked in with
    `KAFKA_INGEST_CONSUMER_COMMIT_ON_PUBLISH` and
    `KAFKA_INGEST_CONSUMER_COMMIT_ASYNC` toggles so operators can disable
    auto-commit and rely on the bridge for manual acknowledgements while still
    mirroring events onto the runtime bus. The [Kafka ingest offset recovery runbook](operations/runbooks/kafka_ingest_offset_recovery.md) captures the manual procedure for advancing group offsets and verifying lag telemetry when the consumer falls behind.【F:src/data_foundation/streaming/kafka_stream.py†L1452-L1756】【F:tests/data_foundation/test_kafka_stream.py†L320-L660】【F:src/runtime/predator_app.py†L500-L540】【F:docs/operations/runbooks/kafka_ingest_offset_recovery.md†L1-L66】
    The consumer can also emit `telemetry.kafka.lag` snapshots by invoking
    `capture_consumer_lag`, publishing per-partition offsets, aggregated lag
    totals, and metadata on a configurable cadence so Kafka lag observability
    lands alongside ingest payloads; pytest covers the probe contract and interval
    guardrails.【F:src/data_foundation/streaming/kafka_stream.py†L259-L415】【F:src/data_foundation/streaming/kafka_stream.py†L1700-L1919】【F:tests/data_foundation/test_kafka_stream.py†L453-L564】
  - `backfill_ingest_dimension_to_kafka` replays Timescale snapshots into configured ingest topics with explicit backfill metadata so fresh environments inherit historical ingest telemetry, and pytest coverage documents the replay payload alongside topic validation.【F:src/data_foundation/streaming/kafka_stream.py†L460-L620】【F:tests/data_foundation/test_kafka_stream.py†L340-L420】
  - `KafkaTopicProvisioner` now auto-creates ingest topics when `KAFKA_INGEST_AUTO_CREATE_TOPICS` is set, logging summaries in the runtime, capturing metadata via `build_institutional_ingest_config`, and exercising the provisioning logic with admin-client fakes in pytest. The new offset runbook closes the outstanding operational guidance for this slice.【F:src/data_foundation/streaming/kafka_stream.py†L226-L470】【F:src/data_foundation/ingest/configuration.py†L1-L210】【F:src/runtime/runtime_builder.py†L752-L806】【F:tests/data_foundation/test_kafka_stream.py†L60-L360】【F:tests/data_foundation/test_timescale_config.py†L1-L160】【F:docs/operations/runbooks/kafka_ingest_offset_recovery.md†L1-L66】
- `decide_ingest_failover` inspects ingest health reports, publishes `telemetry.ingest.failover` events, and triggers the DuckDB bootstrap replay when required Timescale slices fail so institutional runs inherit a documented rollback path.【F:src/data_foundation/ingest/failover.py†L1-L120】【F:src/runtime/runtime_builder.py†L618-L650】【F:tests/data_foundation/test_ingest_failover.py†L1-L120】
  - `execute_failover_drill` mutates recent ingest results to simulate Timescale outages, confirms failover policies, publishes `telemetry.ingest.failover_drill`, and records Markdown summaries on the professional runtime so operators can rehearse the DuckDB fallback without manual intervention.【F:src/operations/failover_drill.py†L1-L213】【F:src/runtime/runtime_builder.py†L1015-L1072】【F:src/runtime/predator_app.py†L145-L180】【F:tests/operations/test_failover_drill.py†L1-L88】【F:tests/runtime/test_professional_app_timescale.py†L610-L650】
  - `evaluate_cross_region_failover` compares primary ingest runs with replica history, grades scheduler readiness, and publishes `telemetry.ingest.cross_region_failover` snapshots so operators can rehearse cross-region cutover directly from runtime summaries and event bus telemetry.【F:src/operations/cross_region_failover.py†L1-L238】【F:src/runtime/runtime_builder.py†L900-L1165】【F:src/runtime/predator_app.py†L180-L340】【F:tests/operations/test_cross_region_failover.py†L1-L130】【F:tests/runtime/test_runtime_builder.py†L250-L360】
- Sprint brief translating this outcome into reviewable tickets: [`docs/context/sprint_briefs/next_sprint_redis_kafka_toggle.md`](context/sprint_briefs/next_sprint_redis_kafka_toggle.md).
- Deliver a minimally viable HOW organ and revive the ANOMALY sensor so the default
  runtime exercises all five dimensions with calibrated thresholds and audit logs.
  - `HowSensor` now wraps the institutional intelligence engine, exposing liquidity,
    participation, and volatility telemetry with documented warn/alert thresholds
    while `AnomalySensor` upgrades the anomaly path with sequence-driven detection
    and market-data fallbacks. Both sensors emit structured audit metadata consumed
    during Tier‑0 ingest and runtime summaries.【F:src/sensory/how/how_sensor.py†L1-L121】【F:src/sensory/anomaly/anomaly_sensor.py†L1-L116】【F:tests/sensory/test_how_anomaly_sensors.py†L1-L73】
  - `BootstrapSensoryPipeline` and the Professional Predator runtime now retain the
    fused dimensional readings as an audit trail so operators and CI can inspect the
    last decisions alongside the HOW/ANOMALY outputs directly from the runtime
    summary.【F:src/orchestration/bootstrap_stack.py†L24-L126】【F:src/runtime/bootstrap_runtime.py†L150-L189】【F:src/runtime/predator_app.py†L220-L303】【F:tests/current/test_bootstrap_runtime_integration.py†L34-L64】
  - `evaluate_sensory_drift` condenses the audit history into
    `telemetry.sensory.drift` snapshots, logging Markdown summaries, publishing the
    feed on the runtime event bus, and surfacing the latest snapshot inside
    `ProfessionalPredatorApp.summary()` so operators can track WHY/HOW drift as
    institutional ingest runs land.【F:src/operations/sensory_drift.py†L1-L215】【F:src/runtime/runtime_builder.py†L960-L1062】【F:src/runtime/predator_app.py†L320-L360】【F:tests/operations/test_sensory_drift.py†L1-L64】【F:tests/runtime/test_professional_app_timescale.py†L720-L780】
  - `WhenSensor` now fuses session overlap intensity, macro-event proximity, and
    option gamma posture through the reusable `GammaExposureAnalyzer` helpers so
    the WHEN organ reflects dealer pin risk. The analyzer surfaces dominant
    strike profiles that bubble the pinned strike into runtime metadata, and
    pytest coverage documents the impact scoring and strike breakdown
    contract.【F:src/sensory/when/when_sensor.py†L89-L143】【F:src/sensory/when/gamma_exposure.py†L213-L244】【F:tests/sensory/test_when_gamma.py†L36-L83】
  - `WhySensor` now blends realized volatility heuristics with yield-curve
    analytics from `YieldSlopeTracker`, emitting richer metadata for operators
    and recording the new regression coverage in `tests/sensory/test_why_yield.py`.
    The snapshot highlights inversion risk, slopes, and macro bias so the WHY
    organ keeps pace with the upgraded data backbone.【F:src/sensory/why/why_sensor.py†L1-L104】【F:src/sensory/dimensions/why/yield_signal.py†L1-L116】【F:tests/sensory/test_why_yield.py†L19-L85】
- Replace the stub genome provider with a small but real genome catalogue sourced
  from historical strategies; wire population management into the evolution engine
  behind a feature flag.
  - The institutional genome catalogue now lives in `src/genome/catalogue.py`,
    exposing metadata, sampling helpers, and calibrated entries spanning trend,
    carry, liquidity, volatility, and macro overlays. `PopulationManager` and the
    evolution engine seed populations from the catalogue when
    `EvolutionConfig.use_catalogue` or the `EVOLUTION_USE_CATALOGUE` flag is set,
    reporting catalogue metadata through population statistics. Pytest coverage
    documents catalogue sampling and feature-flagged seeding for reviewers and
    operators.【F:src/genome/catalogue.py†L1-L221】【F:src/core/population_manager.py†L1-L420】【F:src/core/evolution/engine.py†L1-L120】【F:tests/evolution/test_genome_catalogue.py†L1-L27】【F:tests/current/test_population_manager_catalogue.py†L1-L39】
  - `EvolutionCycleOrchestrator` now records catalogue snapshots, exposes them via
    `catalogue_snapshot`, and publishes `telemetry.evolution.catalogue` events so
    runtime dashboards and CI health checks can confirm seeded populations while
    pytest covers the telemetry payload and event bus fan-out.【F:src/orchestration/evolution_cycle.py†L150-L360】【F:tests/current/test_evolution_orchestrator.py†L1-L170】【F:tests/evolution/test_catalogue_snapshot.py†L1-L43】
  - `EvolutionLineageSnapshot` captures champion lineage (parents, mutation history,
    species drift) and publishes `telemetry.evolution.lineage` alongside the
    orchestrator telemetry so governance can audit how champions evolve away from
    their seeded catalogue entries.【F:src/evolution/lineage_telemetry.py†L1-L165】【F:src/orchestration/evolution_cycle.py†L120-L392】【F:tests/evolution/test_lineage_snapshot.py†L1-L74】【F:tests/current/test_evolution_orchestrator.py†L1-L190】
  - Strategy registry persistence now captures catalogue provenance for every champion
    and the compliance workflow checklist surfaces a "Strategy governance" block that
    consumes the registry summary so approvals reference the seeded desk templates.
    【F:src/governance/strategy_registry.py†L1-L420】【F:src/compliance/workflow.py†L1-L760】【F:tests/governance/test_strategy_registry.py†L1-L60】【F:tests/compliance/test_compliance_workflow.py†L1-L140】
- Publish compliance starter kits – MiFID II/Dodd-Frank control matrices, KYC/AML
  workflow outlines, and audit storage requirements mapped to existing interfaces.
  - `evaluate_compliance_workflows` distils trade compliance and KYC telemetry into
    MiFID II, Dodd-Frank, and audit trail checklists, publishes
    `telemetry.compliance.workflow`, and records Markdown summaries inside the
    professional runtime so operators inherit executable starter kits alongside
    readiness feeds.【F:src/compliance/workflow.py†L1-L392】【F:src/runtime/runtime_builder.py†L1200-L1336】【F:src/runtime/predator_app.py†L1-L520】【F:tests/compliance/test_compliance_workflow.py†L1-L98】【F:tests/runtime/test_professional_app_timescale.py†L200-L320】【F:tests/runtime/test_runtime_builder.py†L160-L240】
- The trade compliance monitor now enforces policy thresholds for execution reports,
  publishes `telemetry.compliance.trade` snapshots on the event bus, and records audit
  entries so runtime summaries expose regulatory guardrails alongside ingest telemetry.
  `TimescaleComplianceJournal` persists every snapshot into the `telemetry.compliance.audit`
  table so institutional deployments inherit a durable compliance trail, and the professional
  runtime surfaces the latest journal entry for operators and dashboards.
  【F:src/compliance/trade_compliance.py†L1-L420】【F:src/data_foundation/persist/timescale.py†L1-L900】【F:src/runtime/predator_app.py†L560-L700】【F:tests/compliance/test_trade_compliance.py†L1-L160】【F:tests/data_foundation/test_timescale_compliance_journal.py†L1-L44】【F:tests/runtime/test_professional_app_timescale.py†L232-L308】
- `evaluate_compliance_readiness` fuses trade-compliance and KYC telemetry into a
  `telemetry.compliance.readiness` snapshot published after each institutional ingest run,
  logging Markdown summaries, updating the runtime status block, and giving operators a
  single view of regulatory posture alongside data backbone and professional readiness feeds.
  【F:src/operations/compliance_readiness.py†L1-L237】【F:src/runtime/runtime_builder.py†L1200-L1288】【F:src/runtime/predator_app.py†L52-L216】【F:tests/operations/test_compliance_readiness.py†L1-L76】【F:tests/runtime/test_runtime_builder.py†L150-L228】
- `KycAmlMonitor` evaluates onboarding dossiers, grades risk posture, and publishes
  `telemetry.compliance.kyc` snapshots with Markdown summaries while optionally journaling
  to Timescale's `telemetry.compliance_kyc` table. The professional runtime wires the monitor
  through feature flags, exposes the latest case snapshot inside `summary()`, and pytest covers
  both the monitor contract and Timescale journal round-trips so institutional KYC/AML posture is
  observable alongside trade compliance feeds.【F:src/compliance/kyc.py†L1-L332】【F:src/data_foundation/persist/timescale.py†L920-L1265】【F:src/runtime/predator_app.py†L80-L390】【F:tests/compliance/test_kyc_monitor.py†L1-L70】【F:tests/data_foundation/test_timescale_compliance_journal.py†L1-L78】【F:tests/runtime/test_professional_app_timescale.py†L200-L308】
- `tools.telemetry.export_risk_compliance_snapshots` packages risk telemetry, execution readiness,
  compliance readiness/workflow blocks, and Timescale journal statistics into a governance-friendly
  JSON bundle, reusing the journal aggregation helpers with dedicated CLI regression coverage so
  reviewers can export evidence without manual SQL.【F:tools/telemetry/export_risk_compliance_snapshots.py†L1-L308】【F:src/data_foundation/persist/timescale.py†L1211-L2163】【F:tests/tools/test_risk_compliance_export.py†L1-L113】【F:tests/data_foundation/test_timescale_compliance_journal.py†L57-L206】【F:tests/data_foundation/test_timescale_execution_journal.py†L86-L123】
- Publish risk posture telemetry so operators can see drawdown, open-position, and
  liquidity guardrails alongside compliance and ingest feeds. `evaluate_risk_posture`
  converts portfolio monitor state and risk-gateway checks into `telemetry.risk.posture`
  snapshots, `TradingManager` logs the markdown summary, and the runtime exposes the
  feed through `ProfessionalPredatorApp.summary()` for dashboards and runbooks.
  【F:src/risk/telemetry.py†L1-L247】【F:src/trading/trading_manager.py†L1-L240】【F:src/runtime/predator_app.py†L200-L320】【F:tests/risk/test_risk_telemetry.py†L1-L102】【F:tests/current/test_bootstrap_runtime_integration.py†L1-L70】
- Harden TradingManager with a reusable risk policy so `RiskConfig` limits govern
  minimum position sizes, aggregate exposure, leverage, and stop-loss enforcement
  before trades reach execution. The policy enriches gateway decisions with policy
  metadata, blocks violations outside research mode, and is covered by dedicated
  unit tests alongside new risk gateway regression scenarios.
  【F:src/trading/risk/risk_policy.py†L1-L214】【F:src/trading/trading_manager.py†L44-L210】【F:tests/trading/test_risk_policy.py†L1-L103】【F:tests/current/test_risk_gateway_validation.py†L1-L170】
  - `build_policy_snapshot` emits structured policy decision telemetry on
    `telemetry.risk.policy`, and the runtime surfaces the latest decision with
    Markdown summaries in control-centre and runtime reports so operators can
    audit individual trade approvals alongside aggregate risk posture.
    【F:src/trading/risk/policy_telemetry.py†L1-L195】【F:src/trading/trading_manager.py†L44-L360】【F:src/operations/bootstrap_control_center.py†L1-L260】【F:src/runtime/predator_app.py†L1-L360】【F:tests/trading/test_risk_policy_telemetry.py†L1-L109】
- Extend operational telemetry by defining monitoring SLOs, alert channels, and
  backup/restore drills for the new data services.
  - `evaluate_ingest_slos` now converts ingest metrics and health reports into an
    operational SLO snapshot, merges default alert routes with extras-defined
    overrides, and publishes the payload on `telemetry.operational.slos` so the
    runtime logs actionable markdown plus event-bus telemetry for ops drills.
    【F:src/operations/slo.py†L1-L206】【F:src/runtime/runtime_builder.py†L240-L333】【F:src/data_foundation/ingest/configuration.py†L1-L244】【F:tests/operations/test_slo.py†L1-L70】【F:tests/data_foundation/test_timescale_config.py†L1-L220】
  - `RuntimeHealthServer` exposes `/health` with FIX connectivity, market-data
    freshness, and telemetry exporter checks, and the runtime builder now starts
    the server automatically so operators and probes inherit a single health
    endpoint with configurable thresholds.【F:src/runtime/healthcheck.py†L1-L258】【F:src/runtime/runtime_builder.py†L1816-L1863】【F:tests/runtime/test_healthcheck.py†L1-L170】
  - `evaluate_event_bus_health` grades queue depth, dropped events, and handler
    failures into a reusable snapshot, publishes `telemetry.event_bus.health`,
    and records the markdown summary in the professional runtime so operators
    can audit event delivery alongside ingest and cache telemetry. The runtime now
    routes the event bus worker and fan-out tasks through the shared
    `TaskSupervisor`, keeping background loops cancellable and exposing them in
    professional summaries for operators.【F:src/operations/event_bus_health.py†L1-L220】【F:src/runtime/runtime_builder.py†L1872-L1916】【F:src/runtime/predator_app.py†L200-L420】【F:src/core/_event_bus_impl.py†L1-L420】【F:tests/operations/test_event_bus_health.py†L1-L82】【F:tests/current/test_event_bus_task_supervision.py†L1-L78】【F:tests/runtime/test_professional_app_timescale.py†L200-L360】
  - OpenTelemetry tracing can now be toggled via `OTEL_*` extras; the runtime
    calls `configure_event_bus_tracer` and threads the resulting tracer through
    `AsyncEventBus`, capturing queue depth and dispatch lag attributes on every
    publish/handler span so distributed traces reflect event bus behaviour. The
    same settings feed a runtime tracer that wraps startup/shutdown callbacks,
    workload execution, and the Timescale ingest orchestrator, recording
    metadata for plan evaluation, fallback drills, and ingest success inside
    distributed traces.【F:src/observability/tracing.py†L1-L244】【F:src/runtime/predator_app.py†L1560-L1610】【F:src/runtime/runtime_builder.py†L250-L408】【F:src/runtime/runtime_builder.py†L1996-L2320】【F:tests/core/test_event_bus_tracing.py†L1-L118】【F:tests/runtime/test_runtime_tracing.py†L1-L134】
  - `evaluate_system_validation` parses the concept-aligned
    `system_validation_report.json`, converts check outcomes into a
    `telemetry.operational.system_validation` snapshot, logs markdown summaries,
    and records the latest report block in professional runtime summaries so
    operators can see validation posture next to readiness feeds.【F:src/operations/system_validation.py†L1-L230】【F:src/runtime/runtime_builder.py†L1905-L1950】【F:src/runtime/predator_app.py†L120-L360】【F:tests/operations/test_system_validation.py†L1-L120】【F:tests/runtime/test_professional_app_timescale.py†L400-L455】【F:tests/runtime/test_runtime_builder.py†L200-L360】
  - `export_operational_snapshots` surfaces professional readiness, security,
    incident response, and system validation blocks through
    `tools/telemetry/export_operational_snapshots.py`, emitting a JSON payload
    that dashboards can ingest directly instead of scraping Markdown summaries.
    pytest exercises the happy path, missing-section warnings, and the
    `--allow-missing` override.【F:tools/telemetry/export_operational_snapshots.py†L1-L143】【F:tests/tools/test_operational_export.py†L1-L86】
  - `evaluate_security_posture` grades MFA coverage, credential and secrets
    rotation, incident drills, intrusion detection, and TLS posture into a
    reusable snapshot, publishes `telemetry.operational.security`, and records
    the markdown summary on the professional runtime so operators inherit an
    institutional security feed alongside existing readiness blocks.
    【F:src/operations/security.py†L1-L318】【F:src/runtime/runtime_builder.py†L1090-L1280】【F:src/runtime/predator_app.py†L120-L340】【F:tests/operations/test_security.py†L1-L120】【F:tests/runtime/test_runtime_builder.py†L120-L240】【F:tests/runtime/test_professional_app_timescale.py†L140-L260】
  - `evaluate_incident_response` consolidates runbook coverage, responder
    rosters, training cadence, and postmortem backlog telemetry into
    `telemetry.operational.incident_response`, publishes Markdown summaries, and
    records the snapshot inside professional runtime summaries so operators can
    audit incident readiness alongside security, cache, and execution feeds.
    【F:src/operations/incident_response.py†L1-L233】【F:src/runtime/runtime_builder.py†L2160-L2215】【F:src/runtime/predator_app.py†L260-L360】【F:tests/operations/test_incident_response.py†L1-L108】【F:tests/runtime/test_runtime_builder.py†L360-L520】【F:tests/runtime/test_professional_app_timescale.py†L520-L600】
  - `evaluate_backup_readiness` inspects Timescale backup policies and recent
    telemetry to publish `telemetry.operational.backups`, log markdown summaries,
    and surface the latest snapshot inside `ProfessionalPredatorApp.summary()` so
    operators can audit backup posture alongside ingest health and SLO feeds.
    【F:src/operations/backup.py†L1-L206】【F:src/runtime/runtime_builder.py†L300-L360】【F:src/runtime/predator_app.py†L260-L380】【F:tests/operations/test_backup.py†L1-L80】【F:tests/runtime/test_professional_app_timescale.py†L160-L220】
  - `evaluate_execution_readiness` fuses order fill rates, rejection rates,
    latency, drop-copy metrics, and connection health into an execution snapshot,
    publishes `telemetry.operational.execution`, and records the block in
    professional runtime summaries so operators can audit execution posture
    alongside ingest, risk, and compliance telemetry.
    【F:src/operations/execution.py†L1-L430】【F:src/runtime/runtime_builder.py†L1700-L1840】【F:src/runtime/predator_app.py†L200-L360】【F:tests/operations/test_execution.py†L1-L110】【F:tests/runtime/test_professional_app_timescale.py†L360-L420】
  - `TimescaleExecutionJournal` persists execution readiness snapshots into the
    `telemetry.execution_snapshots` table and `ProfessionalPredatorApp.summary()`
    now surfaces recent and latest execution records so operators can audit
    institutional execution history alongside live readiness telemetry.
    【F:src/data_foundation/persist/timescale.py†L900-L1290】【F:src/runtime/predator_app.py†L200-L760】【F:tests/data_foundation/test_timescale_execution_journal.py†L1-L91】【F:tests/runtime/test_professional_app_timescale.py†L400-L460】

### 90-day considerations (Later)
- Graduate the data backbone by stress-testing Kafka/Spark batch jobs, documenting
  failover procedures, and proving that ingest tiers can switch without downtime.
  - `execute_spark_stress_drill` now exercises Spark export plans through
    configurable cycles, enforces warn/fail thresholds, publishes
    `telemetry.ingest.spark_stress`, and records markdown-backed summaries inside
    the professional runtime so operators can rehearse batch resilience alongside
    ingest telemetry.【F:src/operations/spark_stress.py†L1-L166】【F:src/runtime/runtime_builder.py†L774-L1256】【F:tests/operations/test_spark_stress.py†L1-L87】【F:tests/runtime/test_professional_app_timescale.py†L740-L813】
- Expand sensory/evolution validation with live-paper trading experiments, anomaly
  drift monitoring, and feedback loops that tune strategy genomes automatically.
  - `evaluate_evolution_experiments` aggregates paper-trading experiment logs and ROI posture into `telemetry.evolution.experiments`, with the trading manager recording experiment events, the runtime builder publishing the snapshot, and professional summaries exposing the markdown block alongside ingest telemetry.【F:src/operations/evolution_experiments.py†L1-L248】【F:src/trading/trading_manager.py†L1-L320】【F:src/runtime/runtime_builder.py†L1959-L2118】【F:src/runtime/predator_app.py†L320-L918】【F:tests/operations/test_evolution_experiments.py†L1-L114】【F:tests/trading/test_trading_manager_execution.py†L1-L140】【F:tests/runtime/test_professional_app_timescale.py†L820-L908】
  - `evaluate_evolution_tuning` fuses experiment and strategy telemetry into actionable recommendations, publishes `telemetry.evolution.tuning`, and records the markdown summary inside the professional runtime so operators can review automated tuning guidance alongside experiment metrics.【F:src/operations/evolution_tuning.py†L1-L443】【F:src/runtime/runtime_builder.py†L2566-L2649】【F:src/runtime/predator_app.py†L229-L515】【F:src/runtime/predator_app.py†L1098-L1104】【F:tests/operations/test_evolution_tuning.py†L1-L172】【F:tests/runtime/test_professional_app_timescale.py†L1298-L1338】
- Deliver the first broker/FIX integration pilot complete with supervised async
  lifecycles, expanded risk gates, compliance checkpoints, and observability hooks.
  - `FixIntegrationPilot` now supervises FIX session lifecycles, binds message queues,
    refreshes broker initiators, and exposes runtime snapshots that track queue metrics,
    order activity, and compliance summaries for downstream evaluators.【F:src/runtime/fix_pilot.py†L1-L210】
  - `evaluate_fix_pilot` grades the pilot state against configurable policy thresholds,
    renders Markdown summaries, and publishes `telemetry.execution.fix_pilot` snapshots
    alongside execution readiness, with regression coverage for state evaluation and
    runtime orchestration flows.【F:src/operations/fix_pilot.py†L1-L240】【F:src/runtime/runtime_builder.py†L2040-L2130】【F:tests/runtime/test_fix_pilot.py†L1-L190】【F:tests/operations/test_fix_pilot_ops.py†L1-L80】
  - `FixDropcopyReconciler` connects the FIX drop-copy session to the pilot, reconciles
    broker order state, and reports backlog or mismatch issues through runtime summaries
    and fix-pilot telemetry with dedicated regression coverage.【F:src/runtime/fix_dropcopy.py†L1-L228】【F:src/runtime/predator_app.py†L1112-L1690】【F:tests/runtime/test_fix_dropcopy.py†L1-L60】
- Publish ROI instrumentation – track fee savings, infrastructure spend, and
  capital efficiency so the €250 → institutional journey is grounded in evidence.
  - ROI telemetry now lives in `evaluate_roi_posture`, wiring the trading manager,
    control centre, and runtime summary to publish `telemetry.operational.roi`
    snapshots with markdown summaries while pytest locks the cost model contract.
    【F:src/operations/roi.py†L1-L164】【F:src/trading/trading_manager.py†L1-L280】【F:src/operations/bootstrap_control_center.py†L1-L320】【F:src/runtime/predator_app.py†L400-L720】【F:tests/operations/test_roi.py†L1-L80】
  - Strategy performance telemetry now aggregates trading-manager experiment events
    and ROI snapshots into `telemetry.strategy.performance`, publishes Markdown summaries,
    and records the latest block in professional runtime summaries so desks can monitor
    execution/rejection mix per strategy alongside ROI posture.【F:src/operations/strategy_performance.py†L1-L537】【F:src/runtime/runtime_builder.py†L2240-L2298】【F:src/runtime/predator_app.py†L91-L986】【F:tests/runtime/test_runtime_builder.py†L281-L523】【F:tests/runtime/test_professional_app_timescale.py†L217-L265】
- Update marketing and onboarding assets once the above pilots demonstrate the
  promised capabilities.

## Document-driven high-impact streams

These streams translate the encyclopedia ambitions into execution tracks. Each
stream should keep the concept excerpts, current-state references, and acceptance
criteria together inside its epic template so engineers inherit the same context.

### Stream A – Institutional data backbone

**Mission** – Replace the bootstrap-only ingest helpers with tier-aware, resilient
data services (TimescaleDB, Redis, Kafka, Spark) that can scale alongside clients.

**Key deliverables**

- TimescaleDB schema design (markets, macro, alternative data) with migration
  automation and retention policies documented.
- Redis caching strategy for hot symbols, limits, and session state, including
  eviction policies and observability hooks.
- Kafka topics that replicate intraday updates into the runtime event bus and feed
  downstream Spark jobs for batch analytics.
- Orchestration logic capable of switching between bootstrap and institutional tiers
  with confidence checks, rollback steps, and operator documentation.
- Validation suites (pytest + notebooks) that prove ingest freshness, latency, and
  recovery times meet agreed SLOs.

**Dependencies & context** – Coordinate with existing state-store refactors,
deployment automation, and the ops telemetry initiative to ensure new services are
monitored from day one.

### Stream B – 4D+1 sensory cortex & evolution engine

**Mission** – Deliver all five sensory organs with calibrated data feeds and revive
the evolution engine so strategies adapt continuously and surface anomaly insights.

**Key deliverables**

- HOW organ implementation focused on order-flow/microstructure or execution-cost
  analytics, with dependency injection for new data feeds.
- Anomaly detection service connected to the runtime bus, leveraging statistical or
  ML detectors with explainability hooks.
- Upgraded WHY/WHAT/WHEN heuristics that ingest macro data, news sentiment, and
  technical signals via the new data backbone.
- Evolution engine enhancements: real genome catalogue, population lifecycle,
  fitness evaluation metrics, and experiment logging for auditability.
- Integration tests and evaluation harnesses that compare sensor outputs against
  historical benchmarks and flag drift.

**Dependencies & context** – Needs the data backbone, risk/compliance input on
acceptable automated adaptations, and collaboration with the research team for
feature sourcing.

### Stream C – Execution, risk, compliance, and ops readiness

**Mission** – Graduate from the in-memory simulator to institutional-grade
execution and governance with supervised async lifecycles and documented controls.

**Key deliverables**

- FIX/EMS adapter pilot with retry/backoff, drop-copy ingestion, and reconciling
  ledgers.
- Expanded risk engine enforcing tiered limits, drawdown guards, leverage checks,
  and configurable rule sets stored in version-controlled policy files.
- Compliance starter kits turned into executable workflows: KYC/AML checklist,
  MiFID II transaction reporting drafts, audit-trail persistence, and operator
  review cadences.
- Operational hardening: monitoring SLOs, alert routing, security controls,
  credential rotation playbooks, backup/restore runbooks.
- Evidence log demonstrating ROI metrics, cost savings, and institutional readiness
  improvements for stakeholder communications.

**Dependencies & context** – Builds on Streams A/B, the observability roadmap, and
ongoing documentation updates to keep operators and compliance informed.

## Long-horizon remediation plan

| Timeline | Outcomes | Key workstreams |
| --- | --- | --- |
| **0–3 months (Align & prototype)** | Concept promises decomposed into epics, first production data slice live, and sensory/evolution scaffolds in place. | - Publish alignment briefs and context packs linking encyclopedia excerpts to code gaps.<br>- Deploy the TimescaleDB prototype with ingest smoke tests and Redis/Kafka design notes.<br>- Implement HOW organ skeleton, revive anomaly hooks, and replace stub genomes with historical seeds.<br>- Clarify compliance tone in public docs and capture validation hypotheses for ROI claims. |
| **3–6 months (Build & integrate)** | Data backbone reliable in CI/staging, sensors and evolution engine operating on new feeds, compliance workflows documented. | - Harden TimescaleDB/Redis/Kafka services with monitoring, backups, and orchestration toggles.<br>- Finish 4D+1 sensor uplift, integrate anomaly analytics, and run paper-trading evolution experiments.<br>- Deliver compliance starter kits, risk policy files, and operational telemetry for new services.<br>- Begin FIX/EMS adapter pilot with supervised async patterns and reconciliation tests. |
| **6–12 months (Institutionalise)** | Execution stack, risk controls, and compliance workflows withstand institutional scrutiny with evidence-backed ROI. | - Expand into Spark batch analytics, tiered deployment automation, and failover drills.<br>- Onboard real broker integrations with audit trails, policy versioning, and continuous validation dashboards.<br>- Track ROI metrics (cost savings, capital efficiency) and iterate marketing claims based on data.<br>- Prepare external audits or partner reviews leveraging the evidence log and documentation corpus. |

## Portfolio snapshot

| Initiative | Phase | Outcome we need | Current status | Next checkpoint |
| --- | --- | --- | --- | --- |
| Institutional data backbone | A | Bootstrap ingest upgraded to TimescaleDB + Redis + Kafka with switchable tiers and recovery drills documented. | Timescale ingest, Redis caching, Kafka streaming, Spark exports, and readiness telemetry ship through the professional runtime, with bootstrap/institutional toggles proven in CI. | Cross-region failover rehearsal and automated scheduler cutover using the failover drill plus readiness feeds (Q3). |
| Sensory cortex & evolution uplift | B | All five sensory organs online with calibrated feeds and the evolution engine managing real genomes under feature flags. | HOW/WHY/WHEN/ANOMALY organs now publish calibrated telemetry; drift monitoring, catalogue seeding, lineage snapshots, and evolution experiments run behind feature flags. | Extend live-paper experiments and automated tuning loops powered by the evolution experiment telemetry (Q3). |
| Execution, risk, compliance, ops readiness | C | Broker/FIX integration pilot operating with expanded risk controls, compliance workflows, and observability. | FIX pilot supervision, execution readiness, risk policy telemetry, and compliance workflows/monitors ship with persistent Timescale journals and runtime summaries. | Expand broker connectivity with drop-copy ingestion and reconciliation for the FIX pilot (Q3). |
| Supporting modernization (formatter, regression, telemetry) | Legacy | Foundational hygiene remains green while high-impact streams ramp up. | Formatter rollout, coverage/formatter metrics automation, and flake telemetry keep hygiene signals green while modernization streams land. | Collapse the remaining formatter allowlist and feed CI telemetry into deployment dashboards (Q3). |

> **Automation:** Run `python -m tools.roadmap.snapshot` to regenerate this table from the current repository state. The CLI also exposes a JSON feed for dashboards via `python -m tools.roadmap.snapshot --format json`.

## Active modernization streams

Legacy initiatives below remain in flight to keep the repo healthy while the
document-driven streams ramp. Treat them as supporting tracks—ensure they stay
green but do not let them crowd out the higher leverage work above.

### Initiative 1 – Formatter normalization (Phase 6)

**Mission** – Land `ruff format` in reviewable slices until the repository passes
`ruff format --check .` without leaning on `config/formatter/ruff_format_allowlist.txt`.

**Definition of done**

- Stage 0 (`tests/current/`) formatted, pytest green, and allowlist expanded
  accordingly.
- Stage 1 (`src/system/`, `src/core/configuration.py`) formatted with documented
  manual edits and follow-up tickets for non-mechanical cleanups.
- Remaining directories sequenced in [`docs/development/formatter_rollout.md`](development/formatter_rollout.md)
  with owners and merge order captured.
- CI enforces the formatter globally and the allowlist shrinks to empty (or is
  removed entirely) without exceptions.
- Contributor guidance in [`docs/development/setup.md`](development/setup.md) and
  the PR checklist reflects the post-rollout workflow.

**Key context**

- [`docs/development/formatter_rollout.md`](development/formatter_rollout.md) – historical rollout log
- [`docs/development/setup.md`](development/setup.md) – contributor workflow
- [`docs/status/ci_health.md`](status/ci_health.md) – formatter telemetry trendlines

**Recent progress**

- Repo-wide Ruff enforcement landed; the allowlist/script workflow was retired and CI now runs `ruff format --check .`.
- Stage 4 directories (`src/data_integration/`, `src/operational/`, `src/performance/`) were formatted with telemetry/documentation refreshed to reflect the new guard.
- Formatter sequencing notes remain in the rollout guide for historical reference alongside updated contributor docs.

**Now**

- [x] Normalize `src/data_foundation/replay/` with targeted replays to verify no data
      integrity drift before widening the allowlist. (Verified via `ruff format` and
      replay smoke checks.)
- [x] Diff `src/data_foundation/schemas.py` against formatter output, capture any
      manual reconciliations, and stage pytest smoke checks for downstream users.
- [x] Publish a Stage 4 briefing that lines up `src/operational/` and
      `src/performance/` slices with nominated reviewers and freeze windows.

**Next**

- [x] Land the data integration, operational, and performance formatting PRs with
      focused pytest runs covering `src/operational/metrics.py`,
      `src/performance/vectorized_indicators.py`, and the ingestion slices.
- [x] Collapse the remaining allowlist entries and wire `ruff format --check .` into
      the main CI workflow once the Stage 4 backlog clears.
- [x] Update contributor docs (`setup.md`, PR checklist) to describe the new default
      formatter workflow and local tooling expectations.

**Delivery checkpoints**

- Data foundation replay + schemas formatting ready for review (Week 1).
- Operational/performance slices rehearsed and queued (Week 3).
- Allowlist removal RFC circulated with rollout guardrails (Week 5).

**Later**

- [x] Retire the allowlist guard and remove redundant formatters (for example,
      Black) once Ruff owns enforcement end-to-end.

**Risks & watchpoints**

- High-churn directories causing repeated merge conflicts – mitigate by staging PRs
  early in the week and communicating freeze windows.
- Generated files or vendored assets accidentally formatted – confirm each slice
  honors project-level excludes before landing.
- Formatting changes obscuring behavior tweaks – insist on mechanical-only commits
  paired with focused follow-ups for real fixes.

**Telemetry**

- Track formatter and coverage trendlines in `tests/.telemetry/ci_metrics.json` and surface updates in the CI health snapshot.
- Flag the rollout timeline in retrospectives so future contributors understand the
  historical sequencing.

### Initiative 2 – Regression depth in trading & risk (Phase 7)

**Mission** – Convert the CI baseline hotspots into deterministic regression suites
that guard trading execution, risk controls, and orchestration wiring.

**Definition of done**

- Coverage for `src/operational/metrics.py`, `src/trading/models/position.py`,
  `src/data_foundation/config/`, and
  `src/sensory/dimensions/why/yield_signal.py` improves measurably and holds steady
  in CI.
- FIX execution flows (order routing, reconciliation, error handling) have
  explicit success and failure scenarios documented and enforced by pytest.
- Risk guardrails (position limits, drawdown gates, Kelly sizing) are exercised
  across happy paths and failure modes with clear assertions.
- Orchestration composition tests verify adapters, event bus wiring, and optional
  module degradation across supported configurations.
- Coverage deltas captured in [`docs/status/ci_health.md`](status/ci_health.md)
  after each regression batch.

**Key context**

- [`docs/ci_baseline_report.md`](ci_baseline_report.md)
- Existing regression suites in `tests/current/`
- `src/trading/models/position.py`
- `src/operational/metrics.py`
- `src/data_foundation/config/`
- `src/sensory/dimensions/why/yield_signal.py`

**Recent progress**

- Baseline hotspots decomposed into regression tickets and logged in
  [`docs/status/regression_backlog.md`](status/regression_backlog.md).
- Deterministic FIX failure-path coverage added in
  `tests/current/test_fix_manager_failures.py`.
- Regression tests landed for position lifecycle accounting, risk guardrails, data
  foundation config loaders, and orchestration runtime smoke checks.

**Now**

- [x] Extend `tests/current/test_execution_engine.py` (or add a new suite) to cover
      partial fills, retries, and reconciliation paths in
      `src/trading/execution/execution_engine.py`. (`tests/current/test_execution_engine.py`)
- [x] Add regression coverage for drawdown recovery and Kelly-sizing adjustments in
      `src/risk/risk_manager_impl.py` with fixtures that mirror production configs.
      (`tests/current/test_risk_manager_impl.py`)
- [x] Introduce property-based tests around order mutation flows in
      `src/trading/models/order.py` to lock in serialization and validation logic.
      (`tests/current/test_order_model_properties.py`)

**Next**

- [x] Chain orchestration, execution, and risk modules in an end-to-end scenario test
      that verifies event bus wiring and fallback behavior. `tests/current/test_orchestration_execution_risk_integration.py`
      now drives the bootstrap trading stack against the async event bus, records
      risk/policy/ROI telemetry fan-out, and proves trades still execute when the bus is
      offline by asserting local snapshots keep advancing while events are dropped.【F:tests/current/test_orchestration_execution_risk_integration.py†L117-L284】
- [ ] Record coverage deltas in the CI health snapshot after each regression landing
      and alert on regressions outside agreed thresholds.
- [x] Expand sensory regression focus to `src/sensory/dimensions/why/yield_signal.py`
      with fixtures derived from historical market data. `tests/sensory/test_why_yield.py`
      now covers inversion detection and the upgraded WHY sensor blend of macro
      and yield-curve telemetry, keeping the roadmap’s sensory regression slice
      anchored to executable tests.【F:tests/sensory/test_why_yield.py†L19-L85】

**Later**

- [ ] Expand into scenario-based integration tests once formatter noise is gone and
      churn stabilizes.
- [ ] Capture coverage trendlines in the CI dashboard and celebrate modules that
      cross agreed thresholds.

**Risks & watchpoints**

- Mocked subsystems may give false confidence – document limitations and plan for
  real integrations in future roadmaps.
- Flaky tests can erode trust – add telemetry hooks to surface retries and
  investigate immediately.

**Telemetry**

- Update coverage by module in the CI health snapshot after each landing.
- Track flaky-test counts or retry rates to feed back into the observability plan.

### Initiative 3 – Operational telemetry & alerting (Phase 8)

**Mission** – Provide actionable observability for CI so failures surface without
manual log digging or the deprecated Kilocode bridge.

**Definition of done**

- A single alerting channel (GitHub issue automation, Slack webhook, or email
  digest) selected, documented, and validated with a forced failure.
- Lightweight dashboard or README section summarizing recent CI runs, formatter
  status, and coverage trends published and maintained.
- On-call expectations, escalation paths, and response targets codified alongside
  the observability plan.
- CI failure telemetry (e.g., pytest flake metadata) stored in an accessible
  location for trend analysis.

**Key context**

- `.github/workflows/ci.yml`
- `.github/workflows/ci-failure-alerts.yml`
- [`docs/operations/observability_plan.md`](operations/observability_plan.md)
- [`docs/status/ci_health.md`](status/ci_health.md)

**Recent progress**

- GitHub issue automation is live and validated through the `alert_drill` dispatch
  input in CI.
- Flake telemetry captured in `tests/.telemetry/flake_runs.json` with documentation
  in the observability plan.
- CI health dashboard refreshed with formatter expansion notes and alert-drill
  references.

**Now**

- [x] Document the Slack/webhook integration plan (owners, secrets, rollout steps)
      directly in the observability plan and CI health snapshot.
- [x] Automate ingestion of `tests/.telemetry/flake_runs.json` into a lightweight
      dashboard or summary table that highlights retry frequency and failure types.
- [x] Publish the alert-drill calendar (quarterly cadence) and add a checklist for
      pre/post drill verification steps.

**Next**

- [x] Deliver the Slack/webhook bridge once credentials are provisioned and run a
      forced-failure drill to validate the end-to-end flow. The
      `notify-slack` job in `.github/workflows/ci-failure-alerts.yml` mirrors the
      managed alert issue into `#ci-alerts` when the `SLACK_CI_WEBHOOK` secret is
      present, and the September drill confirmed delivery.
- [x] Expand telemetry capture to include coverage trendlines and formatter adoption
      metrics with references in CI artifacts. `tools/telemetry/update_ci_metrics.py`
      ingests coverage XML and the Ruff allowlist to append trend entries to
      `tests/.telemetry/ci_metrics.json`, and pytest exercises the CLI plus JSON
      contract so CI dashboards can surface the new telemetry.【F:tools/telemetry/update_ci_metrics.py†L1-L80】【F:tools/telemetry/ci_metrics.py†L1-L148】【F:tests/tools/test_ci_metrics.py†L1-L133】【F:tests/.telemetry/ci_metrics.json†L1-L4】
- [ ] Evaluate whether runtime (non-CI) observability hooks belong in this initiative
      or a follow-on roadmap.

**Later**

- [ ] Evaluate whether runtime observability (beyond CI) should join the roadmap
      once formatter and regression work settles.
- [ ] Schedule quarterly reviews of alert effectiveness with on-call participants.

**Risks & watchpoints**

- Alert fatigue if the signal is noisy – ensure alerts auto-resolve and include
  actionable context.
- Ownership drift – keep rotation details in the observability plan and revisit at
  each roadmap review.

**Telemetry**

- Capture alert tests, run history, and open incidents in the CI health document.
- Track mean time to acknowledgment (MTTA) and resolution (MTTR) once alerts are
  live.

### Initiative 4 – Dead code remediation & modular cleanup (Phase 9)

**Mission** – Keep the dead-code signal actionable while decomposing high-fanin
modules into maintainable components.

**Definition of done**

- Latest dead-code audit findings triaged as delete, refactor, or intentional
  dynamic hook with rationale captured.
- Confirmed dead paths removed with documentation or tests updated to reflect any
  behavioral changes.
- Remaining monolithic modules in `src/core` and adjacent packages decomposed into
  focused components without breaking public interfaces.
- Recurring audits scheduled after each cleanup batch, with results appended to
  [`docs/reports/dead_code_audit.md`](reports/dead_code_audit.md).

**Key context**

- [`docs/reports/dead_code_audit.md`](reports/dead_code_audit.md)
- `scripts/audit_dead_code.py`
- `.github/workflows/dead-code-audit.yml`
- Architecture references in `docs/architecture/`

**Recent progress**

- September audit triage logged with decisions on retained protocol parameters.
- Unused `_nn` type import and stale parity-checker helper removed alongside the
  latest smoke tests.
- Audit backlog prioritized for upcoming cleanup passes.

**Now**

- [x] Decompose `src/core/state_store.py` by extracting persistence adapters and
      documenting the new interfaces.
- [x] Triage `src/core/performance/` and `src/core/risk/` helpers for deletion or
      consolidation, mapping dependencies before submitting PRs.
- [x] Align relevant docs and examples after each deletion/refactor so onboarding
      material stays current.

**Next**

- [ ] Schedule automated dead-code audits post-merge and file tickets for anything
      that persists across two consecutive scans.
- [ ] Pair with the architecture guild to identify additional high-fanin modules and
      charter decomposition spikes.
- [ ] Integrate module fan-in metrics into the CI health snapshot to track progress.

**Later**

- [ ] Continue scheduled audits after each cleanup batch, noting trends and
      tracking resolved findings in the report.
- [ ] Evaluate whether additional tooling or static analysis would reduce audit
      noise.

**Risks & watchpoints**

- Accidentally deleting dynamic entry points – document intentional keepers and add
  regression tests where reasonable.
- Audit fatigue – keep the report curated so future scans remain trustworthy.

**Telemetry**

- Log each audit pass, decisions made, and resulting PRs in the audit report.
- Track module fan-in or fan-out metrics where available to measure decomposition
  impact.

### Initiative 5 – Runtime orchestration & risk hardening (Phase 10)

**Mission** – Harden the live-trading runtime by carving clear service boundaries,
supervising async workloads, and enforcing the risk policies that already exist in
configuration.

**Definition of done**

- `main.py` backed by a dependency-injected application builder (`src/runtime/runtime_builder.py`)
  with discrete entrypoints for ingestion and live trading plus documented shutdown
  hooks.
- Background services supervised via structured task groups with deterministic
  cancellation and shutdown tests.
- `RiskManager` enforces `RiskConfig` inputs (position sizing, leverage, drawdown,
  exposure) with regression coverage and documentation updates.
- Operator playbooks, architecture notes, and the top-level `README.md` reflect the
  new runtime builder, task supervision story, and rollback procedures.
- Safety and risk configuration changes flow through a documented review workflow
  with compliance/operations sign-off and audit breadcrumbs.
- All system validation checks in `system_validation_report.json` passing in CI,
  with dashboards highlighting drift.
- Public API exports (for example `src/core/__init__.py`) cleaned up so advertised
  symbols exist, are documented, or are intentionally deprecated.

**Key context**

- `main.py`, `src/runtime/runtime_builder.py`
- `src/risk/risk_manager_impl.py`
- `src/core/__init__.py`
- `src/brokers/fix/` adapters and sensory orchestrators
- `system_validation_report.json`
- `README.md`, `docs/ARCHITECTURE_REALITY.md`, and future runtime/runbook drafts

**Recent progress**

- Technical debt assessment documented the monolithic runtime, unsupervised async
  tasks, and ineffective risk enforcement.
- Preliminary regression suites now touch execution-engine partial fills and risk
  drawdown recovery, providing scaffolding for broader runtime coverage.
- Formatter and modular cleanup work reduced noise around `src/core/` so runtime
  refactors can proceed with less churn.

**Now**

- [x] Draft a runtime builder and shutdown sequence design that separates ingestion
      and trading workloads, including testing strategy and rollout plan.【F:src/runtime/runtime_builder.py†L76-L898】【F:tests/runtime/test_runtime_builder.py†L1-L118】
- [x] Inventory `asyncio.create_task` usage across brokers, sensory organs, and
      orchestrators, introducing a reusable `TaskSupervisor` so runtime
      background work funnels through managed cancellation and logging instead
      of ad-hoc tasks.【F:src/runtime/task_supervisor.py†L1-L152】【F:src/runtime/predator_app.py†L95-L261】【F:tests/runtime/test_task_supervisor.py†L1-L64】
- [x] Map `RiskConfig` parameters to required enforcement logic, outlining tests,
      documentation updates, and telemetry hooks. `RiskManagerImpl` now accepts
      canonical `RiskConfig` inputs, enforces mandatory stop losses, min/max
      sizing, and aggregates exposure/leverage through `RealRiskManager` while
      regression suites cover the new guardrails and sizing bounds.
      【F:src/risk/risk_manager_impl.py†L1-L360】【F:src/risk/real_risk_manager.py†L1-L170】【F:tests/current/test_risk_manager_impl.py†L1-L200】【F:tests/current/test_real_risk_manager.py†L1-L120】
- [ ] Define the compliance/operations review workflow for safety configuration
      changes and catalogue the observability gaps in FIX/orchestrator adapters.

**Next**

- [x] Implement the application builder with dedicated CLIs, integrate structured
      shutdown hooks, and add smoke tests for restart flows. The new `emp-runtime`
      CLI exposes `summary`, `run`, `ingest-once`, and `restart` subcommands that
      wrap `RuntimeApplication`, handle signal-aware shutdowns/timeouts, and let
      operators rehearse restart cycles without bespoke scripts; pytest covers JSON
      summaries, trading suppression, and restart sequencing to keep the contract
      locked in CI.【F:src/runtime/cli.py†L1-L258】【F:tests/runtime/test_runtime_cli.py†L1-L88】
- [x] Replace ad-hoc event loops in FIX adapters with supervised task factories so
      FIX sensory, broker, and drop-copy workers register with the runtime
      `TaskSupervisor` and exit cleanly; pytest now asserts supervisor tracking in
      the pilot, drop-copy, and professional runtime suites.
- [ ] Rebuild `RiskManager` to honor leverage, exposure, and drawdown limits with
      deterministic pytest coverage and updated operator guides.
- [x] Deliver structured logging, metrics, and health checks for the runtime
      builder, FIX bridges, and orchestrators; record steady-state expectations in
      the runbook. Structured logging now flows through `configure_structured_logging`
      and the runtime builder extras (`RUNTIME_LOG_STRUCTURED`, `RUNTIME_LOG_LEVEL`,
      `RUNTIME_LOG_CONTEXT`) so operators can enable JSON logs without code
      changes.【F:src/observability/logging.py†L1-L122】【F:src/runtime/runtime_builder.py†L1838-L1877】【F:tests/observability/test_logging.py†L1-L74】【F:tests/runtime/test_runtime_builder.py†L600-L676】

**Later**

- [ ] Evaluate carving FIX and orchestration adapters into separately deployable
      services with health checks and metrics once the builder lands.
- [ ] Extend system validation into continuous monitoring (dashboards, alerts) so
      drift is caught immediately.
- [ ] Introduce configuration-as-code or policy-versioning workflows once risk
      enforcement stabilizes.
- [x] Capture audit trails for runtime configuration changes and integrate them
      with SBOM/policy reporting as part of the compliance toolkit. The new
      configuration audit module grades run-mode, tier, and credential shifts,
      publishes `telemetry.runtime.configuration`, and stores markdown snapshots
      on the professional runtime while Timescale persistence keeps a durable
      journal for audits.【F:src/operations/configuration_audit.py†L1-L308】【F:src/runtime/runtime_builder.py†L1-L260】【F:src/runtime/predator_app.py†L1-L400】【F:src/data_foundation/persist/timescale.py†L1-L2150】【F:tests/operations/test_configuration_audit.py†L1-L94】【F:tests/data_foundation/test_timescale_configuration_journal.py†L1-L47】【F:tests/runtime/test_professional_app_timescale.py†L1-L200】

**Risks & watchpoints**

- Runtime refactors can destabilize trading loops – insist on feature flags and
  rollback plans for each landing.
- Async supervision changes may expose latent race conditions; schedule paired
  regression runs and soak tests.
- Risk guardrail reimplementation could block orders if misconfigured – document
  defaults and provide sandbox rehearsals before rollout.

**Telemetry**

- Track runtime validation status and shutdown test coverage alongside existing CI
  health metrics.
- Record risk enforcement outcomes (violations caught, config versions) to prove
  guardrails are active.
- Capture namespace cleanup deltas (removed/added exports) in changelogs for
  downstream consumers.

## Completed phases (0–5)

| Phase | Focus | Status | Highlights |
| --- | --- | --- | --- |
| 0 | Immediate hygiene | ✅ Complete | Retired the Kilocode bridge, resolved `.github/workflows/ci.yml` merge markers, and captured the CI baseline. |
| 1 | Policy consolidation | ✅ Complete | Centralized forbidden-integration checks and aligned `config.yaml` with the FIX-only posture. |
| 2 | Dependency & environment hygiene | ✅ Complete | Promoted `requirements/base.txt`, pinned the development toolchain, and added the runtime requirements check CLI. |
| 3 | Repository cleanup | ✅ Complete | Pruned stray artifacts, added guardrails to keep them out, and seeded the dead-code audit backlog. |
| 4 | Test coverage & observability | ✅ Complete | Expanded regression nets across configuration, FIX parity, risk management, and orchestration while improving CI log surfacing. |
| 5 | Strategic refactors | ✅ Complete | Decomposed high-coupling modules (e.g., `src/core/interfaces`, trading performance tracker) and refreshed architecture docs to match the layering contract. |

Consult the linked documentation in each phase for implementation details when
debugging regressions or planning follow-up work.

## Review cadence & reporting

- Keep a shared checklist linked to this roadmap so contributors can claim items
  and capture findings.
- Review progress in weekly debt triage meetings and update the Now / Next / Later
  board accordingly.
- Revisit the roadmap quarterly (or after completing any initiative above) to
  adjust priorities based on emerging risks or product requirements.
- Note material updates in commit messages so change history explains why items
  moved or definitions shifted.

## Coding guidance for the recovery programme

Carry these practices into every ticket so CI stays green while we work through the backlog:

- **Treat `mypy` as a design partner.** Run targeted checks locally before committing and lean on the [strict package list](mypy.ini) to decide which modules must pass cleanly today. Capture each sweep inside [`docs/mypy_status_log.md`](mypy_status_log.md) so the roadmap reflects current error counts.【F:mypy.ini†L1-L28】【F:docs/mypy_status_log.md†L1-L89】
- **Prefer typed coercion helpers over ad-hoc casts.** Reuse the shared numeric utilities and mapping coercers so heterogeneous telemetry payloads remain safe without `# type: ignore` clutter.【F:src/core/coercion.py†L1-L180】【F:src/risk/telemetry.py†L1-L247】
- **Stabilise optional dependencies with shims.** Keep Redis, Kafka, and OpenTelemetry imports behind typed interfaces or stub packages, and mirror any runtime fallback in the corresponding `.pyi` files to stop drift between implementation and static analysis.【F:src/data_foundation/cache/redis_cache.py†L1-L240】【F:src/data_foundation/streaming/kafka_stream.py†L1-L1919】【F:stubs/redis/__init__.pyi†L1-L23】【F:docs/mypy_playbooks.md†L1-L131】
- **Document the intent next to the code.** When you normalise telemetry payloads or tighten protocols, update the recovery plan, backlog inventory, or relevant runbook in the same change so future sweeps inherit the reasoning.【F:docs/ci_recovery_plan.md†L1-L132】【F:docs/mypy_backlog_inventory.md†L1-L85】【F:docs/operations/runbooks/kafka_ingest_offset_recovery.md†L1-L66】
- **Retire ignores aggressively.** Replace broad `# type: ignore` markers with precise fixes or annotated casts, and log any unavoidable exceptions with the error code plus a follow-up entry in the backlog. Use [`docs/mypy_conventions.md`](mypy_conventions.md) and [`docs/mypy_playbooks.md`](mypy_playbooks.md) as the canonical references for acceptable suppressions.【F:docs/mypy_conventions.md†L1-L70】【F:docs/mypy_playbooks.md†L1-L131】
- **Engineer for async determinism.** Ensure task factories accept coroutines, awaitable flows return concrete payload types, and long-lived tasks are supervised so type hints mirror actual runtime guarantees.【F:src/runtime/task_supervisor.py†L1-L220】【F:src/runtime/runtime_builder.py†L1-L2320】【F:src/trading/integration/fix_broker_interface.py†L1-L210】

> **Implementation tip:** When a sweep touches multiple packages, stage commits by subsystem (e.g. cache, telemetry, runtime) and include the relevant roadmap checklist link in each message. This keeps reviewers anchored to the plan while the backlog burns down.
