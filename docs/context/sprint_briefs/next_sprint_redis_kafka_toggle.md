# Sprint brief – Redis/Kafka ingest hardening (Next)

**Sprint window:** 2 weeks focused on converting the data backbone "Next"
roadmap outcomes into executable tickets once the Timescale prototype is in
place.

## Concept anchors

- The concept blueprint specifies TimescaleDB for time-series storage with Redis
  caching and Kafka streaming in the professional tiers.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L378-L386】【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L569-L579】
- The roadmap's 60-day "Next" outcomes call for a production-grade ingest
  vertical that combines TimescaleDB, Redis caching, Kafka mirroring, and richer
  operational telemetry.【F:docs/roadmap.md†L69-L81】
- The alignment brief still flags Redis and Kafka as open gaps blocking tier
  switching and ingest observability even after the Timescale prototype.
  【F:docs/context/alignment_briefs/institutional_data_backbone.md†L61-L114】

## Reality signals

- The runtime now wires Timescale ingest and provisions Redis clients when
  institutional credentials are supplied, but cache policies, telemetry, and
  Kafka mirroring remain open gaps for the professional tier toggle.【F:src/runtime/runtime_builder.py†L700-L898】【F:src/runtime/predator_app.py†L1-L420】
- Portfolio monitoring persists to an in-memory Redis stub by default, proving a
  managed Redis path is still missing.【F:src/trading/monitoring/portfolio_monitor.py†L64-L135】
- CI telemetry does not report ingest freshness, cache hit rates, or Kafka lag,
  reinforcing the roadmap call for upgraded observability.【F:docs/status/ci_health.md†L8-L52】

## Sprint intent

Deliver a vertical slice that:

1. replaces Redis stubs with injectable production clients and cache policies,
2. stands up Kafka streaming for intraday updates feeding the runtime bus, and
3. introduces a configuration toggle with validation hooks so operators can
   switch between bootstrap (DuckDB) and professional (Timescale/Redis/Kafka)
   tiers.

## Workstreams and ticket scaffolding

### 1. Redis production integration (estimated 3 tickets)

- **Ticket A – Connection + config plumbing (1 day)**
  - Add `RedisConnectionSettings` mirroring the Timescale helper with URL,
    credentials, TLS, and pool sizing sourced from `config.yaml` overlays.
  - Update dependency injection so Tier‑1 apps instantiate a real Redis client
    when credentials exist, falling back to `InMemoryRedis` only for Tier‑0.
  - Validation: pytest stub that patches a fake Redis server (`fakeredis` or
    local container) and asserts state survives process restarts.
  - **Status:** Delivered – Redis settings live under `src/data_foundation/cache/`, runtime dependency injection now selects managed Redis in institutional mode, and pytest coverage exercises both the fakeredis-backed path and the bootstrap fallback.【F:src/data_foundation/cache/redis_cache.py†L1-L212】【F:tests/runtime/test_predator_app_redis.py†L1-L73】
- **Ticket B – Cache policy + eviction (1 day)**
  - Define caching strategy for hot FX symbols (TTL, max keys, invalidation
    hooks) and document defaults in the alignment brief appendices.
  - Extend portfolio monitor to emit cache hit/miss counters via the event bus.
  - Validation: unit test verifying TTL expiration and eviction metrics.
  - **Status:** Delivered – portfolio monitor cache telemetry now normalises hit/miss
    counters, derives hit-rate, stamps namespace/backing metadata, and publishes the
    configuration flag so downstream consumers inherit deterministic cache evidence,
    with regression coverage locking the payload contract.【F:src/trading/monitoring/portfolio_monitor.py†L155-L220】【F:tests/trading/test_portfolio_monitor_cache_metrics.py†L1-L97】
- **Ticket C – Failure drills & runbooks (0.5 day)**
  - Capture Redis outage playbook (failover to Timescale reads, warmup after
    reconnect) in `docs/operations/`.
  - Add health check integration (ping on startup, background heartbeat).
  - Validation: tox/notebook or scripted drill verifying fallback path.
  - **Status:** Completed – the [Redis cache outage runbook](../../operations/runbooks/redis_cache_outage.md)
    now documents detection, stabilisation, recovery, and validation tied to the
    runtime telemetry.【F:docs/operations/runbooks/redis_cache_outage.md†L1-L60】

### 2. Kafka ingest mirroring (estimated 3 tickets)

- **Ticket D – Kafka client wrapper + config (1 day)**
  - Introduce `KafkaConnectionSettings` with SASL/TLS support, aligning with the
    concept stack.
  - Build a small producer helper that serialises Yahoo/ingest deltas into a
    `market-data.daily-bars` topic with schema version metadata.
  - Validation: integration test using `pytest` + `aiokafka`/`kafka-python`
    harness to assert message shape.
  - **Status:** Topic parsing now feeds `KafkaTopicProvisioner`, which auto-creates ingest topics when `KAFKA_INGEST_AUTO_CREATE_TOPICS` (or `KAFKA_AUTO_CREATE_TOPICS`) is set; `build_institutional_ingest_config` exposes the resolved topic list in metadata, and pytest covers the provisioning flow with a fake admin client.【F:src/data_foundation/streaming/kafka_stream.py†L502-L620】【F:src/data_foundation/ingest/configuration.py†L1-L210】【F:tests/data_foundation/test_kafka_stream.py†L60-L360】【F:tests/data_foundation/test_timescale_config.py†L1-L160】
- **Ticket E – Runtime event bus bridge (1 day)**
  - Subscribe the runtime event bus to the Kafka topic and replay into existing
    sensor/risk queues without breaking Tier‑0 behaviour.
  - Provide feature flag to disable streaming for local runs.
  - Validation: async test proving events forwarded to `EventBus.publish`.
- **Ticket F – Replay & backfill tooling (1 day)**
  - Implement CLI/notebook to backfill Kafka topic from Timescale snapshots,
    documenting retention, idempotency, and offset management.
  - Validation: smoke test ensuring replay honours sequence ordering and skips
    duplicates.
  - Status: `backfill_ingest_dimension_to_kafka` now replays Timescale snapshots into configured ingest topics with `backfill` metadata and pytest coverage documenting the replay payload; the new [Kafka ingest offset recovery runbook](../../operations/runbooks/kafka_ingest_offset_recovery.md) closes the operational follow-up for offset management.【F:src/data_foundation/streaming/kafka_stream.py†L1191-L1374】【F:tests/data_foundation/test_kafka_stream.py†L340-L420】【F:docs/operations/runbooks/kafka_ingest_offset_recovery.md†L1-L66】

### 3. Tier toggle & observability (estimated 2 tickets)

- **Ticket G – Configurable tier switch (1 day)**
  - Extend `SystemConfig` to surface a `data_backbone_mode` enum with options
    `bootstrap` and `institutional`.
  - Wire CLI/ENV toggles so operators can flip tiers without code changes,
    defaulting to bootstrap unless Redis/Kafka credentials exist.
  - Validation: integration test verifying Tier‑0 uses DuckDB/InMemoryRedis
    while Tier‑1 uses Timescale/Redis/Kafka paths.
  - Status: `SystemConfig` now exposes the enum and the runtime entrypoint
    respects the toggle when routing ingest, leaving Redis/Kafka wiring as the
    remaining work for this ticket.【F:src/governance/system_config.py†L1-L231】【F:src/runtime/runtime_builder.py†L700-L898】
- **Ticket H – Telemetry + CI surfacing (1 day)**
  - Add ingest freshness, Redis hit ratio, and Kafka consumer lag metrics to the
    CI health snapshot and control centre dashboards.
  - Emit Prometheus/Grafana-compatible metrics and ensure pytest smoke tests
    capture them in assertions.
  - Validation: docs update + pytest ensuring metrics produced during ingest
    pipeline run.
  - **Status:** Timescale ingest health reports now publish on `telemetry.ingest.health` and stream to Kafka via `KafkaIngestHealthPublisher`, with pytest coverage documenting the payload shape for downstream consumers.【F:src/runtime/runtime_builder.py†L445-L650】【F:src/data_foundation/streaming/kafka_stream.py†L1064-L1520】【F:tests/data_foundation/test_kafka_stream.py†L300-L420】

## Definition of Done checkpoints

- Redis and Kafka clients configurable via `config.yaml`/environment overrides
  with secure defaults and documented rotation steps.
- Tier toggle verified in CI: bootstrap path continues to run without external
  services; institutional path exercises Redis/Kafka helpers using local
  containers or mocks.
- Telemetry surfaces in `docs/status/ci_health.md` and any dashboards so the
  ingest vertical exposes freshness, cache, and streaming health.
- Runbooks capture Redis/Kafka outage response and replay drills.

## Instrumentation & validation plan

- Extend existing Timescale pytest suite with Redis/Kafka fixtures and tier
  toggle integration tests.
- Capture Kafka + Redis docker-compose snippets (or references to managed
  services) for developers running the suite locally.
- Publish a sprint-close summary quoting metrics before/after to prove the gap
  closed.

## Dependencies & coordination

- Align with runtime refactor work to ensure dependency injection handles the
  new clients cleanly.【F:docs/technical_debt_assessment.md†L36-L76】
- Coordinate with compliance/risk streams to capture audit requirements for data
  retention once Redis/Kafka hold state beyond Timescale.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L893-L925】
- Schedule Redis/Kafka infrastructure provisioning with the ops telemetry
  initiative so monitoring lands alongside the services.【F:docs/roadmap.md†L95-L117】
