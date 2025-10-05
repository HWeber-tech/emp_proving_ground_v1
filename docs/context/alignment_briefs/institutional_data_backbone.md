# Alignment brief – Institutional data backbone

## Concept promise

- Professional tiers require TimescaleDB for time-series storage, Redis for hot
  caches, Kafka for streaming, and Spark for batch analytics as part of the
  layered data flow that feeds the sensory cortex and execution stack.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L376-L436】
- The architecture overview confirms the layered runtime that should host these
  services once implemented.【F:docs/architecture/overview.md†L9-L37】

## Reality snapshot

- The development status report classifies ingest, evolution, execution, and
  strategy modules as mock frameworks with no production-grade market data or
  risk sizing pipelines.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Technical debt findings flag unsupervised async tasks, hollow risk checks, and
  namespace drift that block dependable runtime assembly.【F:docs/technical_debt_assessment.md†L33-L80】
- Canonical risk configuration now lives under `src/config/risk/risk_config.py`,
  and the evolution configuration resides in `src/core/evolution/engine.py`,
  eliminating the deprecated shim that previously risked misconfiguration once
  real services arrive.【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L43】

## Gap themes

1. **Infrastructure reality** – Provision real Timescale/Redis/Kafka services,
   parameterise SQL queries, and capture operational runbooks.
2. **Runtime discipline** – Adopt the builder abstraction everywhere, supervise
   background jobs, and remove deprecated shims.
3. **Observability** – Extend CI and runtime telemetry to include ingest health,
   cache hit ratios, streaming lag, and failover drills beyond the current
   baseline.【F:docs/ci_baseline_report.md†L8-L27】

## Delivery plan

### Now (0–30 days)

- Complete the security remediation tranche for SQL construction and `eval`
  removal in ingest modules.【F:docs/development/remediation_plan.md†L34-L61】
  - Progress: Real portfolio monitoring now uses managed SQLite connections with
    parameterised statements and typed errors, eliminating blanket exception
    handlers and inline literals in the trading slice’s persistence path.
    【F:src/trading/portfolio/real_portfolio_monitor.py†L1-L572】
- Progress: Data backbone readiness snapshots now capture optional-trigger
  degradation, failover decisions, and Timescale recovery plans with pytest
  coverage, giving operators actionable metadata when optional slices drift
  instead of generic warnings.【F:src/operations/data_backbone.py†L488-L515】【F:tests/operations/test_data_backbone.py†L289-L347】
- Progress: Timescale retention telemetry now aggregates component coverage,
  stamps WARN/FAIL severities, records evaluated policy metadata, and publishes
  via the failover helper so dashboards inherit actionable retention posture
  even during event-bus degradation.【F:src/operations/retention.py†L1-L334】【F:tests/operations/test_data_retention.py†L1-L220】
- Progress: JSONL persistence hardening now raises typed errors for
  unserialisable payloads, logs filesystem failures, and deletes partial files so
  ingest tooling reports genuine write issues instead of silently returning empty
  paths.【F:src/data_foundation/persist/jsonl_writer.py†L1-L69】【F:tests/data_foundation/test_jsonl_writer.py†L1-L37】
- Progress: Parquet ingest writer now guards the pandas DataFrame constructor,
  logs conversion and filesystem errors, and returns explicit sentinels under
  regression coverage so institutional ingest slices capture failed telemetry
  persists rather than silently discarding events.【F:src/data_foundation/persist/parquet_writer.py†L1-L75】【F:tests/data_foundation/test_parquet_writer.py†L1-L93】
- Progress: Core configuration shim now proxies legacy sections to the canonical
  `SystemConfig`, normalising environment overrides and YAML parsing so ingest and
  runtime modules consume the same source of truth while migration off the legacy
  surface continues.【F:src/core/configuration.py†L1-L188】
- Progress: Timescale ingest scheduler now registers with the runtime task
  supervisor, tagging interval/jitter metadata and exposing live snapshots so
  institutional pipelines inherit supervised background jobs instead of orphaned
  tasks, with guardrail coverage spanning steady-state execution, failure
  cut-offs, jitter bounds, supervisor telemetry, snapshot builders, and event
  publishing.【F:src/data_foundation/ingest/scheduler.py†L1-L138】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L200】
- Progress: Timescale ingest regression now covers migrator bootstrap,
  idempotent upserts for empty plans, and macro event ingestion so coverage
  catches silent failures before institutional pipelines depend on them.【F:tests/data_foundation/test_timescale_ingest.py†L1-L213】
- Progress: Timescale backbone orchestrator now emits metadata for requested
  symbols, fetched rows, macro windows, and ingest results on every slice while
  guardrail tests lock macro window fallbacks and zero-payload execution so
  institutional telemetry reflects what was ingested or skipped.【F:src/data_foundation/ingest/timescale_pipeline.py†L70-L213】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L200】
- Progress: Production ingest slice coordinates the orchestrator, provisioner,
  Redis cache, and Kafka bridge behind a supervised `TaskSupervisor`, exposes
  deterministic summaries, and supports scheduler lifecycles under pytest
  coverage so institutional environments inherit a single managed entrypoint
  instead of bespoke ingest wiring.【F:src/data_foundation/ingest/production_slice.py†L1-L170】【F:tests/data_foundation/test_production_ingest_slice.py†L1-L176】
- Progress: Institutional ingest provisioner now spins up supervised Timescale
  schedules alongside Redis caches and Kafka consumers, wiring the bridge into
  the task supervisor with redacted metadata, publishing a managed manifest that
  lists configured topics, and exposing async/sync connectivity probes so
  operators can surface recovery requirements and live health checks without
  bespoke wiring.【F:src/data_foundation/ingest/institutional_vertical.py†L96-L260】【F:tests/runtime/test_institutional_ingest_vertical.py†L86-L262】【F:docs/operations/timescale_failover_drills.md†L1-L27】
- Progress: Tier-0 Yahoo ingest script now sanitises symbols and intervals,
  enforces mutually exclusive period/start-end windows, normalises timestamps,
  and persists via a DuckDB helper that escapes table identifiers and binds
  parameters while the gateway adapter logs rejected fetches, with pytest
  coverage locking the contract for institutional bootstrap datasets.【F:src/data_foundation/ingest/yahoo_ingest.py†L82-L305】【F:src/data_foundation/ingest/yahoo_gateway.py†L1-L53】【F:tests/data_foundation/test_yahoo_ingest_security.py†L1-L132】【F:tests/data_foundation/test_yahoo_gateway.py†L1-L69】
- Progress: Timescale ingest helpers now validate schema/table identifiers at
  construction time and assert the contract under regression coverage so policy
  payloads cannot inject unsafe SQL into institutional ingest jobs.【F:src/data_foundation/persist/timescale.py†L1-L120】【F:tests/data_foundation/test_timescale_ingest.py†L1-L83】
- Wire all runtime entrypoints through `RuntimeApplication` and a task supervisor
  so ingest, cache, and stream jobs are supervised.【F:docs/technical_debt_assessment.md†L33-L56】
- Document current gaps and expected telemetry in updated runbooks and status
  pages (this brief, roadmap, high-impact reports).

### Next (30–90 days)

- Stand up managed Timescale, Redis, and Kafka environments in staging, including
  schema migrations, connection pooling, and credential rotation procedures.
- Implement cache health, ingest quality, and Kafka lag probes with pytest
  coverage and CI export.
- Replace deprecated configuration imports in ingest and runtime modules with
  canonical equivalents to prevent namespace drift.

### Later (90+ days)

- Exercise cross-region failover and batch backfill drills; capture playbooks for
  operators and compliance reviewers.
- Integrate Spark export pipelines and storage retention audits aligned with the
  encyclopedia’s enterprise claims.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L376-L395】
- Automate dead-code sweeps to delete obsolete ingest paths once new services are
  stable.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Risk enforcement work must land concurrently so ingest telemetry feeds policy
  decisions safely.【F:docs/technical_debt_assessment.md†L58-L72】
- Operational readiness initiatives (alert routing, incident response) rely on
  accurate ingest telemetry; coordinate milestone sequencing accordingly.
