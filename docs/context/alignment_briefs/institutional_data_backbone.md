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
  persists rather than silently discarding events.【F:src/data_foundation/persist/parquet_writer.py†L1-L85】【F:tests/data_foundation/test_parquet_writer.py†L1-L93】
- Progress: Retired the legacy core configuration shim; the legacy import path now fails under regression tests while ingestion tooling relies on `SystemConfig.from_yaml`, eliminating the duplicate YAML parser and keeping runtime modules on the canonical loader.【F:tests/current/test_core_configuration_runtime.py†L1-L14】【F:src/governance/system_config.py†L200-L292】【F:tests/governance/test_system_config_yaml.py†L1-L96】
- Progress: Timescale ingest scheduler now registers with the runtime task
  supervisor, tagging interval/jitter metadata and exposing live snapshots so
  institutional pipelines inherit supervised background jobs instead of orphaned
  tasks, with guardrail coverage spanning steady-state execution, failure
  cut-offs, jitter bounds, supervisor telemetry, snapshot builders, and event
  publishing.【F:src/data_foundation/ingest/scheduler.py†L1-L138】【F:tests/data_foundation/test_ingest_scheduler.py†L1-L200】
- Progress: Production ingest slice now serialises concurrent runs behind an
  async lock, preserving result snapshots and invalidation caching while guard
  tests block orchestrator stampedes when multiple triggers fire.【F:src/data_foundation/ingest/production_slice.py†L84-L126】【F:tests/data_foundation/test_production_ingest_slice.py†L133-L190】
- Progress: Timescale ingest regression now covers migrator bootstrap,
  idempotent upserts for empty plans, and macro event ingestion so coverage
  catches silent failures before institutional pipelines depend on them.【F:tests/data_foundation/test_timescale_ingest.py†L1-L213】
- Progress: Timescale backbone orchestrator now emits metadata for requested
  symbols, fetched rows, macro windows, and ingest results on every slice while
  guardrail tests lock macro window fallbacks and zero-payload execution so
  institutional telemetry reflects what was ingested or skipped.【F:src/data_foundation/ingest/timescale_pipeline.py†L70-L213】【F:tests/data_foundation/test_timescale_backbone_orchestrator.py†L1-L200】
- Progress: Production ingest slice coordinates the orchestrator, provisioner,
  Redis cache, and Kafka bridge behind a supervised `TaskSupervisor`, records
  last results/timestamps, invalidates Redis caches after successful runs, and
  exposes deterministic summaries so institutional environments inherit a
  single managed entrypoint instead of bespoke ingest wiring under guardrail
  coverage.【F:src/data_foundation/ingest/production_slice.py†L1-L220】【F:tests/data_foundation/test_production_ingest_slice.py†L1-L220】
- Progress: Institutional ingest provisioner now spins up supervised Timescale
  schedules alongside Redis caches and Kafka consumers, wiring the bridge into
  the task supervisor with redacted metadata, publishing a managed manifest that
  lists configured topics, and exposing async/sync connectivity probes so
  operators can surface recovery requirements and live health checks without
  bespoke wiring. The provisioner now auto-configures Redis via the shared
  client helper when no factory is supplied, warns when a client cannot be
  created, and surfaces the active Redis backing class in service summaries and
  manifests so responders know which cache implementation is live under pytest
  coverage of the configure path.【F:src/data_foundation/ingest/institutional_vertical.py†L96-L399】【F:tests/runtime/test_institutional_ingest_vertical.py†L140-L185】【F:docs/operations/timescale_failover_drills.md†L1-L27】
- Progress: SystemConfig extras now drive Redis cache policy overrides for the
  institutional ingest slice, parsing TTL, capacity, namespace, and invalidation
  prefixes into the resolved config, surfacing the metadata in runtime summaries
  and managed manifests, and propagating the custom policy into supervised Redis
  caches so operators can tune hot datasets without code changes under guardrail
  coverage.【F:src/data_foundation/ingest/configuration.py†L814-L908】【F:src/data_foundation/ingest/production_slice.py†L34-L120】【F:tests/data_foundation/test_timescale_config.py†L130-L150】【F:tests/runtime/test_institutional_ingest_vertical.py†L120-L205】
- Progress: Professional runtime builder now calls the provisioner automatically,
  reuses any managed Redis client already attached to the app, records managed
  connector manifests, propagates the manifest into data-backbone readiness
  telemetry, and defers the first scheduled Timescale run after the bootstrap
  ingest so supervised deployments inherit telemetry-backed connectors without
  extra wiring.【F:src/runtime/runtime_builder.py†L1124-L1895】【F:src/runtime/runtime_builder.py†L2453-L2568】【F:docs/operations/timescale_failover_drills.md†L7-L33】
- Progress: Institutional ingest services now expose `run_failover_drill()` so
  Timescale rehearsals reuse managed connector manifests, normalise requested
  dimensions, honour fallback policies, and attach redacted service summaries,
  while the CLI loads dotenv env files and `--extra` overrides so operators can
  reproduce managed `SystemConfig` payloads without exporting secrets, under
  regression coverage and refreshed drill documentation.【F:src/data_foundation/ingest/institutional_vertical.py†L384-L466】【F:tests/runtime/test_institutional_ingest_vertical.py†L434-L496】【F:tools/operations/run_failover_drill.py†L18-L200】【F:tests/tools/test_run_failover_drill_cli.py†L66-L117】【F:docs/operations/timescale_failover_drills.md†L52-L82】
- Progress: Managed connector summaries now reuse redacted Kafka metadata even
  when consumers fail to provision, normalise boolean/timeout knobs, capture
  consumer group defaults plus configured topics and counts, and expose
  asynchronous `connectivity_report()` probes with timeout-aware error formatting
  so the managed-ingest CLI and readiness dashboards inherit the same health
  annotations and failure reasons under refreshed pytest coverage.【F:src/data_foundation/ingest/institutional_vertical.py†L132-L420】【F:tools/operations/managed_ingest_connectors.py†L200-L259】【F:tests/data_foundation/test_institutional_vertical.py†L309-L351】【F:tests/tools/test_managed_ingest_connectors.py†L30-L77】
- Progress: Institutional ingest services now surface Redis cache metrics in
  runtime summaries and managed manifests, recording namespace, hit, and miss
  telemetry while guarding best-effort collection so operators can audit cache
  effectiveness without shell access under new supervised and production-slice
  coverage.【F:src/data_foundation/ingest/institutional_vertical.py†L256-L337】【F:src/data_foundation/ingest/institutional_vertical.py†L584-L637】【F:tests/runtime/test_institutional_ingest_vertical.py†L138-L210】【F:tests/data_foundation/test_production_ingest_slice.py†L317-L330】
- Progress: `build_institutional_ingest_config` now resolves Redis connection
  settings from extras, tags ingest metadata with the active client name/SSL
  posture, and shares the resolved settings with provisioners plus managed CLI
  tooling so drills, manifests, and production runs report the exact cache
  endpoint without manual overrides under refreshed regression coverage.【F:src/data_foundation/ingest/configuration.py†L728-L909】【F:src/data_foundation/ingest/institutional_vertical.py†L522-L611】【F:tests/data_foundation/test_timescale_config.py†L150-L171】【F:tests/runtime/test_institutional_ingest_vertical.py†L80-L140】【F:tools/operations/managed_ingest_connectors.py†L198-L226】【F:tools/operations/run_production_ingest.py†L255-L318】
- Progress: Tier-0 Yahoo ingest script now sanitises symbols and intervals,
  enforces mutually exclusive period/start-end windows, normalises timestamps,
  and persists via a DuckDB helper that escapes table identifiers and binds
  parameters while the gateway adapter logs rejected fetches, with pytest
  coverage locking the contract for institutional bootstrap datasets.【F:src/data_foundation/ingest/yahoo_ingest.py†L23-L355】【F:src/data_foundation/ingest/yahoo_gateway.py†L1-L53】【F:tests/data_foundation/test_yahoo_ingest_security.py†L1-L132】【F:tests/data_foundation/test_yahoo_gateway.py†L1-L69】
- Progress: Timescale ingest helpers now validate schema/table identifiers at
  construction time and assert the contract under regression coverage so policy
  payloads cannot inject unsafe SQL into institutional ingest jobs.【F:src/data_foundation/persist/timescale.py†L1-L120】【F:tests/data_foundation/test_timescale_ingest.py†L1-L83】
- Progress: Macro calendar ingestion now uses a hardened FRED client that reads
  API keys from dotenv files, normalises release metadata, and returns sorted
  `MacroEvent` payloads so Timescale plans can annotate runs with real economic
  events under pytest coverage of credential fallbacks and HTTP handling.【F:src/data_foundation/ingest/fred_calendar.py†L1-L148】【F:src/data_foundation/ingest/__init__.py†L57-L116】【F:tests/data_foundation/test_fred_calendar.py†L1-L129】
- Progress: Timescale ingestor now reflects tables through SQLAlchemy, streams
  PostgreSQL upserts via `pg_insert`, binds SQLite fallbacks, and chunks writes
  so ingest runs avoid manual SQL while retaining deterministic freshness
  metrics under regression coverage.【F:src/data_foundation/persist/timescale.py†L2337-L2489】【F:tests/data_foundation/test_timescale_ingest.py†L165-L220】
- Wire all runtime entrypoints through `RuntimeApplication` and a task supervisor
  so ingest, cache, and stream jobs are supervised.【F:docs/technical_debt_assessment.md†L33-L56】
- Document current gaps and expected telemetry in updated runbooks and status
  pages (this brief, roadmap, high-impact reports).

### Next (30–90 days)

- Stand up managed Timescale, Redis, and Kafka environments in staging, including
  schema migrations, connection pooling, and credential rotation procedures.
- Progress: A docker compose stack and dotenv template now provision local
  Timescale/Redis/Kafka services with matching `SystemConfig` extras, and the
  managed connector CLI accepts an `--env-file` flag so operators can validate
  connectivity against the stack without exporting secrets globally; operations
  docs capture the workflow alongside failover drill guidance.【F:docker/institutional-ingest/docker-compose.yml†L1-L67】【F:env_templates/institutional_ingest.env†L1-L24】【F:tools/operations/managed_ingest_connectors.py†L20-L159】【F:docs/operations/managed_ingest_environment.md†L1-L73】
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
