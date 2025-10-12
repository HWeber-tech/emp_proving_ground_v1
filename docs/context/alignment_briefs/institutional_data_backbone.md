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
  - Progress: Portfolio monitor cache telemetry now normalises hit/miss metrics,
    derives hit-rate, tags namespace/backing metadata, and records configuration
    status so event-bus consumers inherit deterministic cache evidence even when
    Redis falls back to in-memory, with pytest coverage guarding the payload
    contract.【F:src/trading/monitoring/portfolio_monitor.py†L155-L220】【F:tests/trading/test_portfolio_monitor_cache_metrics.py†L1-L97】
- Progress: Data backbone readiness snapshots now capture optional-trigger
  degradation, failover decisions, and Timescale recovery plans with pytest
  coverage, giving operators actionable metadata when optional slices drift
  instead of generic warnings.【F:src/operations/data_backbone.py†L488-L515】【F:tests/operations/test_data_backbone.py†L289-L347】
- Progress: Developer data backbone presets and env templates now provision the
  local Timescale/Redis/Kafka stack via `docker-compose`, document the setup
  flow, and ship connectivity regressions covering Timescale, Redis, and Kafka
  round-trips so contributors can run institutional pipelines without bespoke
  wiring.【F:docker-compose.yml†L17-L160】【F:config/system/dev_data_backbone.yaml†L1-L20】【F:env_templates/dev_data_services.env†L1-L32】【F:docs/development/setup.md†L40-L128】【F:tests/operations/test_dev_data_services.py†L1-L142】
- Progress: Timescale retention telemetry now aggregates component coverage,
  stamps WARN/FAIL severities, records evaluated policy metadata, and publishes
  via the failover helper so dashboards inherit actionable retention posture
  even during event-bus degradation.【F:src/operations/retention.py†L1-L334】【F:tests/operations/test_data_retention.py†L1-L220】
- Progress: Legacy JSONL persistence shim now raises a guided `ModuleNotFoundError`
  from the package entry point and stub module so ingest flows migrate to the
  governed Timescale writers, with regression coverage asserting imports fail fast.【F:src/data_foundation/persist/__init__.py:9】【F:src/data_foundation/persist/jsonl_writer.py:1】【F:tests/data_foundation/test_jsonl_writer.py:1】
- Progress: Parquet writer shim mirrors the removal guard, raising a descriptive
  `ModuleNotFoundError` and blocking attribute access so institutional pipelines cannot
  silently resurrect the legacy path under pytest coverage.【F:src/data_foundation/persist/__init__.py:14】【F:src/data_foundation/persist/parquet_writer.py:1】【F:tests/data_foundation/test_parquet_writer.py:1】
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
- Progress: RealDataManager now derives Timescale, Redis, and Kafka settings from governance extras, publishes ingest telemetry, exposes cache metrics, and supervises shutdown so production pipelines inherit the managed backbone defaults under regression coverage, with the runbook documenting the end-to-end workflow for operators.【F:src/data_integration/real_data_integration.py†L107-L357】【F:tests/data_integration/test_real_data_manager.py†L83-L198】【F:docs/runbooks/data_foundation.md†L171-L210】
- Progress: Real-data slice tooling now allows either CSV fixtures or provider-backed downloads, with `RealDataSliceConfig` enforcing mutual exclusion, the CLI exposing a `--provider` flag, and integration coverage proving provider fetches hydrate Timescale while preserving supervised ingest telemetry for the sensory organ.【F:src/data_integration/real_data_slice.py†L109-L196】【F:tools/data_ingest/run_real_data_slice.py†L45-L125】【F:tests/integration/test_real_data_slice_ingest.py†L12-L89】
- Progress: Real data slice tooling persists EURUSD fixtures into Timescale, rehydrates the managed caches, emits sensory/belief snapshots, and now ships an operational-backbone CLI so operators can rehearse live ingest evidence or full store→cache→stream drills that also surface belief/regime telemetry, understanding decisions, and ingest-failure diagnostics under integration coverage.【F:src/data_integration/real_data_slice.py†L95-L198】【F:tests/integration/test_real_data_slice_ingest.py†L11-L39】【F:src/data_foundation/pipelines/operational_backbone.py†L82-L366】【F:tools/data_ingest/run_operational_backbone.py†L1-L378】【F:tests/integration/test_operational_backbone_pipeline.py†L198-L295】【F:tests/tools/test_run_operational_backbone.py†L17-L105】
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
  annotations and failure reasons under refreshed pytest coverage.【F:src/data_foundation/ingest/institutional_vertical.py†L132-L420】【F:tools/operations/managed_ingest_connectors.py†L200-L284】【F:tests/runtime/test_institutional_ingest_vertical.py:336】【F:tests/tools/test_managed_ingest_connectors.py†L30-L77】
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
- Progress: Operational backbone pipeline now ships a config-driven factory that hydrates `RealDataManager` from `SystemConfig` extras, resolves Kafka ingest topics, and supervises streaming ingest loops via `TaskSupervisor`; async regressions cover factory wiring, cache metrics, EURUSD rehearsals, and streaming/event-bus shutdown so live onboarding can lean on the same path.【src/data_foundation/pipelines/operational_backbone.py:100】【src/data_foundation/pipelines/operational_backbone.py:401】【tests/data_foundation/test_operational_backbone_factory.py:1】【tests/integration/test_operational_backbone_pipeline.py:145】【tests/integration/test_operational_backbone_pipeline.py:322】
- Wire all runtime entrypoints through `RuntimeApplication` and a task supervisor
  so ingest, cache, and stream jobs are supervised.【F:docs/technical_debt_assessment.md†L33-L56】
- Document current gaps and expected telemetry in updated runbooks and status
  pages (this brief, roadmap, high-impact reports).

### Next (30–90 days)

- Stand up managed Timescale, Redis, and Kafka environments in staging, including
  schema migrations, connection pooling, and credential rotation procedures.
- Progress: A docker compose stack and dotenv template now provision local
  Timescale/Redis/Kafka services with matching `SystemConfig` extras, the
  managed connector CLI loads dotenv overrides, injects extras, provisions Kafka
  topics, and emits Markdown/JSON manifests, and the paired readiness CLI wraps
  connectivity probes with optional failover drills so operators capture a
  single artefact for promotion reviews; the operations quickstart documents the
  workflow end-to-end.【F:docker/institutional-ingest/docker-compose.yml†L1-L67】【F:env_templates/institutional_ingest.env†L1-L24】【F:tools/operations/managed_ingest_connectors.py†L47-L416】【F:tools/operations/institutional_ingest_readiness.py†L1-L246】【F:tests/tools/test_institutional_ingest_readiness_cli.py†L30-L95】【F:docs/operations/managed_ingest_environment.md†L40-L100】
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

## Definition of Done template

**Operational Data Backbone Online.** The deliverable is complete when the managed connector CLI provisions Timescale, Redis, and Kafka via the repository docker-compose stack with all connector health probes returning `status == ok`, the guardrails in `tests/integration/test_operational_backbone_pipeline.py::test_operational_backbone_pipeline_full_cycle` and `tests/operations/test_data_backbone.py::test_evaluate_data_backbone_readiness_combines_signals` exercise a real store→cache→stream cycle without mocks, supervised ingest jobs appear in the TaskSupervisor snapshot and `system_validation_report.json` with zero orphaned tasks, and the operations dashboard together with `docs/operations/managed_ingest_environment.md` capture the updated connection endpoints and retention metrics emitted by `src/operations/data_backbone.py`.

## Dependencies & coordination

- Risk enforcement work must land concurrently so ingest telemetry feeds policy
  decisions safely.【F:docs/technical_debt_assessment.md†L58-L72】
- Operational readiness initiatives (alert routing, incident response) rely on
  accurate ingest telemetry; coordinate milestone sequencing accordingly.
