# Data Foundation Runbook

This runbook captures the operational workflows that back the roadmap's
"Data Foundation Hardening" track.  The goal is to ensure every developer and
operator can hydrate, inspect, and validate datasets without bespoke scripts.

## Canonical Pricing Pipeline

* Entrypoint: `scripts/data_bootstrap.py`
* Core logic: `src/data_foundation/pipelines/pricing_pipeline.py`

The bootstrap CLI wraps the `PricingPipeline` so Yahoo, Alpha Vantage, and
deterministic file vendors share the same normalisation and validation flow.
It persists artefacts via `PricingCache`, which writes Parquet (with CSV
fallback) plus JSON metadata into `data_foundation/cache/pricing/`.  Retention
and entry limits can be configured from the CLI (`--cache-retention-days` and
`--cache-max-entries`).

Common usage:

```bash
poetry run python scripts/data_bootstrap.py \
  --symbols "EURUSD=X,GBPUSD=X" \
  --vendor yahoo \
  --lookback-days 120 \
  --cache-retention-days 7
```

The command prints a JSON summary with dataset location, row counts, and any
quality issues raised by the pipeline validators.

## Multi-source Aggregator

* Module: `src/data_foundation/ingest/multi_source.py`

The `MultiSourceAggregator` coordinates Yahoo Finance, Alpha Vantage, and FRED
fetchers so Tier-0 bootstrap datasets inherit the same schema and quality
checks.  Each provider is declared via a `ProviderSpec` that references a fetch
callable.  The aggregator normalises the resulting frames (UTC timestamps,
uppercase symbols), stitches gaps in priority order, and executes the
roadmap-required validators:

- `CoverageValidator` – ensures each symbol hits minimum bar coverage over the
  requested window.
- `StalenessValidator` – checks the freshest observation versus the requested
  end timestamp.
- `CrossSourceDriftValidator` – surfaces price divergence between providers for
  overlapping candles.

Usage inside notebooks or scripts mirrors the following pattern:

```python
from datetime import datetime, timedelta, timezone

from src.data_foundation.ingest.multi_source import (
    CoverageValidator,
    CrossSourceDriftValidator,
    MultiSourceAggregator,
    ProviderSpec,
    StalenessValidator,
)

aggregator = MultiSourceAggregator(
    providers=[
        ProviderSpec("yahoo", fetch=yahoo_fetch),
        ProviderSpec("alpha_vantage", fetch=alpha_fetch),
        ProviderSpec("fred", fetch=fred_fetch),
    ],
    validators=[
        CoverageValidator(frequency="1D"),
        StalenessValidator(max_staleness=timedelta(days=2)),
        CrossSourceDriftValidator(tolerance=0.01),
    ],
)

result = aggregator.aggregate(["EURUSD"], start=datetime(2024, 1, 1, tzinfo=timezone.utc), end=datetime.now(timezone.utc))
dataset = result.data
quality = result.quality_findings
```

The returned `AggregationResult` includes the consolidated dataframe, per-provider
snapshots, and validator findings so CI pipelines and dashboards can publish the
same diagnostics captured in the roadmap checklist.

### Validation diagnostics & failure handling

Quality findings expose a `severity` of `ok`, `warning`, or `error` via
`DataQualitySeverity`. The highest severity is surfaced through
`AggregationResult.status` so orchestrators can gate downstream steps.

- `CoverageValidator` computes per-symbol coverage versus the requested window
  (`warn_ratio=0.9`, `error_ratio=0.75`). A warning means at least one symbol
  dropped below 90% coverage—investigate vendor throttling or missing trading
  days before promoting the dataset. An error (<75%) must block distribution;
  rerun the fetch, widen the window, or fall back to cached data.
- `StalenessValidator` compares each symbol’s latest candle with
  `max_staleness`. A warning signals drift toward the threshold; schedule a
  refill or verify the market was closed. An error is emitted when no data is
  available or the latest candle exceeds the limit—treat the window as stale
  and escalate to operations or switch to another provider.
- `CrossSourceDriftValidator` measures relative close-price drift between
  overlapping provider candles. When drift exceeds `warn_tolerance` (defaults to
  the hard `tolerance`), examine the affected timestamps for vendor-specific
  adjustments. Any error indicates the drift crossed the hard limit and the
  dataset should be quarantined until providers reconcile.

For CI or scheduled jobs, fail fast by checking `result.status` and aborting on
`DataQualitySeverity.error`. When running `scripts/data_bootstrap.py`, supply
`--fail-on-quality` to exit with status 2 if the pricing pipeline surfaces error
level issues so stale datasets never land in cache.

### Pricing pipeline guardrails

`PricingPipeline._validate_frame` issues structured `PricingQualityIssue`
records. Only `no_data` is flagged as `error`; other checks (`duplicate_rows`,
`missing_rows`, `flat_prices`, `stale_series`) are `warning`s that include
symbol-level context. Warnings should trigger manual review but still allow the
run to proceed. When `--fail-on-quality` is set, the CLI aborts if any
error-severity issues are present. Without the flag, the JSON summary emitted by
the CLI still lists all findings so dashboards and notebooks can surface them.

### Timescale ingest scoring

`evaluate_ingest_quality` grades Timescale backbone runs by combining coverage
and freshness into a per-dimension score. Scores below the default warn (0.85)
or error (0.6) thresholds escalate the aggregate `IngestQualityStatus`. Freshness
breaches append explanatory messages, and any anomalies detected by
`detect_feed_anomalies` are attached to the relevant dimension metadata.

- `status == ok`: publish telemetry as normal.
- `status == warn`: proceed, but raise a backlog ticket and monitor subsequent
  runs—look for declining coverage ratios or increasing freshness drift.
- `status == error`: fail the deployment gate or halt the ingest job, repair the
  upstream source, and rerun the plan with fresh expectations before resuming
  downstream processing.

## Streaming Latency Benchmarking

* Module: `src/data_foundation/streaming/latency_benchmark.py`

The `StreamingLatencyBenchmark` collects producer and consumer timestamps to
measure end-to-end ingest latency without binding to a specific Kafka client.
CI jobs and notebooks can record samples with `record()` or `extend()` and then
call `summarise()` to obtain percentile statistics for dashboards or
readiness reviews.  The resulting `LatencyBenchmarkReport` serialises cleanly to
JSON so operators can archive latency regressions alongside ingest quality
artifacts.

Example usage when validating a new streaming consumer:

```python
from datetime import datetime, timezone

from src.data_foundation.streaming import StreamingLatencyBenchmark

benchmark = StreamingLatencyBenchmark()
producer_ts = datetime.now(timezone.utc)

# After consuming a payload...
benchmark.record(producer_ts, dimension="daily_bars", metadata={"topic": "market.prices"})
report = benchmark.summarise()
print(report.as_dict())
```

## Feed Anomaly Detection

* Module: `src/data_foundation/ingest/anomaly_detection.py`

Roadmap Workstream 3A emphasises proactive detection of data feed breaks and
false ticks.  The `detect_feed_anomalies` helper inspects
`TimescaleIngestResult` snapshots to flag stalled or stale dimensions, while
`detect_false_ticks` performs robust statistical checks (median + MAD) to
identify outlier candles before they poison downstream analytics.  The
functions return structured dataclasses that integrate with
`evaluate_ingest_quality`, ensuring anomaly metadata propagates into CI
artifacts and operational dashboards without bespoke glue code.

## Reference Data Loader

* Module: `src/data_foundation/reference/reference_data_loader.py`

The `ReferenceDataLoader` consolidates instrument metadata, trading sessions,
and holiday calendars.  It defaults to the repository configs:

* Instruments — `config/system/instruments.json`
* Sessions — `config/reference/sessions.json`
* Holidays — `config/reference/holidays.json`

The loader memoizes results and exposes a `refresh` flag when changes need to
be detected at runtime.  Use it to seed services such as the strategy registry
or risk engine with a single, validated source of truth:

```python
from src.data_foundation.reference import ReferenceDataLoader

loader = ReferenceDataLoader()
dataset = loader.load_all()
instrument = dataset.instruments["EURUSD"]
```

Custom deployments can override the config paths when integrating with managed
data stores.

## Macro Event Calendar

* Module: `src/data_foundation/ingest/fred_calendar.py`

`fetch_fred_calendar(start, end, api_key=None, session=None)` wraps the FRED
calendar endpoint so ingest plans can attach real macro release events to
Timescale runs.  The helper:

* Pulls credentials from the optional `api_key` argument or `FRED_API_KEY`
  environment variable.
* Normalises release metadata (calendar name, event name, currency, importance)
  and parses timestamps into timezone-aware datetimes.
* Sorts results and returns a list of `MacroEvent` dataclasses guarded by
  regression tests for missing credentials, HTTP failures, and payload parsing.

Pass the function to ingest pipelines or schedule it alongside other macro
fetchers; when credentials are absent the helper emits a warning and returns an
empty list so CI and local runs remain deterministic.

## Operational Timescale Backbone

* Module: `src/data_integration/real_data_integration.py`

`RealDataManager.ingest_market_slice()` wires TimescaleDB, Redis caching, and
Kafka telemetry into a single call so operators can hydrate the production
backbone without bespoke scripts.  Supply the target symbols alongside optional
intraday and macro parameters; the helper normalises symbols, executes the
Timescale ingest plan, and invalidates cached query results when new rows land
in the warehouse.【F:src/data_integration/real_data_integration.py†L213-L314】

Use `python -m tools.data_ingest.run_operational_backbone` to rehearse the
same store→cache→stream drill without writing glue code. The CLI loads
`SystemConfig` from environment variables, optional dotenv files, or
`--extra` overrides, promotes the config to institutional backbone mode, and
emits JSON/Markdown summaries covering ingest rows, cache metrics, Kafka
events, sensory snapshots, belief/regime telemetry, understanding decisions,
and any ingest failures encountered during fallback. Tests patch the pipeline
to assert symbols, connection metadata, and Markdown rendering so the
runbook’s workflow stays reproducible.【F:tools/data_ingest/run_operational_backbone.py†L1-L378】【F:tests/tools/test_run_operational_backbone.py†L17-L105】
The pipeline inspects composite Kafka publishers to derive expected
dimensions, synthesises telemetry events for ingest results that lack a broker
echo, reorders the final event list, and threads cache metrics plus ingest
errors into the validation snapshot so evidence stays complete even if
streaming pauses.【F:src/data_foundation/pipelines/operational_backbone.py†L305-L499】【F:src/runtime/runtime_builder.py†L1671-L1770】
Latest iteration also attaches the CLI and pipeline to a shared `TaskSupervisor`
so rehearsal workloads are registered and cancelled cleanly, adds
`task_supervision`/`streaming_snapshots` blocks to the JSON and Markdown outputs,
and records the supervised roster whenever pipeline results are generated; guardrail
tests assert the supervisor list surfaces in both output formats and that the CLI
invokes `cancel_all()` during shutdown.【F:tools/data_ingest/run_operational_backbone.py†L400-L520】【F:src/data_foundation/pipelines/operational_backbone.py†L120-L566】【F:tests/tools/test_run_operational_backbone.py†L71-L150】
Connectivity probes now flow through the same result payload: the pipeline
captures `RealDataManager.connectivity_report()` and the CLI renders a
`connectivity_report` block in JSON/Markdown so reviewers see Timescale/Redis/Kafka
status alongside ingest metrics, with regression tests covering probe wiring and
rendered summaries.【F:src/data_foundation/pipelines/operational_backbone.py†L513-L749】【F:tools/data_ingest/run_operational_backbone.py†L401-L482】【F:tests/tools/test_run_live_shadow.py†L200-L260】
Set `KAFKA_INGEST_ENABLE_STREAMING=false` when rehearsing without a Kafka
broker; the institutional provisioner will still publish ingest telemetry and
surface topic metadata, but the supervised bridge remains idle so local runs
do not stall waiting for a consumer.【F:src/data_foundation/ingest/configuration.py†L872-L948】【F:src/data_foundation/ingest/institutional_vertical.py†L790-L860】【F:tests/runtime/test_institutional_ingest_vertical.py†L570-L596】

`TimescaleConnectionSettings.from_mapping()` now honours optional pooling
overrides exposed via `SystemConfig.extras` (`TIMESCALEDB_POOL_SIZE`,
`TIMESCALEDB_MAX_OVERFLOW`, `TIMESCALEDB_POOL_TIMEOUT`, `TIMESCALEDB_POOL_RECYCLE`,
`TIMESCALEDB_POOL_PRE_PING`). The overrides only apply when the URL targets a
PostgreSQL/Timescale backend, leaving SQLite fallbacks unchanged, and are
propagated directly to SQLAlchemy so production rehearsals can tune pool sizing
without editing code; regression coverage exercises the Postgres vs SQLite
paths.【F:src/data_foundation/persist/timescale.py†L111-L255】【F:tests/data_foundation/test_timescale_connection_settings.py†L18-L122】

When `TIMESCALEDB_URL` is absent the same helper now assembles a connection URL
from discrete credentials (`TIMESCALEDB_HOST`, `TIMESCALEDB_PORT`,
`TIMESCALEDB_DB`, `TIMESCALEDB_USER`, `TIMESCALEDB_PASSWORD`) and optional
Postgres-style TLS hints (`TIMESCALEDB_SSLMODE`, `TIMESCALEDB_SSLROOTCERT`,
`TIMESCALEDB_SSLKEY`, `TIMESCALEDB_SSLCERT`).  This keeps configuration aligned
with managed Timescale deployments that distribute host/port secrets via
environment injection while still allowing a prebuilt URL to take precedence;
tests lock the new assembly path and the explicit-URL override behaviour.【F:src/data_foundation/persist/timescale.py†L130-L210】【F:tests/data_foundation/test_timescale_connection_settings.py†L92-L122】

When `SystemConfig.data_backbone_mode` is `institutional`, the operational
backbone factory automatically requires Timescale and Redis extras and, when
`KAFKA_INGEST_ENABLE_STREAMING` resolves truthy (accepting `1/true/yes/on` or
`0/false/no/off` forms), also requires Kafka connection settings. Missing
prerequisites raise targeted `RuntimeError`s so misconfigured drills fail fast
with actionable messages; tests cover fakeredis-managed caches, missing
Timescale URLs, and streaming-on Kafka requirements.【F:src/data_foundation/pipelines/operational_backbone.py†L49-L910】【F:tests/data_foundation/test_operational_backbone_factory.py†L87-L239】

Need a smaller rehearsal? `python tools/data_ingest/run_real_data_slice.py`
now accepts either a CSV fixture or a `--provider` flag (for example,
`--provider yahoo --symbol EURUSD`) so operators with market-data API access
can hydrate Timescale without bundling local files. The helper enforces
mutually exclusive CSV/provider configuration, derives sensible defaults for
symbols and source labels, and surfaces ingest/belief snapshots identical to
the automation path under regression coverage.【F:tools/data_ingest/run_real_data_slice.py†L45-L125】【F:tests/integration/test_real_data_slice_ingest.py†L12-L89】

```python
from datetime import datetime, timezone

from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_integration.real_data_integration import RealDataManager

settings = TimescaleConnectionSettings.from_mapping({
    "TIMESCALEDB_URL": "postgresql+psycopg://trader:secret@localhost:5432/timescale",
})
manager = RealDataManager(timescale_settings=settings)

results = manager.ingest_market_slice(
    symbols=["EURUSD", "GBPUSD"],
    daily_lookback_days=90,
    intraday_lookback_days=3,
    intraday_interval="5m",
    macro_start="2024-03-01",
    macro_end="2024-03-08",
)
print({dimension: result.rows_written for dimension, result in results.items()})

bars = manager.fetch_data("EURUSD", interval="1m", end=datetime.now(timezone.utc))
print(manager.cache_metrics())
```

`cache_metrics(reset=True)` exposes Redis hit/miss counters so the ingest team
can confirm the roadmap’s “store → cache → retrieve” flow.  Call `shutdown()`
or `close()` when the run completes to dispose of Timescale engines and Redis
clients cleanly.【F:src/data_integration/real_data_integration.py†L292-L357】

`RealDataManager.connectivity_report()` surfaces Timescale, Redis, and Kafka
status in a single payload by running inexpensive probes (including the
producer `checkup()` helper) so operators can gate institutional toggles on
verified connectivity; regression coverage exercises both default and
fully-provisioned stacks.【F:src/data_integration/real_data_integration.py†L121-L605】【F:src/data_foundation/streaming/kafka_stream.py†L1374-L1384】【F:tests/data_integration/test_real_data_manager.py†L263-L299】

## Artefact Expectations

* Parquet/CSV datasets stored under `data_foundation/cache/pricing/`
* Metadata JSON files suffixed with `_metadata.json`
* Issues JSON files suffixed with `_issues.json`
* Reference data lives under `config/reference/`

All artefacts are Git-ignored and regenerated on demand; the runbook ensures a
repeatable workflow for onboarding new venues or data vendors without drifting
from the encyclopedia baseline.

## Data Lineage & SLA Source of Truth

Regulated deployments require auditable lineage of every dataset consumed by
strategies and risk controls. Generate the authoritative markdown snapshot via
`python scripts/generate_data_lineage.py`. The script populates
`docs/deployment/data_lineage.md` with freshness, completeness, and retention
expectations for each layer of the ingestion stack so operational reviews can
reference a single, version-controlled document.

## Encyclopedia Cross-References

The EMP Encyclopedia outlines which data vendors underpin the free Tier-0
bootstrap versus premium professional feeds. Cross-linking these tables keeps
runbook guidance aligned with the canonical operating model:

- Tier-0 free sources (Yahoo, Alpha Vantage, FRED) and their rate limits guide
  default bootstrap expectations.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L6864-L6881】
- Tier-1/Tier-2 premium feeds (IC Markets FIX, Alpha Vantage Premium, Refinitiv,
  Bloomberg) establish the upgrade path operators should plan for when moving
  beyond bootstrap mode.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L6882-L6903】

Operators updating vendor configurations should reference these tables when
raising change requests so budgeting, latency targets, and entitlement checks
match the encyclopedia narrative.
