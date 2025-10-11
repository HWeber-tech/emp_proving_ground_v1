# Trading Throughput Monitoring Baseline

AlphaTrade's trade throttle now exposes richer performance context via the
`ThroughputMonitor`. The monitor captures each trade intent's processing time
and the lag between ingestion and execution handling. These metrics help
validate the roadmap's "Performance Tuning & Throttle" objective by making event
loop backlog visible and providing data for CPU/memory profiling sessions.

## Metrics Emitted

`TradingManager.get_execution_stats()` now includes a `throughput` payload with
the following keys:

| Field | Description |
| --- | --- |
| `samples` | Number of intents observed in the rolling window (default 256). |
| `avg_processing_ms` | Mean processing latency in milliseconds. |
| `p95_processing_ms` | 95th percentile processing latency. |
| `max_processing_ms` | Maximum processing latency. |
| `avg_lag_ms` | Average delay between ingest timestamp and processing start. |
| `max_lag_ms` | Maximum ingest delay detected. |
| `throughput_per_min` | Approximate processed intents per minute across the window. |

The monitor accepts ingestion timestamps from `ingested_at`, `created_at`,
`timestamp`, or `ts` attributes on trade intents. Timestamps may be aware
`datetime` objects, Unix epoch numbers, or ISO-8601 strings.

`TradingManager.assess_throughput_health()` consumes the snapshot and evaluates
latency budgets:

| Field | Description |
| --- | --- |
| `healthy` | `True` when both processing and lag stay within configured limits. |
| `processing_within_limit` | Whether the observed max processing latency is under `max_processing_ms`. |
| `lag_within_limit` | Whether ingest lag stays under `max_lag_ms`. |
| `max_processing_ms` | Highest processing latency seen in the sampled window. |
| `max_lag_ms` | Highest ingest lag observed. |
| `samples` | Number of intents contributing to the assessment. |

Use tighter bounds for high-frequency drills (e.g. 100 ms processing, 500 ms lag)
and widen them for slower venues.

### Backlog posture

Event ingest lag also drives a dedicated `backlog` snapshot populated by the new
`EventBacklogTracker`. The payload exposes:

| Field | Description |
| --- | --- |
| `samples` | Count of lag measurements retained in the sliding window. |
| `threshold_ms` | Configured maximum tolerated lag before flagging a breach. |
| `max_lag_ms` / `avg_lag_ms` | Observed extremes and average lag. |
| `breaches` | Number of lag breaches recorded in the current window. |
| `healthy` | `False` when `max_lag_ms` exceeds `threshold_ms`. |
| `last_breach_at` | ISO timestamp of the most recent breach (if any). |
| `worst_breach_ms` | Worst lag breach encountered in the retained window. |

Operators can tighten or relax `threshold_ms` when instantiating the trading
manager. Monitoring the breach counter alongside throughput makes backlog
surges explicit during replay drills.

### Resource usage snapshot

The `resource_usage` entry captures CPU and memory data for the trading process:

| Field | Description |
| --- | --- |
| `timestamp` | Sampling time in ISO-8601 format (UTC). |
| `cpu_percent` | Instantaneous CPU utilisation reported by `psutil`. |
| `memory_mb` | Resident set size in megabytes. |
| `memory_percent` | Fraction of system memory used by the process. |

If `psutil` is unavailable the monitor falls back to `None` values, preserving a
consistent schema so baseline scripts can still emit records.

## Baseline Usage

1. Run a high-frequency replay (or paper trading session) and periodically call
   `TradingManager.get_execution_stats()`.
2. Confirm `throughput.avg_processing_ms` remains within your latency budget and
   that `avg_lag_ms` stays near zero (indicating no backlog).
3. Call `TradingManager.assess_throughput_health()` with the desired budgets and
   record both the raw metrics and the derived health verdict alongside `backlog`
   and `resource_usage` snapshots when establishing CPU/memory baselines for the
   paper trading environment.

The health helper doubles as an ops checklist: automate it in smoke tests so
regressions surface before operators notice backlog on dashboards.

### Scoped throttles for hot strategies

`TradeThrottle` now supports *scoped* rate limits via the `scope_fields`
configuration option. When present, the throttle keeps independent rolling
windows per unique combination of the supplied metadata fields (for example a
`("strategy_id",)` scope). This prevents a single hot strategy from consuming
the entire global quota during bursts while still enforcing per-strategy caps.
Throttle snapshots include both the resolved `scope_key` and a human readable
`scope` map so diaries and dashboards surface which strategy tripped the guard.
Missing metadata gracefully collapses to a shared "None" scope, ensuring events
without identifying context remain governed.

This instrumentation provides the quantitative evidence required by the roadmap
to demonstrate that throttling keeps the system responsive under bursty trading
conditions.
