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

## Baseline Usage

1. Run a high-frequency replay (or paper trading session) and periodically call
   `TradingManager.get_execution_stats()`.
2. Confirm `throughput.avg_processing_ms` remains within your latency budget and
   that `avg_lag_ms` stays near zero (indicating no backlog).
3. Call `TradingManager.assess_throughput_health()` with the desired budgets and
   record both the raw metrics and the derived health verdict alongside CPU/
   memory telemetry when establishing resource baselines for the paper trading
   environment.

The health helper doubles as an ops checklist: automate it in smoke tests so
regressions surface before operators notice backlog on dashboards.

This instrumentation provides the quantitative evidence required by the roadmap
to demonstrate that throttling keeps the system responsive under bursty trading
conditions.
