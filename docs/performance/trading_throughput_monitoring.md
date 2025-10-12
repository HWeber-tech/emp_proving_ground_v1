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
| `throttle_retry_in_seconds` | Countdown until the trade throttle allows another order (``None`` when no throttle is active). |

The monitor accepts ingestion timestamps from `ingested_at`, `created_at`,
`timestamp`, or `ts` attributes on trade intents. Timestamps may be aware
`datetime` objects, Unix epoch numbers, or ISO-8601 strings.

Execution stats additionally expose aggregate backlog counters so operators can
prove that bursts remain under control:

| Field | Description |
| --- | --- |
| `backlog_breaches` | Total lag breaches recorded since the manager was instantiated. |
| `last_backlog_breach` | ISO timestamp of the most recent backlog breach warning. |

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

### Performance health helper

`TradingManager.assess_performance_health()` bundles throughput, backlog, and
resource thresholds into a single verdict. In addition to the throughput
assessment above it:

- Compares the latest `EventBacklogTracker` snapshot against either the tracker’s
  configured threshold or an override supplied via `backlog_threshold_ms`.
- Validates CPU and memory samples from `ResourceUsageMonitor` using optional
  limits (`max_cpu_percent`, `max_memory_mb`, `max_memory_percent`).
- Surfaces the throttle state (active/reason/message) alongside the health
  summary so operators can see whether rate limits contributed to a slowdown,
  now including remaining trade credits, window utilisation, retry timers,
  and scope metadata so dashboards can display the exact guardrail posture
  without re-parsing the full snapshot payload.

When populated, `assessment["throttle"]` contains:

- `remaining_trades` and `max_trades` for the active scope
- `window_utilisation`, `window_reset_in_seconds`, and `window_reset_at`
- `retry_in_seconds` / `retry_at` countdown values
- The resolved `context`, `scope`, and `scope_key` entries used to evaluate the throttle

The helper returns a dictionary containing the component verdicts (throughput,
backlog, resource) plus an overall `healthy` flag. When resource thresholds are
defined but no process metrics are available, the helper marks the resource
status as `"no_data"` so missing baselines surface in smoke tests.

For evidence packs or runbook snippets, call
`TradingManager.generate_performance_health_report()` which renders the same
assessment as a Markdown report. The renderer mirrors the structure of the
execution performance summary, surfacing throughput bounds, backlog posture,
resource utilisation, and throttle context in a reviewer-friendly format.

### Backlog posture

Event ingest lag also drives a dedicated `backlog` snapshot populated by the new
`EventBacklogTracker`. The payload exposes:

| Field | Description |
| --- | --- |
| `samples` | Count of lag measurements retained in the sliding window. |
| `threshold_ms` | Configured maximum tolerated lag before flagging a breach. |
| `max_lag_ms` / `avg_lag_ms` | Observed extremes and average lag. |
| `latest_lag_ms` | Most recent lag measurement to highlight current posture. |
| `p95_lag_ms` | High-percentile lag for spotting creeping backlog. |
| `breaches` | Number of lag breaches recorded in the current window. |
| `breach_rate` | Fraction of samples breaching the threshold (0–1). |
| `max_breach_streak` | Longest run of consecutive breach samples. |
| `healthy` | `False` when `max_lag_ms` exceeds `threshold_ms`. |
| `last_breach_at` | ISO timestamp of the most recent breach (if any). |
| `worst_breach_ms` | Worst lag breach encountered in the retained window. |

Operators can tighten or relax `threshold_ms` when instantiating the trading
manager. Monitoring the breach counter, breach rate, and streak alongside
throughput makes backlog surges explicit during replay drills. Every breach now returns a
`BacklogObservation`, triggers a warning log, increments the global
`backlog_breaches` counter, and emits a `backlog_breach` experiment event with
the offending lag payload so diaries and dashboards capture the exact moment
latency budgets were exceeded.

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
   that `avg_lag_ms`/`p95_lag_ms` stay near zero with `breach_rate` close to 0
   (indicating no backlog).
3. Call `TradingManager.assess_throughput_health()` with the desired budgets and
   record both the raw metrics and the derived health verdict alongside `backlog`
   and `resource_usage` snapshots when establishing CPU/memory baselines for the
   paper trading environment.
4. For end-to-end paper drills, the simulation helper now persists
   `execution_stats` and `performance_health` in its JSON report, so the CLI
   baseline artifact automatically carries the latest throughput, backlog, and
   resource posture without extra scripting.

The health helper doubles as an ops checklist: automate it in smoke tests so
regressions surface before operators notice backlog on dashboards.

## Configuration Knobs

`TradingManager` lets operators tune monitoring without patching internal
helpers. Supply a custom `throughput_monitor` or `throughput_window` to adjust
the rolling window used by the `ThroughputMonitor`. Similarly, pass either a
pre-built `backlog_tracker` or the `backlog_threshold_ms`/`backlog_window`
overrides to calibrate the lag posture. Reusing a shared
`ResourceUsageMonitor` instance is also supported via the `resource_monitor`
parameter when several managers run inside the same process.

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

### Scope snapshot exports

`TradingManager.get_execution_stats()` now publishes a `trade_throttle_scopes`
list alongside the aggregate throttle snapshot, capturing the latest posture for
each active scope so dashboards and evidence packs can diff per-strategy
utilisation without re-querying the throttle in-process. The manager also
exposes `get_trade_throttle_scope_snapshots()` for callers that need a tuple of
deep-copied payloads, and `collect_performance_baseline()` adds the same data as
`throttle_scopes` so performance dossiers archive the guardrail state observed
during a drill.【F:src/trading/trading_manager.py†L621-L725】【F:src/trading/execution/performance_baseline.py†L52-L74】
Guardrail suites assert the scopes list populates when throttles block, the
baseline helper includes the snapshots, and both surfaces clear the list once
throttling is disabled.【F:tests/trading/test_trade_throttle.py†L171-L210】【F:tests/trading/test_trading_manager_execution.py†L2092-L2138】【F:tests/trading/test_trading_manager_execution.py†L2648-L2666】

### Minimum spacing guardrails

The throttle also enforces a configurable minimum interval between trades via
`min_spacing_seconds`. When set, each scope must wait the specified number of
seconds after a permitted order before another trade is allowed, even if the
rate-limit window still has capacity. Breaches emit a `min_interval` state with
an operator-friendly message such as “Throttled: minimum interval of 30 seconds
between trades…”, plus a `retry_at` timestamp so teams know precisely when the
strategy may resume. Snapshots surface the configured spacing in
`metadata.min_spacing_seconds`, enabling dashboards to contrast per-strategy
cooldowns with observed behaviour. This guardrail satisfies the roadmap’s need
to damp oscillatory bursts while keeping throttle explanations transparent.

Every evaluation also returns a `TradeThrottleDecision.retry_in_seconds`
payload which mirrors the countdown in snapshot metadata and flows into
`TradingManager.get_execution_stats()["throttle_retry_in_seconds"]`. Dashboards
and evidence packs can therefore display a single authoritative timer for when
each strategy regains capacity, whether they inspect the decision diary entry,
live throttle snapshot, or execution stats feed.

### Throttle sizing multipliers

Governance teams can now soften bursty tactics instead of blocking them outright
by setting `TradeThrottleConfig.multiplier`. When present, the trading manager
scales validated trade quantities by the multiplier before reserving exposure
and dispatching the order. The adjustment is recorded as a
`throttle_scaled` experiment event, increments the execution stats via
`throttle_scalings`, and surfaces a `throttle_multiplier` summary on trade
outcomes so runbooks and dashboards capture both the original and adjusted
sizes. Runtime configuration honours the `TRADE_THROTTLE_MULTIPLIER` extra,
and throttle snapshots now expose the configured multiplier for observability.

### Window reset telemetry

Throttle snapshots now report how saturated the rolling window is and when it
will reset. Every evaluation emits a bounded `metadata.window_utilisation`
ratio alongside `metadata.window_reset_at` and
`metadata.window_reset_in_seconds`, giving dashboards a deterministic count-down
to the next trade credit. The instrumentation clamps negative deltas to zero so
reporting remains stable near the boundary, and regression coverage asserts both
UTC scheduling and utilisation maths across rate-limit, cooldown, and minimum
spacing scenarios.【F:src/trading/execution/trade_throttle.py†L204-L312】【F:tests/trading/test_trade_throttle.py†L15-L205】

### Runtime configuration extras

`build_professional_predator_app()` now resolves trade throttle settings straight
from `SystemConfig.extras`. It first honours a JSON blob (`TRADE_THROTTLE_CONFIG`)
or file path (`TRADE_THROTTLE_CONFIG_PATH`), then normalises individual knobs
such as `TRADE_THROTTLE_NAME`, `TRADE_THROTTLE_MAX_TRADES`,
`TRADE_THROTTLE_WINDOW_SECONDS`, `TRADE_THROTTLE_MIN_SPACING_SECONDS`,
`TRADE_THROTTLE_COOLDOWN_SECONDS`, `TRADE_THROTTLE_MULTIPLIER`, and
`TRADE_THROTTLE_SCOPE_FIELDS`. The resulting payload is passed into
`BootstrapRuntime(trade_throttle=…)`, ensuring paper pilots and supervised boots
can dial governance guardrails via environment variables or config files without
touching code, while regression coverage asserts the extras path produces the
expected scope-aware throttle snapshot.【F:src/runtime/predator_app.py†L1723-L2470】【F:src/runtime/bootstrap_runtime.py†L123-L201】【F:tests/runtime/test_trade_throttle_configuration.py†L1-L55】

This instrumentation provides the quantitative evidence required by the roadmap
to demonstrate that throttling keeps the system responsive under bursty trading
conditions.

`build_execution_performance_report()` now renders backlog posture and resource
usage snapshots next to the throttle and throughput sections. The Markdown
report includes threshold values, breach counters, and recent CPU/memory samples
so baseline captures can be dropped straight into runbooks without additional
formatting.
