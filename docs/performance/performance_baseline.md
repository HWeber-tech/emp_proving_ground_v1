# Performance Baseline Capture

The trading manager now exposes a dedicated helper for snapshotting execution,
throughput, backlog, and throttle posture in a single payload. Use it during
replay drills to prove the "Performance Tuning & Throttle" milestone meets its
definition of done.

## Python helper

```python
from src.trading.execution.performance_baseline import collect_performance_baseline

baseline = collect_performance_baseline(
    manager,
    max_processing_ms=500.0,
    max_lag_ms=500.0,
)
```

The returned dictionary contains:
- `execution`: raw execution statistics (orders, throttle counters, backlog snapshot).
- `throughput`: health verdict derived from the rolling `ThroughputMonitor` window.
- `performance`: full assessment including backlog/resource posture and
  throttle summary.
- `reports`: Markdown renderings that can be pasted directly into ops dossiers.
- `throttle_scopes`: list of per-scope throttle snapshots captured at the time of
  baseline generation for downstream evidence packs.
- `options`: the runtime budgets used for the capture (`max_processing_ms`,
  `max_lag_ms`, CPU/memory guardrails) plus a normalised `trade_throttle`
  configuration distilled from the trading manager so reviewers can see which
  limits and scopes were active without rehydrating the runtime.

The backlog snapshot now surfaces `latest_lag_ms`, `p95_lag_ms`, `breach_rate`,
and `max_breach_streak` so trend analysis and drift detection can be done from a
single payload.

## CLI utility

A lightweight driver lives in `tools/performance_baseline.py`. Run it with the
project on the Python path to simulate a short burst of intents, trigger the
shared trade throttle, and emit a JSON baseline. The CLI accepts knobs for both
the throttle limits and the health budgets:

```bash
PYTHONPATH=. python3 tools/performance_baseline.py \
  --trades 6 \
  --throttle-max-trades 1 \
  --throttle-window-seconds 60 \
  --throttle-max-notional 1000 \
  --max-lag-ms 150 \
  --max-processing-ms 150 \
  --max-cpu-percent 75 \
  --output artifacts/perf_baseline.json
```

Use `--throttle-max-notional` to bound total notional per window; the baseline payload will surface `throttle_consumed_notional`, `throttle_remaining_notional`, and `throttle_notional_utilisation` alongside the existing trade-count fields. The `options.trade_throttle` block is populated automatically with the active throttle name, trade count limits, time window, cooldown/min-spacing settings, multiplier, and scope hints so governance packs inherit the runtime guardrails verbatim.

Sample (truncated) output:

```json
{
  "options": {
    "backlog_threshold_ms": null,
    "max_cpu_percent": 75.0,
    "max_lag_ms": 150.0,
    "max_memory_mb": null,
    "max_memory_percent": null,
    "max_processing_ms": 150.0,
    "trade_throttle": {
      "name": "default",
      "max_trades": 1,
      "window_seconds": 60.0,
      "cooldown_seconds": 15.0,
      "min_spacing_seconds": 2.5,
      "multiplier": 1.0,
      "scope": {
        "strategy_id": "alpha",
        "symbol": "EURUSD"
      }
    },
    "trades": 6
  },
  "execution": {
    "orders_submitted": 1,
    "trade_throttle": {
      "state": "rate_limited",
      "message": "Throttled: too many trades in short time (limit 1 trade per 1 minute)",
      "metadata": {
        "retry_at": "2025-10-12T21:11:19.776944+00:00",
        "window_utilisation": 1.0
      }
    }
  },
  "throughput": {
    "samples": 3,
    "max_processing_ms": 95.315,
    "max_lag_ms": 0.013,
    "healthy": true
  },
  "throttle_scopes": [
    {
      "state": "rate_limited",
      "metadata": {
        "scope": {"strategy_id": "alpha", "symbol": "EURUSD"},
        "remaining_trades": 0
      }
    }
  ],
  "performance": {
    "healthy": true,
    "backlog": {
      "threshold_ms": 250.0,
      "max_lag_ms": 0.013,
      "avg_lag_ms": 0.004,
      "p95_lag_ms": 0.012,
      "latest_lag_ms": 0.001,
      "breach_rate": 0.0,
      "max_breach_streak": 0
    },
    "throttle": {
      "state": "rate_limited",
      "retry_at": "2025-10-12T21:11:19.776944+00:00",
      "context": {"symbol": "EURUSD", "strategy_id": "alpha"}
    }
  },
  "reports": {
    "execution": "# Execution performance summary...",
    "performance": "# Performance health assessment..."
  }
}
```

Operators can archive the Markdown reports alongside the JSON payload to show
which throttle and resource budgets were used (`options.*` in the payload),
prove CPU/lag thresholds remain within guardrails, and demonstrate that the
trade throttle activates deterministically under burst conditions. The
dedicated `get_trade_throttle_scope_snapshots()` helper on the trading manager
returns the same per-scope payloads used in the baseline so observability
tooling can reuse the snapshots without rerunning the collection
routine.【F:src/trading/trading_manager.py†L542-L569】【F:src/trading/trading_manager.py†L2614-L2619】【F:src/trading/execution/performance_baseline.py†L52-L74】【F:tests/trading/test_trading_manager_execution.py†L2860-L2923】
