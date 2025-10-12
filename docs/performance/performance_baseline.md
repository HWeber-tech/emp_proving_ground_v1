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

The backlog snapshot now surfaces `latest_lag_ms`, `p95_lag_ms`, `breach_rate`,
and `max_breach_streak` so trend analysis and drift detection can be done from a
single payload.

## CLI utility

A lightweight driver lives in `tools/performance_baseline.py`. Run it with the
project on the Python path to simulate a short burst of intents, trigger the
shared trade throttle, and emit a JSON baseline:

```bash
PYTHONPATH=. python3 tools/performance_baseline.py
```

Sample (truncated) output:

```json
{
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
baseline CPU/lag budgets remain within guardrails and that the trade throttle
activates deterministically under burst conditions. The dedicated
`get_trade_throttle_scope_snapshots()` helper on the trading manager returns the
same per-scope payloads used in the baseline so observability tooling can reuse
the snapshots without rerunning the collection routine.【F:src/trading/trading_manager.py†L621-L725】【F:src/trading/execution/performance_baseline.py†L52-L74】【F:tests/trading/test_trading_manager_execution.py†L2648-L2666】
