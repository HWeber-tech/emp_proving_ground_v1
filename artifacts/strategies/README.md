# Strategy Backtest Artifacts

This directory stores deterministic scenario backtest outputs produced by
`scripts/generate_strategy_backtests.py`.  Artifacts are checked into CI to
provide reproducible uplift evidence against the baseline moving average
strategy referenced in the High-Impact Development Roadmap.

Artifacts are JSON documents containing:

- the catalog version used for the evaluation,
- strategy definitions at the time of execution,
- per-scenario expected returns, baseline comparisons, and uplift metrics.

Artifacts can be regenerated locally via:

```
python scripts/generate_strategy_backtests.py
```
