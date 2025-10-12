# EMP Experimentation Cycle

The experimentation cycle adds a lightweight feedback loop for iterating on strategy ideas. It captures ideas, screens them quickly, and promotes the most promising candidate for a full backtest.

## Workflow overview

1. **Idea ingestion** – New parameter sets are recorded in the SQLite findings memory.
2. **Quick screen** – A minimal data slice is backtested to estimate quick metrics.
3. **Candidate selection** – Screened ideas are ranked via a UCB-lite heuristic.
4. **Full test** – The chosen idea is evaluated with the standard backtest and compared to the baseline.
5. **Progress tracking** – Findings and baselines are persisted for future iterations.

## Findings memory schema

The SQLite database lives at `data/experiments.sqlite` by default and exposes a single table:

| Column | Description |
| --- | --- |
| `id` | Auto-increment primary key |
| `created_at` | Timestamp of insertion |
| `stage` | Current lifecycle stage (`idea`, `screened`, `tested`, `progress`) |
| `params_json` | JSON payload for strategy parameters |
| `novelty` | Novelty score between 0 and 1 |
| `quick_metrics_json` | JSON of quick backtest metrics |
| `quick_score` | Scalar score derived from quick metrics |
| `full_metrics_json` | JSON of full backtest metrics |
| `notes` | Free-form annotations |

## Novelty calculation

Novelty is measured as the cosine distance between an idea's hashed parameter vector and prior entries. Parameters are flattened into sorted `"key=value"` tokens, mapped into eight integer buckets via a rolling hash, and compared using cosine distance. A score of `1.0` means the idea is entirely new; values near `0.0` indicate a close match to existing findings.

## Tuning quick screen and selection

* Quick evaluation uses `score = 0.6 * min(profit_factor, 3) + 0.3 * clip(sharpe, -3, 3) - 0.1 * (max_dd_abs / 100)`.
* The CLI option `--quick-threshold` controls the minimum score required to promote an idea to the screened stage.
* The selection step employs `ucb_lite = quick_score + c * novelty`; adjust `--ucb-c` to trade off exploration vs exploitation.

## Integrating real strategies and data

* Provide a factory function via `EMP_STRATEGY_FACTORY="module:function"` so `emp.core.strategy_factory.make_strategy` can construct real strategies.
* The quick screen receives a lightweight `data_slice` dictionary (`{"days": N, "symbols": [...]}`), which you can adapt inside your factory to load actual price data.
* Override `--slice-days` and `--slice-symbols` to tune the quick screen sample.
* For the full backtest, expose a `full_backtest()` method on your strategy, or fall back to `backtest(None)`.

## Updating the baseline

The full backtest metrics are compared to `data/baseline.json` (created automatically if missing). When a candidate improves the baseline Sharpe ratio without exceeding the baseline max drawdown, the new metrics are atomically written back to the baseline file.

Run a cycle with:

```bash
python -m emp.cli.emp_cycle --ideas-json samples/ideas.json --quick-threshold 0.6 --ucb-c 0.3
```

The command emits concise stage summaries and updates both the findings database and baseline snapshot.
