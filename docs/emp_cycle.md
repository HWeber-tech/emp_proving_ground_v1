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
| `notes` | Free-form annotations (includes run metadata and failure reasons) |
| `params_hash` | SHA1 hash of the sorted params JSON used for de-duplication |
| `params_vec` | JSON list of eight integers representing the cached novelty vector |

Identical ideas (same `params_hash`) are de-duplicated while in the `idea` or `screened` stages. Legacy rows are lazily backfilled with hash/vector information the first time they are touched.

## Novelty calculation

Novelty is measured as the cosine distance between an idea's hashed parameter vector and the most similar recent entries (bounded to the latest ~5k rows). Parameters are flattened into sorted `"key=value"` tokens, mapped into eight integer buckets via a rolling hash, and compared using cosine distance. Cached vectors stored in `params_vec` keep the computation fast even with a large history. A score of `1.0` means the idea is entirely new; values near `0.0` indicate a close match.

## Tuning quick screen and selection

* Quick evaluation uses `score = 0.6 * min(profit_factor, 3) + 0.3 * clip(sharpe, -3, 3) - 0.1 * (max_dd_abs / 100)`.
* The CLI option `--quick-threshold` controls the minimum score required to promote an idea to the screened stage.
* The selection step employs `ucb_lite = quick_score + c * novelty`; adjust `--ucb-c` to trade off exploration vs exploitation.

## Progress KPI configuration

The promotion decision is configurable:

* `--kpi` selects the primary metric (default: `sharpe`).
* `--kpi-threshold` enforces an absolute minimum for the KPI instead of comparing to the baseline.
* `--risk-max-dd` constrains absolute drawdown (e.g., `25` for 25%).
* `--kpi-secondary` accepts additional gates such as `winrate:>=:0.55`; specify multiple flags for multiple gates.

The baseline (`data/baseline.json`) is updated only when the configured KPI signals progress and all constraints succeed.

## Timeouts, metadata, and failure notes

* `--full-timeout-secs` (default: 1200s) bounds the full backtest duration. Timeouts and exceptions mark the idea as `tested` with a `full_eval_error:<reason>` note instead of aborting the cycle.
* `--seed` and `--git-sha` embed reproducibility metadata directly in the findings notes. The CLI auto-detects the git SHA when possible.

## Integrating real strategies and data

* Provide a factory function via `EMP_STRATEGY_FACTORY="module:function"` so `emp.core.strategy_factory.make_strategy` can construct real strategies.
* The quick screen relies on `emp.core.data_slice.make_slice` which produces a dictionary containing symbols, start/end timestamps, and duration. Adapt this payload inside your factory to load real data.
* Override `--slice-days` and `--slice-symbols` to tune the quick screen sample.
* For the full backtest, expose a `full_backtest()` method on your strategy, or fall back to `backtest(None)`.

## Database hygiene

Use `python -m emp.cli.emp_db_tools prune --keep-days 90 --stages idea,screened` to remove stale rows and `python -m emp.cli.emp_db_tools vacuum` to reclaim disk space.

## Recipes

* **Sharpe with strict drawdown**: `python -m emp.cli.emp_cycle ... --kpi sharpe --risk-max-dd 15`
* **CAR/MDD focus**: set `--kpi car_mdd --kpi-threshold 1.1` and ensure your strategy reports the metric.
* **Win-rate gate**: `--kpi-secondary winrate:>=:0.55`
* **Monthly cleanup**: `python -m emp.cli.emp_db_tools prune --keep-days 30 --stages idea,screened`

Run a cycle with:

```bash
python -m emp.cli.emp_cycle --ideas-json samples/ideas.json --quick-threshold 0.6 --ucb-c 0.3 --kpi sharpe --risk-max-dd 20
```

The command emits concise stage summaries and updates both the findings database and baseline snapshot.
