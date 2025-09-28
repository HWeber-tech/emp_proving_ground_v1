# Strategy Playbooks (Roadmap Alignment)

These playbooks summarise the core strategies delivered in Phase 2 of the
High-Impact Development Roadmap.  They connect the codebase, configuration,
and deterministic scenario backtests introduced in this change so operators can
verify uplift versus the legacy moving-average baseline.

## Catalog Overview

- **Catalog file:** `config/trading/strategy_catalog.yaml`
- **Bootstrap artifact:** `artifacts/strategies/default_scenarios.json`
- **Generation script:** `python scripts/generate_strategy_backtests.py`
- **Registry config:** `config/governance/strategy_registry.yaml`

Each entry defines the canonical capital allocation, configuration parameters,
and governance tags consumed by the SQLite-backed registry
(`src/governance/strategy_registry.py`).

## Strategy Archetypes

### Momentum - Realised Volatility Sizing (`momentum_v1`)
- Targets persistent trends using realised volatility sizing.
- Enabled via catalog with 20-period lookback and 0.55 entry threshold.
- Scenario uplift: outperform baseline in `trend_bull` with >10% expected return.
- Recommended telemetry: momentum score, realised volatility, slippage budget.

### Mean Reversion - Bollinger Style (`mean_rev_v1`)
- Trades contrarian pullbacks with z-score gating and volatility targeting.
- Catalog sets lookback 30, z-score 0.9, leverage cap 1.5.
- Scenario uplift: exceeds baseline on `mean_reversion` overshoot scenario.
- Monitoring: price deviation, realised vol, drawdown triggers.

### Volatility Breakout (`vol_break_v1`)
- Detects compression then breakout using price channel and ATR style filters.
- Catalog uses 10-period breakout window, 30 baseline, multiplier 1.05.
- Scenario uplift: leads `volatility_breakout` scenario with strongest return.
- Ops focus: ensure latency metrics and liquidity guardrails are enabled.

### Multi-Timeframe Momentum (`mtf_momentum_v1`)
- Confirms direction across 15m / 1h / 1d legs with confirmation ratio 0.6.
- Catalog encodes leg weights and volatility timeframe (1d, lookback 20).
- Scenario uplift: positive relative to baseline on `trend_bull` and `trend_bear`.
- Requires data foundation to hydrate multiple timeframes consistently.

## Operating Procedures

1. **Bootstrap catalog** – ensure `config/trading/strategy_catalog.yaml` is
   committed and reviewed with governance.
2. **Generate artifacts** – run `python scripts/generate_strategy_backtests.py`
   after modifying strategy parameters. Commit updated JSON for reproducibility.
3. **Registry sync** – load catalog into the SQLite strategy registry via the
   orchestration layer or CLI, ensuring provenance metadata is stored.
4. **Risk sign-off** – verify expected return uplift matches CI artifact before
   enabling strategies in live or paper environments.

## Encyclopedia References

- Chapter 10 / 24: Execution lifecycle, referencing scenario backtest outputs.
- Chapter 17: Alpha Ops diagnostics – pair with `src/trading/strategies/analytics`.
- Appendix F: Incident management – include strategy identifiers when filing reports.
