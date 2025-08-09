WHY Configuration Guide
=======================

Location: `config/why/why_engine.yaml`

- enable_macro_proximity: enable simple proximity-to-macro-event signal.
- enable_yields: enable yield-curve derived features and signal.
- weights:
  - macro: weight for macro proximity in WHY composite
  - yields: weight for yields signal in WHY composite
- yield_features:
  - slope_2s10s, slope_5s30s: term structure slopes
  - curvature_2_10_30: classic curvature 2*10Y - 2Y - 30Y
  - parallel_shift: average level across available tenors

CLI overrides (backtest):

```
python scripts/backtest_report.py --why-weight-yields 0.7 --why-weight-macro 0.3 --disable-why-macro
```

Artifacts:
- Backtest writes `why_features.csv` and `why_features.jsonl` to `docs/reports/backtests/`.

