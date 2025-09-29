# Evolution Lab Leaderboard

_Auto-generated on 2025-09-28 19:01:38Z using `scripts/generate_evolution_lab.py`._

## Current Experiments

| Experiment | Seed | Fitness | Sharpe | Sortino | Max Drawdown | Total Return | Short | Long | Risk | VaR Guard | Drawdown Guard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma_crossover_ga | 2025 | 4.166 | 3.916 | 5.980 | 0.010 | 0.131 | 8 | 169 | 0.36 | ✅ | ✅ |

## Reproducibility Artifacts

- Manifest: `artifacts/evolution/ma_crossover/manifest.json`
- Dataset: `artifacts/evolution/ma_crossover/dataset.csv`
- Generation history: `artifacts/evolution/ma_crossover/generation_history.csv`

## Follow-on Backlog

- [ ] Introduce speciation and diversity preservation experiments.
- [ ] Evaluate Pareto-front selection for multi-objective fitness.
- [ ] Swap synthetic datasets with live market snapshots for benchmarking.
- [ ] Automate nightly leaderboard refresh with CI artifact publishing.
- [x] Integrate promoted genomes into the strategy registry feature flags.
