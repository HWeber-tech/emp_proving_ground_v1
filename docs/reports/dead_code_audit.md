 codex/assess-technical-debt-in-codebase
# Dead code audit – 2025-09-16 17:16 UTC

# Dead code audit – 2025-09-16 06:33 UTC
 main

*Generated via `vulture src tests --min-confidence 80 --sort-by-size`.*

## Summary

 codex/assess-technical-debt-in-codebase
- **Total candidates**: 8
- **Command exit status**: vulture exited with status 3; review stderr below.
- **By symbol type**: 6 variable(s), 2 import(s)
- **Top modules**: `src.operational.metrics_registry` (3), `src.core.interfaces.base` (1), `src.data_foundation.config.vol_config` (1), `src.intelligence.portfolio_evolution` (1), `src.risk.risk_manager_impl` (1)

- **Total candidates**: 16
- **Command exit status**: vulture exited with status 3; review stderr below.
- **By symbol type**: 9 import(s), 6 variable(s), 1 unreachable(s)
- **Top modules**: `src.operational.metrics_registry` (3), `src.thinking.adversarial.red_team_ai` (3), `src.core.interfaces` (1), `src.core.strategy.templates.mean_reversion` (1), `src.core.strategy.templates.moving_average` (1)
 main

## Top candidates

| Confidence | Size | Type | Module | Object | Line |
| --- | --- | --- | --- | --- | --- |
 codex/assess-technical-debt-in-codebase
| 100 | 1 | variable | src.core.interfaces.base | constraints | 76 |
| 100 | 1 | variable | src.data_foundation.config.vol_config | stream | 12 |
| 90 | 1 | import | src.intelligence.portfolio_evolution | _nn | 21 |

| 100 | 1 | variable | src.core.interfaces | constraints | 89 |
| 90 | 1 | import | src.core.strategy.templates.mean_reversion | SupportsInt | 4 |
| 90 | 1 | import | src.core.strategy.templates.moving_average | SupportsInt | 4 |
| 100 | 1 | variable | src.data_foundation.config.vol_config | stream | 12 |
| 90 | 1 | import | src.intelligence.competitive_intelligence | torch | 161 |
| 90 | 1 | import | src.intelligence.portfolio_evolution | _nn | 21 |
| 90 | 1 | import | src.operational.metrics | _MetricsSinkBase | 277 |
 main
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 66 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 69 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 75 |
| 100 | 1 | variable | src.risk.risk_manager_impl | constraints | 272 |
 codex/assess-technical-debt-in-codebase
| 90 | 1 | import | tests.current.test_parity_checker | parity_module | 106 |

| 90 | 1 | import | src.thinking.adversarial.market_gan | StrategyTestResult | 24 |
| 90 | 1 | import | src.thinking.adversarial.red_team_ai | AttackResult | 18 |
| 90 | 1 | import | src.thinking.adversarial.red_team_ai | ExploitResult | 18 |
| 90 | 1 | import | src.thinking.adversarial.red_team_ai | StrategyAnalysis | 18 |
| 100 | 209 | unreachable | src.trading.portfolio.real_portfolio_monitor | after return | 390 |
 main

## Observations

- Vulture heuristics can report false positives, especially for dynamic imports, registry lookups, or symbols referenced exclusively via strings.
- Review candidates with module owners before deleting code; prioritize entries with high confidence, large size, and no associated tests.

## Next steps

1. Convert high-confidence findings into cleanup tickets, starting with modules that are otherwise unreferenced.
2. Update or silence legitimate dynamic hooks using `# noqa: Vulture` comments to keep future scans focused on actionable debt.
3. Re-run this script after each cleanup pass to monitor progress.
