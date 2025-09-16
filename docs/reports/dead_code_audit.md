# Dead code audit – 2025-09-16 17:16 UTC

*Generated via `vulture src tests --min-confidence 80 --sort-by-size`.*

## Summary

- **Total candidates**: 8
- **Command exit status**: vulture exited with status 3; review stderr below.
- **By symbol type**: 6 variable(s), 2 import(s)
- **Top modules**: `src.operational.metrics_registry` (3), `src.core.interfaces.base` (1), `src.data_foundation.config.vol_config` (1), `src.intelligence.portfolio_evolution` (1), `src.risk.risk_manager_impl` (1)

## Top candidates

| Confidence | Size | Type | Module | Object | Line |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | variable | src.core.interfaces.base | constraints | 76 |
| 100 | 1 | variable | src.data_foundation.config.vol_config | stream | 12 |
| 90 | 1 | import | src.intelligence.portfolio_evolution | _nn | 21 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 66 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 69 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 75 |
| 100 | 1 | variable | src.risk.risk_manager_impl | constraints | 272 |
| 90 | 1 | import | tests.current.test_parity_checker | parity_module | 106 |

## Observations

- Vulture heuristics can report false positives, especially for dynamic imports, registry lookups, or symbols referenced exclusively via strings.
- Review candidates with module owners before deleting code; prioritize entries with high confidence, large size, and no associated tests.

## Next steps

1. Convert high-confidence findings into cleanup tickets, starting with modules that are otherwise unreferenced.
2. Update or silence legitimate dynamic hooks using `# noqa: Vulture` comments to keep future scans focused on actionable debt.
3. Re-run this script after each cleanup pass to monitor progress.
