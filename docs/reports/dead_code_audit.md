# Dead code audit – 2025-09-17 14:18 UTC

*Generated via `vulture src tests --min-confidence 80 --sort-by-size`.*

## Summary

- **Total candidates**: 5
- **Command exit status**: vulture exited with status 3; review stderr below.
- **By symbol type**: 5 variable(s)
- **Top modules**: `src.operational.metrics_registry` (3), `src.core.interfaces.base` (1), `src.risk.risk_manager_impl` (1)

## Top candidates

| Confidence | Size | Type | Module | Object | Line |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | variable | src.core.interfaces.base | constraints | 76 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 66 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 69 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 75 |
| 100 | 1 | variable | src.risk.risk_manager_impl | constraints | 272 |

## Observations

- Vulture heuristics can report false positives, especially for dynamic imports, registry lookups, or symbols referenced exclusively via strings.
- Review candidates with module owners before deleting code; prioritize entries with high confidence, large size, and no associated tests.

## Triage notes – 2025-09-22

- `src.core.interfaces.base:RiskManager.propose_rebalance(constraints)` – parameter name kept for clarity;
  Protocol definition mirrors concrete implementations that expect the keyword.
- `src.operational.metrics_registry.CounterCtor/GaugeCtor/HistogramCtor` – `documentation` argument is required by the Prometheus
  constructors and retained; tracked as a known false positive in the audit log to keep future scans focused on real issues.
- `src.risk.risk_manager_impl.RiskManagerImpl.propose_rebalance(constraints)` – parameter preserved to match the public port even
  though the current adapter performs a no-op rebalance.
- Removed historical false positives: unused `_nn` type import in
  `src/intelligence/portfolio_evolution.py` and the unused `parity_module`
  import in `tests/current/test_parity_checker.py` (see Git history for details).

## Next steps

1. Convert high-confidence findings into cleanup tickets, starting with modules that are otherwise unreferenced.
2. Update or silence legitimate dynamic hooks using `# noqa: Vulture` comments to keep future scans focused on actionable debt.
3. Re-run this script after each cleanup pass to monitor progress.
