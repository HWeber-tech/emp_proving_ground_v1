# Dead code audit – 2025-09-22 10:45 UTC

*Generated via `vulture src tests --min-confidence 80 --sort-by-size`.*

## Summary

- **Total candidates**: 21
- **Command exit status**: vulture exited with status 3; review stderr below.
- **By symbol type**: 21 variable(s)
- **Top modules**: `src.operational.metrics_registry` (3), `tests.observability.test_logging` (3), `tests.runtime.test_runtime_cli` (3), `src.runtime.predator_app` (2), `tests.tools.test_data_backbone_export` (2)

## Top candidates

| Confidence | Size | Type | Module | Object | Line |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | variable | src.core.interfaces.base | constraints | 76 |
| 100 | 1 | variable | src.data_foundation.streaming.kafka_stream | request_timeout | 427 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 69 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 75 |
| 100 | 1 | variable | src.operational.metrics_registry | documentation | 83 |
| 100 | 1 | variable | src.risk.risk_manager_impl | constraints | 427 |
| 100 | 1 | variable | src.runtime.predator_app | exc_type | 250 |
| 100 | 1 | variable | src.runtime.predator_app | tb | 250 |
| 100 | 1 | variable | tests.data_foundation.test_kafka_stream | request_timeout | 153 |
| 100 | 1 | variable | tests.observability.test_logging | reset_logging | 38 |
| 100 | 1 | variable | tests.observability.test_logging | reset_logging | 61 |
| 100 | 1 | variable | tests.observability.test_logging | reset_logging | 76 |
| 100 | 1 | variable | tests.runtime.test_runtime_cli | cli_env | 29 |
| 100 | 1 | variable | tests.runtime.test_runtime_cli | cli_env | 51 |
| 100 | 1 | variable | tests.runtime.test_runtime_cli | cli_env | 68 |
| 100 | 1 | variable | tests.tools.test_data_backbone_export | exc_type | 17 |
| 100 | 1 | variable | tests.tools.test_data_backbone_export | tb | 17 |
| 100 | 1 | variable | tests.tools.test_operational_export | exc_type | 19 |
| 100 | 1 | variable | tests.tools.test_operational_export | tb | 19 |
| 100 | 1 | variable | tests.tools.test_risk_compliance_export | exc_type | 17 |

## Observations

- Vulture heuristics can report false positives, especially for dynamic imports, registry lookups, or symbols referenced exclusively via strings.
- Review candidates with module owners before deleting code; prioritize entries with high confidence, large size, and no associated tests.

## Next steps

1. Convert high-confidence findings into cleanup tickets, starting with modules that are otherwise unreferenced.
2. Update or silence legitimate dynamic hooks using `# noqa: Vulture` comments to keep future scans focused on actionable debt.
3. Re-run this script after each cleanup pass to monitor progress.
