# TRM milestone exit drill

- Status: PASS
- Generated: 2025-10-13T04:14:50.275226+00:00
- Diary source: docs/examples/trm_exit_drill_diaries.jsonl
- Suggestions emitted: 3
- Confidence (min/avg/max): 0.65/0.65/0.65
- Suggestion artifact: docs/examples/trm_exit_drill_run/rim-suggestions-UTC-20251013T041450.jsonl
- Telemetry log: docs/examples/trm_exit_drill_run/rim-20251013.log
- Config: config/reflection/rim.config.example.yml
- Schema: interfaces/rim_types.json

## Drill components
| Component | Status | Summary |
| --- | --- | --- |
| Decision diary intake | PASS | Loaded decision diary window |
| Suggestion generation | PASS | Generated 3 suggestions |
| Schema validation | PASS | All suggestions conform to rim.v1 schema |
| Telemetry capture | PASS | Telemetry written |

## Component details
### Decision diary intake
- Entries loaded: 5

### Suggestion generation
- Aggregated strategies: carry_hedge, mean_reversion_alpha, momentum_v1

### Telemetry capture
- Runtime seconds: 0.0005

## Suggestion snapshot
| ID | Type | Confidence | Primary target | Rationale |
| --- | --- | --- | --- | --- |
| rim-20251013-0000 | WEIGHT_ADJUST | 0.65 | momentum_v1 | Average pnl -373.88 with 1.50 risk flags per trade across 2 entries |
| rim-20251013-0001 | EXPERIMENT_PROPOSAL | 0.65 | carry_hedge | Average pnl 85.60 with 1.00 risk flags per trade across 1 entries |
| rim-20251013-0002 | EXPERIMENT_PROPOSAL | 0.65 | mean_reversion_alpha | Average pnl 285.15 with 0.50 risk flags per trade across 2 entries |