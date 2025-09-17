# Regression coverage plan – 2025-09-16 follow-up

The [CI baseline snapshot](../ci_baseline_report.md) highlighted four modules
whose coverage remains below the repository average. The table below breaks the
hotspots into reviewable tickets so teams can parallelize the backfill work.

| Ticket | Scope | Focused additions | Acceptance criteria | Priority |
| --- | --- | --- | --- | --- |
| RC-1 | `src/operational/metrics.py` | Unit tests that exercise fallback sink wiring, registry mutation hooks, and the Prometheus opt-in guard rails. | Failing sink imports raise actionable errors, registry sinks emit expected labels, and telemetry toggles stay idempotent across repeated registrations. | High |
| RC-2 | `src/trading/models/position.py` | Scenario tests covering lifecycle helpers (open/close transitions), serialization round-trips, and defensive copies returned by getters. | Each helper method has assertions for both success and failure paths; serialization reproduces prior state after `from_dict` / `to_dict` cycles. | High |
| RC-3 | `src/data_foundation/config/` modules | Regression tests for YAML parsing fallbacks, environment overrides, and the `vol_config` normalization helpers. | Missing files and malformed payloads surface informative errors while happy-path loads produce stable objects matching fixture snapshots. | Medium |
| RC-4 | `src/sensory/dimensions/why/yield_signal.py` | Property-based tests covering feature scaling, smoothing windows, and branch-specific signal construction. | Each branch (baseline, variance, stabilization) is asserted with synthetic inputs and edge-case parameters without relying on integration fixtures. | Medium |

### Next steps

1. Create individual GitHub issues (or Linear tickets) for `RC-1` through `RC-4`
   and link them back to this plan so ownership is explicit.
2. Use parametrized pytest cases to keep the regression suites compact while
   covering the branching behavior noted above.
3. Update [`docs/status/ci_health.md`](ci_health.md) with the new coverage
   percentages once each ticket lands so the dashboard remains authoritative.
