# Regression backlog (Phase 7)

This backlog captures the discrete regression tickets called for in the Phase 7
roadmap stream. Update the status column as suites land so we can demonstrate
coverage improvements alongside formatter progress.

## Ticket catalog

| Ticket | Module / hotspot | Scope | Owner | Status | Validation |
| --- | --- | --- | --- | --- | --- |
| REG-001 | `src/trading/models/position.py` | Position lifecycle accounting (open → fill → close) with cap clamps and recovery paths. | Trading | Done | `tests/current/test_trading_position_accounting.py`, risk cap clamps. |
| REG-002 | `src/data_foundation/config/` | YAML override regression suite validating loader fallbacks. | Data Foundation | Done | `tests/current/test_data_foundation_config_loading.py`. |
| REG-003 | `src/operational/metrics.py` | Sanitize FIX and WHY telemetry, enforce negative guards, and bound confidence metrics. | Platform | Done | `tests/current/test_operational_metrics_sanitization.py`. |
| REG-004 | FIX execution adapters | Validate mock start-up failures and initiator fallbacks without live credentials. | Trading | Done | `tests/current/test_fix_manager_failures.py`. |
| REG-005 | Orchestration compose wiring | Smoke test optional module degradation and adapter registration. | Platform | In progress | Extend `tests/current/test_orchestration_compose.py` with degraded-module assertions. |
| REG-006 | Sensory WHY yield signal | Scenario coverage for flattening/steepening transitions and confidence scaling. | Sensory | Queued | Add direction/curvature assertions around `YieldSlopeTracker.signal()`. |
| REG-007 | `src/trading/execution/execution_engine.py` | Partial fills, retries, and reconciliation regression coverage. | Trading | Done | `tests/current/test_execution_engine.py`. |
| REG-008 | `src/risk/risk_manager_impl.py` | Drawdown throttling and risk limit updates regression coverage. | Trading | Done | `tests/current/test_risk_manager_impl.py`. |
| REG-009 | `src/trading/models/order.py` | Property-based invariants around order mutation flows. | Trading | Done | `tests/current/test_order_model_properties.py`. |

## Next steps

- Coordinate owners for REG-005/REG-006 in the sprint planning session and link
  resulting PRs back to this table.
- Capture coverage deltas in [`docs/status/ci_health.md`](ci_health.md) after
  each ticket lands to keep telemetry in sync with regression progress.
- Drop a short summary into the weekly status update once REG-005 closes so the
  orchestration smoke tests become part of the standard regression suite.

