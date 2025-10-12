# Policy & Code Audit - Phase II Completion

## Scope
- src/risk: `risk_manager_impl`, `manager`, `telemetry`, `real_risk_manager`
- src/governance: `policy_ledger`, `safety_manager`, `strategy_registry`
- runtime supervision touchpoints exposed through `task_supervisor`

## Findings

### Critical Fixes
- **Fail-closed market regime detection:** `RiskManagerImpl.update_market_regime` previously propagated detector exceptions, terminating the orchestration loop and leaving risk throttles unchanged. The handler now fails closed -- logging the fault, setting the multiplier to 0, marking telemetry as blocked, and surfacing the error for governance review (see `src/risk/risk_manager_impl.py:639`).

### Risks Accepted / Follow-ups
- **Telemetry debt:** Risk telemetry still exposes legacy fields without structured severities. Recommend migrating to structured `RiskTelemetrySnapshot` schemas so governance dashboards can differentiate soft vs hard failures.
- **Policy ledger metadata hygiene:** Promotions rely on operators supplying consistent `metadata` payloads; a schema-backed check (pydantic model or jsonschema) would prevent malformed entries. Tracked for Phase III backlog.

## Tests
- `pytest --maxfail=1 --disable-warnings --cov=src/governance --cov=src/risk tests/governance tests/risk`
